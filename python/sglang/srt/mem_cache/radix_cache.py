from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import random
import sys
import threading
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.mem_cache.base_prefix_cache import (
    DEFAULT_TENANT_ID,
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.utils import (
    get_eviction_strategy,
    get_hash_str,
    split_node_hash_value,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
    ):
        # token ids sequence (raw ints in both modes)
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram
        # Optional cap on raw tokens: behave as if token_ids were sliced to
        # token_ids[:limit], without the O(n) copy. None = use all tokens.
        self.limit = limit

    def _raw_len(self) -> int:
        n = len(self.token_ids)
        if self.limit is not None and self.limit < n:
            return self.limit
        return n

    def raw_token_ids(self) -> array:
        """token_ids honoring `limit` (copies only when capped)."""
        n = self._raw_len()
        t = self.token_ids
        return t if n == len(t) else t[:n]

    def __len__(self) -> int:
        n = self._raw_len()
        if self.is_bigram:
            return n - 1 if n > 0 else 0
        return n

    # TODO(Jialin): vectorize with numpy without PyLong boxing
    def __iter__(self) -> Iterator:
        t = self.token_ids
        n = self._raw_len()
        if self.is_bigram:
            for i in range(n - 1 if n > 0 else 0):
                yield (t[i], t[i + 1])
        elif n == len(t):
            yield from t
        else:
            for i in range(n):
                yield t[i]

    def __getitem__(self, idx: Union[int, slice]) -> RadixKey:
        # Normalize int -> 1-element slice so the rest handles one shape.
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigrams [start, stop) span raw tokens [start, stop + 1);
            # empty slice -> empty raw tokens (not a dangling boundary token).
            raw = self.token_ids[start : stop + 1] if stop > start else array("q")
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> RadixKey:
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: RadixKey) -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )

    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(len(t0), len(t1))

        # Exponential search for the first diverging token: gallop in doubling
        # windows (one C-level slice compare each), then binary-search the window
        # holding the divergence -- no per-token Python loop on long shared prefixes.
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2

        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        matched_tokens = min(matched_tokens, len(self), len(other))
        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size

    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str] = None) -> str:
        """SHA256 for logical units [start, end); bigram mode feeds overlapping (t_i, t_{i+1}) byte pairs."""
        hash_value = get_hash_str(self[start:end], prior_hash)
        assert isinstance(hash_value, str)
        return hash_value


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        self.write_through_pending_id: Optional[int] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
        # Back-pointer used by tenant-aware wrappers to dispatch lock/evict
        # operations to the tree that owns this node.
        self.owner_cache: Optional[RadixCache] = None

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


class RadixCache(KVCacheEventMixin, BasePrefixCache):
    # A prompt is only promoted into the shared global cache once it has been
    # requested by at least MIN_TO_RAISE distinct users. Beyond that the
    # promotion is rolled probabilistically per tracked insert: with `n`
    # distinct users the chance is (n - MIN_TO_RAISE) / PROMOTION_ODDS_WINDOW,
    # i.e. 1% at MIN_TO_RAISE + 1, 2% at MIN_TO_RAISE + 2, ..., 100% at
    # MIN_TO_RAISE + PROMOTION_ODDS_WINDOW. A low-frequency prompt requested by
    # one user thus stays invisible (a full cache miss) to everyone else while
    # a globally-popular prompt eventually becomes a public cache hit.
    MIN_TO_RAISE = 5
    PROMOTION_ODDS_WINDOW = 100

    def __init__(self, params: CacheInitParams, *, _is_personal_cache: bool = False):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()
        self._is_personal_cache = _is_personal_cache
        self._multi_tenant_enabled = (
            params.enable_multi_tenant_cache and not _is_personal_cache
        )
        self._enable_debug_log = params.enable_radix_cache_debug_log
        self._state_lock = threading.RLock() if self._multi_tenant_enabled else None

        # Multi-tenant state (used by the top-level/global cache only).
        self.personal_caches: Dict[str, RadixCache] = {}
        self.prompt_request_tracker: Dict[Tuple[Optional[str], bool, Tuple[int, ...]], Set[str]] = defaultdict(set)
        self.promoted_prompt_keys: Set[Tuple[Optional[str], bool, Tuple[int, ...]]] = set()
        # node.id -> prompt_key: lets eviction purge `promoted_prompt_keys` when the
        # promoted node's slots leave the global tree so the prompt can be promoted
        # again later.
        self.promoted_prompt_node_ids: Dict[
            int, Tuple[Optional[str], bool, Tuple[int, ...]]
        ] = {}

        self.kv_event_queue = []

        if params.enable_metrics and not self._is_personal_cache:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
        enable_multi_tenant_cache: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
            enable_multi_tenant_cache=enable_multi_tenant_cache,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        if self._multi_tenant_enabled:
            self.personal_caches.clear()
            self.prompt_request_tracker.clear()
            self.promoted_prompt_keys.clear()
            self.promoted_prompt_node_ids.clear()

        self._reset_single_tree_state()

    def _reset_single_tree_state(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.owner_cache = self
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if not self._multi_tenant_enabled:
            return self._match_prefix_single(params)

        user_id = self._resolve_user_id(params.user_id, params.req)
        with self._state_lock:
            # Global tree first: it owns the shared (promoted) prefix slots.
            # A personal cache stores, for post-promotion requests, only the
            # suffix beyond the global match point, so matching personal-first
            # (from position 0) could never re-find those suffix slots and
            # would under-match -- and trip cache_unfinished_req's re-match
            # invariant -- whenever a request is partly served by the global
            # tree (e.g. a chunked prefill that promoted its first chunk).
            global_result = self._match_prefix_single(params)
            global_len = len(global_result.device_indices)

            personal_cache = self.personal_caches.get(user_id)
            if personal_cache is None:
                return global_result

            if global_len == 0:
                # Nothing shared: fall back to the user's own full-key match.
                personal_result = personal_cache._match_prefix_single(params)
                if self._has_match(personal_result):
                    return personal_result
                return global_result

            # Continue from the global match point inside the personal cache,
            # whose post-promotion entries are stored as position-shifted
            # suffixes `key[global_len:]`.
            key = params.key
            key, _ = key.maybe_to_bigram_view(self.is_eagle)
            key = key.page_aligned(self.page_size)
            if global_len >= len(key):
                return global_result

            personal_result = personal_cache._match_prefix_single(
                MatchPrefixParams(key=key[global_len:])
            )
            personal_len = len(personal_result.device_indices)
            if personal_len == 0:
                return global_result

            return MatchResult(
                device_indices=torch.cat(
                    [global_result.device_indices, personal_result.device_indices]
                ),
                last_device_node=personal_result.last_device_node,
                last_host_node=personal_result.last_host_node,
                best_match_node=personal_result.best_match_node,
            )

    def _match_prefix_single(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0).
            ``last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        if self.disable or len(key) == 0:
            return self._empty_match_result

        key = key.page_aligned(self.page_size)

        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if not self._multi_tenant_enabled:
            return self._insert_single(params)

        user_id = self._resolve_user_id(params.user_id)
        with self._state_lock:
            # Single-owner invariant: a physical KV slot must be owned by exactly
            # one tree. The global tree owns the slots of promoted prefixes, so a
            # request served by the global tree must not re-store those shared
            # slots into a personal cache (it would double-own them and trip the
            # pool accounting / leak check).
            if params.key is not None:
                global_match = self._match_prefix_single(
                    MatchPrefixParams(key=params.key)
                )
                global_match_len = len(global_match.device_indices)
                aligned_key, _ = params.key.maybe_to_bigram_view(self.is_eagle)
                aligned_key = aligned_key.page_aligned(self.page_size)
                key_len = len(aligned_key)
                if key_len > 0 and global_match_len >= key_len:
                    # Fully served by the global tree: nothing to store.
                    if self._enable_debug_log:
                        logger.warning(
                            f"[DBG] user_id={user_id} key_len={key_len} "
                            f"global_match_len={global_match_len} hit_global=True"
                        )
                    return InsertResult(
                        prefix_len=global_match_len,
                        last_device_node=global_match.last_device_node,
                    )
            else:
                global_match_len = 0
                aligned_key = None
                key_len = 0

            personal_cache = self._get_or_create_personal_cache(user_id)

            if params.key is not None and global_match_len > 0:
                # Partially served by the global tree: store only the suffix the
                # global tree does not own, so each slot keeps exactly one owner.
                # global_match_len is page-aligned, so the boundary stays on a
                # page boundary and the returned prefix_len is consistent with
                # the caller's freed range.
                truncated_value = (
                    params.value[global_match_len:]
                    if params.value is not None
                    else None
                )
                personal_result = personal_cache._insert_single(
                    InsertParams(
                        key=aligned_key[global_match_len:],
                        value=truncated_value,
                        chunked=params.chunked,
                        priority=params.priority,
                    )
                )
                if self._enable_debug_log:
                    logger.warning(
                        f"[DBG] user_id={user_id} key_len={key_len} "
                        f"global_match_len={global_match_len} truncated=True "
                        f"track={params.track_miss_for_promotion}"
                    )
                if params.track_miss_for_promotion and params.key is not None:
                    self._try_promote_prompt(
                        params=params,
                        user_id=user_id,
                        personal_cache=personal_cache,
                        personal_result=personal_result,
                        global_match_len=global_match_len,
                        global_match_device_indices=global_match.device_indices,
                        aligned_key=aligned_key,
                    )
                return InsertResult(
                    prefix_len=global_match_len + personal_result.prefix_len,
                    last_device_node=personal_result.last_device_node,
                )

            personal_result = personal_cache._insert_single(params)
            if self._enable_debug_log:
                logger.warning(
                    f"[DBG] user_id={user_id} key_len={key_len} "
                    f"global_match_len={global_match_len} "
                    f"track={params.track_miss_for_promotion} "
                    f"evictable={self.evictable_size()} "
                    f"personal_tokens={personal_cache._total_size_helper()}"
                )

            if params.track_miss_for_promotion and params.key is not None:
                self._try_promote_prompt(
                    params=params,
                    user_id=user_id,
                    personal_cache=personal_cache,
                    personal_result=personal_result,
                )

            return personal_result

    def _try_promote_prompt(
        self,
        *,
        params: InsertParams,
        user_id: int,
        personal_cache: "RadixCache",
        personal_result: InsertResult,
        global_match_len: int = 0,
        global_match_device_indices: Optional[torch.Tensor] = None,
        aligned_key: Optional[RadixKey] = None,
    ) -> None:
        """Count a tracked miss toward this prompt's distinct-user tally and, once
        the threshold is reached, move the slots the personal tree just stored
        into the global tree so every user of the prompt shares them.

        When global_match_len > 0 the request was partially served by the global
        tree and only the suffix belongs to the personal tree (truncated insert):
        only that suffix is promoted, re-inserted below the global prefix the
        request already matched, preserving the single-owner invariant.
        """
        prompt_key = self._build_prompt_tracker_key(params.key)
        requestors = self.prompt_request_tracker[prompt_key]
        requestors.add(user_id)
        if (
            self._should_promote(len(requestors))
            and prompt_key not in self.promoted_prompt_keys
        ):
            self.promoted_prompt_keys.add(prompt_key)
            if global_match_len > 0:
                # The personal tree only owns the suffix: re-match it for the
                # valid slots, then prepend the global prefix's slots so the
                # full-key insert walks the existing global prefix and stores
                # the suffix below it instead of as a sibling branch.
                suffix_value = personal_cache._match_prefix_single(
                    MatchPrefixParams(key=aligned_key[global_match_len:])
                ).device_indices
                promoted_value = torch.cat(
                    [global_match_device_indices, suffix_value]
                )
            else:
                # Promote the slots the personal tree actually owns (valid at
                # completion), not the request's raw value which may already be
                # freed for a chunked/growth insert.
                promoted_value = personal_cache._match_prefix_single(
                    MatchPrefixParams(key=params.key)
                ).device_indices
            promoted_insert = self._insert_single(
                InsertParams(
                    key=params.key,
                    value=promoted_value,
                    chunked=params.chunked,
                    priority=params.priority,
                )
            )
            if promoted_insert.last_device_node is not None:
                # Track the node owning the promoted slots so eviction can purge
                # `promoted_prompt_keys` when they leave the tree.
                self.promoted_prompt_node_ids[
                    promoted_insert.last_device_node.id
                ] = prompt_key
            # Ownership of the freshly-stored slots now belongs to the global
            # tree: detach the personal node WITHOUT freeing its slots so each
            # slot has exactly one owner.
            last_node = personal_result.last_device_node
            if last_node is not None and last_node in personal_cache.evictable_leaves:
                personal_cache._delete_leaf(last_node)

    def _insert_single(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len, last_node = self._insert_helper(
            self.root_node, key, value, priority, chunked
        )
        return InsertResult(prefix_len=prefix_len, last_device_node=last_node)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        if self.disable:
            # The protected prefix is not this req's to free.
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.cache_protected_len : kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free_segment(
                kv_indices, start_pos=req.cache_protected_len
            )
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = req.priority or 0
            result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    priority=priority,
                    user_id=req.user_id or req.session_id,
                    track_miss_for_promotion=(
                        self._build_prompt_tracker_key(radix_key)
                        not in self.promoted_prompt_keys
                    ),
                )
            )
            freed_end = result.prefix_len
        else:
            freed_end = key_len

        # duplicates / uninserted range, then the unaligned tail
        self.token_to_kv_pool_allocator.free_segments(
            [
                (
                    kv_indices[req.cache_protected_len : freed_end],
                    req.cache_protected_len,
                ),
                (kv_indices[key_len:], key_len),
            ]
        )

        # Remove req slot release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=req.priority or 0,
                user_id=req.user_id or req.session_id,
                track_miss_for_promotion=(
                    req.cache_protected_len == 0
                    and self._build_prompt_tracker_key(radix_key)
                    not in self.promoted_prompt_keys
                ),
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free_segment(
            kv_indices[req.cache_protected_len : new_prefix_len],
            start_pos=req.cache_protected_len,
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            MatchPrefixParams(key=radix_key, user_id=req.user_id or req.session_id)
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        if self._multi_tenant_enabled:
            print("global cache")
            self._print_helper(self.root_node, 2)
            for user_id, personal_cache in self.personal_caches.items():
                print(f"personal cache user_id={user_id}")
                personal_cache._print_helper(personal_cache.root_node, 2)
            print(f"#tokens: {self.total_size()}")
            return

        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        if self._multi_tenant_enabled:
            total_size = self._total_size_helper()
            for personal_cache in self.personal_caches.values():
                total_size += personal_cache._total_size_helper()
            return total_size
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self._multi_tenant_enabled:
            return self._evict_multi_tenant(params)

        return self._evict_single(params)

    def _evict_single(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # Tree values are page-aligned copies of a kv row: page-exact segment.
            self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_multi_tenant(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        num_evicted = 0

        with self._state_lock:
            eviction_heap = []
            for cache in self._iter_all_trees():
                for node in cache.evictable_leaves:
                    heapq.heappush(
                        eviction_heap,
                        (cache.eviction_strategy.get_priority(node), node, cache),
                    )

            while num_evicted < num_tokens and eviction_heap:
                _priority, node, owner_cache = heapq.heappop(eviction_heap)

                if node.evicted or node.lock_ref > 0:
                    continue

                owner_cache.token_to_kv_pool_allocator.free_segment(node.value, start_pos=0)
                num_evicted += len(node.value)
                owner_cache._delete_leaf(node)

                parent = node.parent
                if parent is not None and len(parent.children) == 0 and parent.lock_ref == 0:
                    heapq.heappush(
                        eviction_heap,
                        (
                            owner_cache.eviction_strategy.get_priority(parent),
                            parent,
                            owner_cache,
                        ),
                    )

                owner_cache._record_remove_event(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self._multi_tenant_enabled:
            owner_cache = node.owner_cache
            if owner_cache is None:
                return IncLockRefResult(delta=0)
            return owner_cache._inc_lock_ref_single(node)

        return self._inc_lock_ref_single(node)

    def _inc_lock_ref_single(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self._multi_tenant_enabled:
            owner_cache = node.owner_cache
            if owner_cache is None:
                return DecLockRefResult(delta=0)
            return owner_cache._dec_lock_ref_single(node)

        return self._dec_lock_ref_single(node)

    def _dec_lock_ref_single(self, node: TreeNode) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert node is self.root_node, "This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        if self._multi_tenant_enabled:
            total = self.evictable_size_
            for personal_cache in self.personal_caches.values():
                total += personal_cache.evictable_size_
            return total
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        if self._multi_tenant_enabled:
            total = self.protected_size_
            for personal_cache in self.personal_caches.values():
                total += personal_cache.protected_size_
            return total
        return self.protected_size_

    def all_values_flatten(self):
        if self._multi_tenant_enabled:
            values = []
            for cache in self._iter_all_trees():
                for child in cache.root_node.children.values():
                    values.append(child.value)
                    cache._collect_values_dfs(child, values)
            return torch.cat(values) if values else torch.empty((0,), dtype=torch.int64)

        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    def _collect_values_dfs(self, node: TreeNode, values: list):
        for child in node.children.values():
            values.append(child.value)
            self._collect_values_dfs(child, values)

    ##### Internal Helper Functions #####

    def _resolve_user_id(
        self, explicit_user_id: Optional[str], req: Optional[Req] = None
    ) -> str:
        if explicit_user_id is not None and explicit_user_id != "":
            return explicit_user_id
        if req is not None:
            if req.user_id is not None and req.user_id != "":
                return req.user_id
            if req.session_id is not None and req.session_id != "":
                return req.session_id
        # No identity at all (e.g. raw /generate requests): fall back to the
        # anonymous tenant so no request can write to or probe the global tree
        # without first meeting the multi-user promotion odds.
        return DEFAULT_TENANT_ID

    def _has_match(self, result: MatchResult) -> bool:
        return len(result.device_indices) > 0 or result.host_hit_length > 0

    def _get_or_create_personal_cache(self, user_id: str) -> RadixCache:
        personal_cache = self.personal_caches.get(user_id)
        if personal_cache is not None:
            return personal_cache

        params = CacheInitParams(
            disable=self.disable,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=self.page_size,
            enable_kv_cache_events=self.enable_kv_cache_events,
            enable_metrics=False,
            is_eagle=self.is_eagle,
            disable_finished_insert=self.disable_finished_insert,
            eviction_policy=self.eviction_policy,
        )
        personal_cache = RadixCache(params, _is_personal_cache=True)
        personal_cache.kv_event_queue = self.kv_event_queue
        self.personal_caches[user_id] = personal_cache
        return personal_cache

    def _build_prompt_tracker_key(
        self, key: Optional[RadixKey]
    ) -> Tuple[Optional[str], bool, Tuple[int, ...]]:
        if key is None:
            return (None, False, ())

        is_bigram = key.is_bigram or self.is_eagle
        normalized_key = RadixKey(
            token_ids=key.raw_token_ids(),
            extra_key=key.extra_key,
            is_bigram=is_bigram,
        ).page_aligned(self.page_size)
        token_ids = tuple(normalized_key.raw_token_ids())
        return (normalized_key.extra_key, normalized_key.is_bigram, token_ids)

    def _get_min_to_raise(self) -> int:
        min_to_raise = self.MIN_TO_RAISE
        if min_to_raise is None:
            return 5
        min_to_raise = int(min_to_raise)
        if min_to_raise <= 0:
            return 1
        return min_to_raise

    def _should_promote(self, num_requestors: int) -> bool:
        """Whether `num_requestors` distinct users suffice to promote a prompt.

        Below MIN_TO_RAISE distinct users a prompt is never promoted. Above
        that, promotion is rolled on each tracked insert with probability
        (num_requestors - MIN_TO_RAISE) / PROMOTION_ODDS_WINDOW. This is the
        exact equivalent of rolling a uniform integer in (MIN_TO_RAISE,
        MIN_TO_RAISE + PROMOTION_ODDS_WINDOW] and promoting only when the roll
        is <= num_requestors: 1% at MIN_TO_RAISE + 1, 2% at MIN_TO_RAISE + 2,
        ..., 100% at MIN_TO_RAISE + PROMOTION_ODDS_WINDOW.
        """
        min_to_raise = self._get_min_to_raise()
        if num_requestors <= min_to_raise:
            return False
        if num_requestors >= min_to_raise + self.PROMOTION_ODDS_WINDOW:
            return True
        return (
            random.random()
            < (num_requestors - min_to_raise) / self.PROMOTION_ODDS_WINDOW
        )

    def _iter_all_trees(self) -> list[RadixCache]:
        return [self] + list(self.personal_caches.values())

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = key.child_key(self.page_size)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.owner_cache = self
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0, node

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.owner_cache = self
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
            node = new_node
        return total_prefix_length, node

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _delete_leaf(self, node):
        # A promoted node leaving the global tree means its slots are freed: drop
        # the guard so the prompt can be promoted again on future full misses.
        prompt_key = self.promoted_prompt_node_ids.pop(node.id, None)
        if prompt_key is not None:
            self.promoted_prompt_keys.discard(prompt_key)

        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5, 6, 7]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [8, 9, 10, 11, 12]))))
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 13, 14])))
        )
    )
