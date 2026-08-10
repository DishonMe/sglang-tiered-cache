"""
Unit tests for the RadixCache implementation.

This module tests the core functionality of RadixCache, RadixKey, and TreeNode
following SGLang testing patterns.

Test Coverage:
- RadixKey: token ID management, slicing, iteration, representation
- TreeNode: node properties, reference counting, hash values
- RadixCache: insert/match operations, eviction, page alignment, error handling
- Cache events and request handling
- Boundary conditions with parameterized testing

Usage:
    python test_radix_cache_unit.py
    python -m pytest test_radix_cache_unit.py -v
    python -m pytest test_radix_cache_unit.py::TestRadixCache::test_insert_basic
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import random
import unittest
import unittest.mock
from array import array

import torch

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored
from sglang.srt.mem_cache.base_prefix_cache import (
    DEFAULT_TENANT_ID,
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.mamba_radix_cache import TreeNode as MambaTreeNode
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.utils import get_device

# Test constants
DEFAULT_PAGE_SIZE = 4


class TestRadixKey(unittest.TestCase):
    """Test cases for RadixKey class."""

    def test_init_with_extra_key(self):
        """Test initialization with extra_key."""
        token_ids = [1, 2, 3]
        extra_key = "test_key"
        key = RadixKey(array("q", token_ids), extra_key)
        self.assertEqual(list(key.token_ids), token_ids)
        self.assertEqual(key.extra_key, extra_key)

    def test_len_and_iter(self):
        """Test __len__ and __iter__ methods."""
        test_cases = [
            ([1, 2, 3], 3),
            ([], 0),
            ([42], 1),
        ]

        for tokens, expected in test_cases:
            with self.subTest(tokens=tokens):
                key = RadixKey(array("q", tokens))
                self.assertEqual(len(key), expected)
                self.assertEqual(list(key), tokens)

    def test_getitem_int(self):
        """Test __getitem__ with int index."""
        test_cases = [
            ([10, 20, 30], 0, [10]),
            ([10, 20, 30], -1, [30]),
            ([10, 20, 30], 2, [30]),
        ]

        for tokens, index, expected in test_cases:
            with self.subTest(tokens=tokens, index=index):
                key = RadixKey(array("q", tokens))
                result = key[index]
                self.assertIsInstance(result, RadixKey)
                self.assertEqual(list(result.token_ids), expected)

    def test_getitem_slice(self):
        """Test __getitem__ with slice and edge cases."""
        key = RadixKey(array("q", [1, 2, 3, 4, 5]), "extra")

        # Basic slice
        sliced = key[1:4]
        self.assertIsInstance(sliced, RadixKey)
        self.assertEqual(list(sliced.token_ids), [2, 3, 4])
        self.assertEqual(sliced.extra_key, "extra")

        # Edge cases
        self.assertEqual(list(key[2:2].token_ids), [])  # Empty slice
        self.assertEqual(list(key[:].token_ids), [1, 2, 3, 4, 5])  # Full slice

    def test_getitem_invalid_index(self):
        """Test __getitem__ with invalid indices."""
        key = RadixKey(array("q", [1, 2, 3]))
        with self.assertRaises(IndexError):
            _ = key[10]  # Out of bounds

    def _assert_match(self, a, b, page_size, expected, is_bigram=False):
        key_a = RadixKey(array("q", a), is_bigram=is_bigram)
        key_b = RadixKey(array("q", b), is_bigram=is_bigram)
        self.assertEqual(key_a.match(key_b, page_size=page_size), expected)

    def test_match_page_size_1(self):
        """match() with page_size=1: full, partial, none, prefix, and empty keys."""
        self._assert_match([1, 2, 3, 4], [1, 2, 3, 4], 1, 4)  # identical
        self._assert_match([1, 2, 3, 4], [1, 2, 9, 9], 1, 2)  # diverge at index 2
        self._assert_match([9, 2, 3], [1, 2, 3], 1, 0)  # diverge at index 0
        self._assert_match([1, 2, 3, 4], [1, 2, 3], 1, 3)  # other is a prefix
        self._assert_match([], [1, 2], 1, 0)  # empty self
        self._assert_match([1, 2], [], 1, 0)  # empty other
        self._assert_match([], [], 1, 0)  # both empty

    def test_match_page_size_gt_1_rounds_down(self):
        """match() with page_size>1 rounds the shared length down to a page."""
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 9, 8], 4, 4)
        self._assert_match(
            [1, 2, 3, 4], [1, 9, 3, 4], 4, 0
        )  # diverge inside first page
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 9, 6, 7, 8], 4, 4)
        self._assert_match([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], 4, 8)
        self._assert_match([1, 2, 3], [1, 2, 3], 4, 0)  # shorter than one page

    def test_match_long_keys_exponential_search(self):
        """Deep divergences exercise the doubling gallop windows + binary search.

        ``base`` has distinct values, so flipping one position diverges the prefix
        exactly there; the shared length is that index rounded down to the page.
        """
        base = list(range(2000))
        for div in (1, 2, 63, 64, 65, 127, 128, 511, 512, 513, 1234, 1999):
            b = base[:]
            b[div] = -1
            for page_size in (1, 4, 64):
                with self.subTest(div=div, page_size=page_size):
                    self._assert_match(
                        base, b, page_size, (div // page_size) * page_size
                    )
        # Full match of a long key: the gallop must reach the end.
        self._assert_match(base, base[:], 64, (2000 // 64) * 64)

    def test_match_bigram(self):
        """is_bigram: L matching raw tokens imply L-1 matching bigrams."""
        self._assert_match([1, 2, 3, 4, 5], [1, 2, 3, 9, 5], 1, 2, is_bigram=True)
        self._assert_match([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1, 4, is_bigram=True)
        self._assert_match([1, 2], [1, 2], 1, 1, is_bigram=True)
        # Raw diverge at token 70 -> 69 matching bigrams -> rounded down to 64.
        long_a = list(range(130))
        long_b = list(range(130))
        long_b[70] = -1
        self._assert_match(long_a, long_b, 64, 64, is_bigram=True)


class TestTreeNode(unittest.TestCase):
    """Test cases for TreeNode class."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_init_basic(self):
        """Test basic initialization of TreeNode."""
        node = TreeNode()
        self.assertEqual(node.id, 0)
        self.assertEqual(len(node.children), 0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.key)
        self.assertIsNone(node.value)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(node.hit_count, 0)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertIsNone(node.host_value)
        self.assertIsNone(node.hash_value)

    def test_init_with_id(self):
        """Test initialization with custom ID."""
        node = TreeNode(id=42)
        self.assertEqual(node.id, 42)
        node2 = TreeNode()
        self.assertEqual(node2.id, 1)  # Counter was incremented

    def test_evicted_backuped_properties(self):
        """Test evicted and backuped properties."""
        test_cases = [
            (False, False, True, False),
            (True, False, False, False),
            (True, True, False, True),
            (False, True, True, True),
        ]

        for (
            has_value,
            has_host_value,
            expected_evicted,
            expected_backuped,
        ) in test_cases:
            with self.subTest(has_value=has_value, has_host_value=has_host_value):
                node = TreeNode()

                if has_value:
                    node.value = torch.tensor([1, 2, 3])
                if has_host_value:
                    node.host_value = torch.tensor([4, 5, 6])

                self.assertEqual(node.evicted, expected_evicted)
                self.assertEqual(node.backuped, expected_backuped)

    def test_protect_release_host(self):
        """Test protect_host and release_host methods."""
        node = TreeNode()
        self.assertEqual(node.host_ref_counter, 0)

        node.protect_host()
        self.assertEqual(node.host_ref_counter, 1)

        node.release_host()
        self.assertEqual(node.host_ref_counter, 0)

        # Test error case
        with self.assertRaises(RuntimeError):
            node.release_host()

    def test_get_last_hash_value(self):
        """Test get_last_hash_value method."""
        node = TreeNode()
        self.assertIsNone(node.get_last_hash_value())

        node.hash_value = ["hash1", "hash2", "hash3"]
        self.assertEqual(node.get_last_hash_value(), "hash3")

    def test_get_prefix_hash_values_not_shared_across_calls(self):
        """Regression guard for cached mutable prefix hash lists."""
        for node_cls in (TreeNode, MambaTreeNode):
            with self.subTest(node_cls=node_cls.__module__):
                root = node_cls()
                n1 = node_cls()
                n1.parent = root
                n1.hash_value = ["h1"]
                n2 = node_cls()
                n2.parent = n1
                n2.hash_value = ["h2"]
                n3 = node_cls()
                n3.parent = n2
                n3.hash_value = ["h3"]

                first = n3.get_prefix_hash_values(n2)
                self.assertEqual(first, ["h1", "h2"])

                # Downstream storage code extends prefix_keys in place while
                # processing pages. A cached list must not be observable by a
                # later call.
                first += ["h3"]

                second = n3.get_prefix_hash_values(n2)
                self.assertEqual(second, ["h1", "h2"])
                self.assertIsNot(second, first)

                n4 = node_cls()
                n4.parent = n3
                n4.hash_value = ["h4"]
                self.assertEqual(n4.get_prefix_hash_values(n3), ["h1", "h2", "h3"])


class TestRadixCache(unittest.TestCase):
    """Test cases for RadixCache class."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_init_variations(self):
        """Test cache initialization with different parameters."""
        test_cases = [
            (1, False, False),
            (4, False, True),
            (1, True, False),
        ]

        for page_size, disable, enable_events in test_cases:
            with self.subTest(
                page_size=page_size, disable=disable, enable_events=enable_events
            ):
                cache = RadixCache.create_simulated(
                    disable=disable,
                    page_size=page_size,
                    enable_kv_cache_events=enable_events,
                )

                self.assertEqual(cache.page_size, page_size)
                self.assertEqual(cache.disable, disable)
                self.assertEqual(cache.enable_kv_cache_events, enable_events)
                self.assertEqual(cache.device, torch.device("cpu"))
                self.assertIsNotNone(cache.root_node)
                self.assertEqual(len(cache.root_node.key), 0)

    def test_reset(self):
        """Test reset method."""
        cache = RadixCache.create_simulated()

        # Insert some data
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        self.assertGreater(cache.total_size(), 0)

        # Reset
        cache.reset()
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_insert_and_match_basic(self):
        """Test basic insert and match operations."""
        for disable_cache in [False, True]:
            with self.subTest(disable_cache=disable_cache):
                cache = RadixCache.create_simulated(disable=disable_cache)

                key = RadixKey(array("q", [1, 2, 3]))
                value = torch.tensor([10, 20, 30], dtype=torch.int64)
                result = cache.insert(InsertParams(key=key, value=value))
                prefix_len = result.prefix_len

                if disable_cache:
                    self.assertEqual(prefix_len, 0)
                    self.assertEqual(cache.total_size(), 0)
                    continue

                self.assertEqual(prefix_len, 0)  # No existing prefix
                self.assertEqual(cache.total_size(), 3)
                self.assertEqual(cache.evictable_size(), 3)

                # Test match_prefix
                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])))
                )
                self.assertEqual(len(result.device_indices), 3)
                torch.testing.assert_close(result.device_indices, value)

                # Test partial match
                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2])))
                )
                self.assertEqual(len(result.device_indices), 2)
                torch.testing.assert_close(
                    result.device_indices, torch.tensor([10, 20], dtype=torch.int64)
                )

    def test_insert_with_none_value(self):
        """Test insert with None value (should use token_ids as list)."""
        cache = RadixCache.create_simulated()

        key = RadixKey(array("q", [1, 2, 3]))
        result = cache.insert(InsertParams(key=key, value=None))
        prefix_len = result.prefix_len

        # When None is passed, it should create value from token_ids
        self.assertEqual(prefix_len, 0)
        self.assertEqual(cache.total_size(), 3)

    def test_total_size(self):
        """Test total_size calculation."""
        cache = RadixCache.create_simulated()

        self.assertEqual(cache.total_size(), 0)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        self.assertEqual(cache.total_size(), 3)

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [4, 5])),
                value=torch.tensor([40, 50], dtype=torch.int64),
            )
        )
        self.assertEqual(cache.total_size(), 5)

    def test_kv_cache_events(self):
        """Test KV cache events functionality."""
        test_cases = [
            (1, True),
            (2, True),
            (1, False),
        ]

        for page_size, enable_events in test_cases:
            with self.subTest(page_size=page_size, enable_events=enable_events):
                cache = RadixCache.create_simulated(
                    page_size=page_size, enable_kv_cache_events=enable_events
                )

                # Insert data
                cache.insert(
                    InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5])), value=None)
                )

                # Take events
                events = cache.take_events()

                if enable_events:
                    self.assertGreater(len(events), 0)
                    # Verify events include BlockStored events (there might be other event types)
                    block_stored_events = [
                        e for e in events if isinstance(e, BlockStored)
                    ]
                    self.assertGreater(len(block_stored_events), 0)
                    for event in block_stored_events:
                        self.assertLessEqual(len(event.token_ids), page_size)
                else:
                    self.assertEqual(len(events), 0)

    def test_kv_cache_events_with_eviction(self):
        """Test KV cache events include removal events."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(
            mock_allocator=mock_allocator,
            page_size=2,
            enable_kv_cache_events=True,
        )

        # Insert and then evict data
        seq = [1, 2, 3, 4]
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
            )
        )
        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(
            result.num_tokens_evicted,
            len(seq),
            f"evicted {result.num_tokens_evicted} tokens, expected at least {len(seq)}",
        )

        # Take events - should include both store and remove events
        events = cache.take_events()
        self.assertGreater(len(events), 0)

        # Check event types
        event_types = [type(event).__name__ for event in events]
        self.assertIn("BlockStored", event_types)

        stored_hashes = [
            event.block_hashes[0] for event in events if isinstance(event, BlockStored)
        ]
        self.assertEqual(len(stored_hashes), 2)

        # Verify BlockRemoved event content
        remove_events = [e for e in events if isinstance(e, BlockRemoved)]
        self.assertEqual(len(remove_events), 1)
        self.assertEqual(remove_events[0].block_hashes, stored_hashes)

    def test_extra_key_isolation(self):
        """Test that keys with different extra_key values are isolated."""
        cache = RadixCache.create_simulated()

        # Insert same token sequence with different extra keys
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), "key1"),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), "key2"),
                value=torch.tensor([40, 50, 60], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3]), None),
                value=torch.tensor([70, 80, 90], dtype=torch.int64),
            )
        )

        # Keys with different extra_key should not match each other
        result1 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "key1"))
        )
        result2 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "key2"))
        )
        result3 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), None))
        )
        result4 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), "nonexistent"))
        )

        # Each should match only its own data
        self.assertEqual(len(result1.device_indices), 3)
        torch.testing.assert_close(
            result1.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )

        self.assertEqual(len(result2.device_indices), 3)
        torch.testing.assert_close(
            result2.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64)
        )

        self.assertEqual(len(result3.device_indices), 3)
        torch.testing.assert_close(
            result3.device_indices, torch.tensor([70, 80, 90], dtype=torch.int64)
        )

        # Non-existent extra_key should not match
        self.assertEqual(len(result4.device_indices), 0)

    def test_tenant_lookup_prefers_personal_then_global(self):
        """Derived property: tenant lookups must prefer personal cache over global.

        This guards the two-tier lookup contract. A future refactor that checks
        the global cache first would silently leak cross-tenant timing benefits
        and return the wrong KV mapping for users with personal entries.

        It also guards the write-side invariant: no insert -- with or without a
        user_id -- may write to the global tree directly. Anonymous inserts land
        in the default tenant's personal cache and only reach the global tree
        via the promotion path.
        """
        cache = RadixCache.create_simulated(enable_multi_tenant_cache=True)
        key = RadixKey(array("q", [1, 2, 3, 4]))

        personal_value = torch.tensor([110, 120, 130, 140], dtype=torch.int64)
        anon_value = torch.tensor([10, 20, 30, 40], dtype=torch.int64)

        # Tenant-scoped insertion path.
        cache.insert(InsertParams(key=key, value=personal_value, user_id="user-a"))
        # Anonymous insertion path (no user_id): must NOT reach the global tree.
        cache.insert(InsertParams(key=key, value=anon_value))

        user_a_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])), user_id="user-a")
        )
        torch.testing.assert_close(user_a_match.device_indices, personal_value)

        # user-b sees nothing: neither a personal entry nor any global write.
        user_b_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])), user_id="user-b")
        )
        self.assertEqual(len(user_b_match.device_indices), 0)

        # The anonymous insert landed in the default tenant's personal cache.
        anon_match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4])), user_id=DEFAULT_TENANT_ID
            )
        )
        torch.testing.assert_close(anon_match.device_indices, anon_value)

    def test_tenant_promotion_never_before_min_to_raise(self):
        """The MIN_TO_RAISE floor is absolute: a prompt cannot be promoted
        before MIN_TO_RAISE distinct users have requested it, even when a
        promotion roll would otherwise win.

        A same-user repeat hits the personal cache before the floor.
        """
        cache = RadixCache.create_simulated(enable_multi_tenant_cache=True)
        tokens = array("q", [1, 2, 3, 4, 5])
        key = RadixKey(tokens)
        value = torch.tensor([11, 22, 33, 44, 55], dtype=torch.int64)

        def lookup(user_id):
            return cache.match_prefix(
                MatchPrefixParams(key=RadixKey(tokens), user_id=user_id)
            ).device_indices

        def miss_insert(user_id):
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id=user_id,
                    track_miss_for_promotion=True,
                )
            )

        miss_insert("user-0")
        # A same-user repeat hits the personal cache before promotion.
        self.assertEqual(len(lookup("user-0")), 5)

        # Force every promotion roll to win: still nothing below the floor.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            for i in range(1, 5):
                self.assertEqual(
                    len(lookup(f"user-{i}")), 0, f"user-{i} must not hit before promotion"
                )
                miss_insert(f"user-{i}")
            # Exactly MIN_TO_RAISE distinct users: the floor itself never raises.
            self.assertEqual(len(lookup("user-zzz")), 0)

    def test_tenant_promotion_lifecycle_winning_roll_promotes_globally(self):
        """End-to-end: the first MIN_TO_RAISE distinct users each miss on a new
        prompt, then a winning roll on the next distinct user promotes it so any
        later user is served by the global cache (and the first users see it).
        """
        cache = RadixCache.create_simulated(enable_multi_tenant_cache=True)
        tokens = array("q", [1, 2, 3, 4, 5])
        key = RadixKey(tokens)
        value = torch.tensor([11, 22, 33, 44, 55], dtype=torch.int64)

        def lookup(user_id):
            return cache.match_prefix(
                MatchPrefixParams(key=RadixKey(tokens), user_id=user_id)
            ).device_indices

        def miss_insert(user_id):
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id=user_id,
                    track_miss_for_promotion=True,
                )
            )

        floor = cache._get_min_to_raise()
        # The first MIN_TO_RAISE distinct users get no cache hit.
        for i in range(floor):
            self.assertEqual(len(lookup(f"user-{i}")), 0)
            miss_insert(f"user-{i}")

        # Force a winning roll: the floor+1-th user's insert promotes the prompt.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            miss_insert("user-trigger")

        # Any later user is served by the global cache; the first users see it.
        post = lookup("user-zzz")
        self.assertEqual(len(post), 5)
        torch.testing.assert_close(post, value)
        self.assertEqual(len(lookup("user-3")), 5)

    def test_tenant_promotion_ramps_with_distinct_users(self):
        """After MIN_TO_RAISE distinct users the promotion odds ramp 1% per
        additional distinct user, and are guaranteed at MIN_TO_RAISE +
        PROMOTION_ODDS_WINDOW (the upper bound of the dice roll).
        """
        cache = RadixCache.create_simulated(enable_multi_tenant_cache=True)
        floor = cache._get_min_to_raise()
        window = cache.PROMOTION_ODDS_WINDOW

        # Below the floor, even a guaranteed-winning roll never promotes.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            self.assertFalse(cache._should_promote(floor))

        # At floor + 1 the chance is 1/100: 0.005 wins, 0.05 loses.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.005
        ):
            self.assertTrue(cache._should_promote(floor + 1))
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.05
        ):
            self.assertFalse(cache._should_promote(floor + 1))

        # From floor + window on, promotion is guaranteed for any roll.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.9999
        ):
            self.assertTrue(cache._should_promote(floor + window))
            self.assertTrue(cache._should_promote(floor + window + 50))

    def test_tenant_promotion_requires_min_unique_miss_users(self):
        """Global promotion requires MIN_TO_RAISE unique miss users: hit-path
        inserts must not count toward promotion, and the floor itself never
        raises (a winning roll only starts at floor + 1).
        """
        cache = RadixCache.create_simulated(enable_multi_tenant_cache=True)
        floor = cache._get_min_to_raise()
        key = RadixKey(array("q", [21, 22, 23, 24]))
        value = torch.tensor([210, 220, 230, 240], dtype=torch.int64)

        cache.insert(
            InsertParams(
                key=key,
                value=value,
                user_id="user-a",
                track_miss_for_promotion=True,
            )
        )

        # A hit-path write (track_miss_for_promotion=False) must not count.
        cache.insert(
            InsertParams(
                key=key,
                value=value,
                user_id="user-b",
                track_miss_for_promotion=False,
            )
        )

        for user_id in ["user-c", "user-d", "user-e"]:
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id=user_id,
                    track_miss_for_promotion=True,
                )
            )

        def lookup(user_id):
            return cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [21, 22, 23, 24])), user_id=user_id)
            ).device_indices

        # With a guaranteed-winning roll, MIN_TO_RAISE unique misses are still
        # required: only floor - 1 tracked users so far, so no promotion.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            self.assertEqual(len(lookup("user-z")), 0)

            # Reaching the floor does not raise yet...
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id="user-f",
                    track_miss_for_promotion=True,
                )
            )
            self.assertEqual(len(lookup("user-z")), 0)

            # ...the floor + 1-th unique miss user gets the winning roll.
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id="user-g",
                    track_miss_for_promotion=True,
                )
            )
            torch.testing.assert_close(lookup("user-z"), value)

        # The tracker counted exactly floor + 1 unique miss users (user-b was a
        # hit-path write and never entered the tracker).
        self.assertEqual(
            len(cache.prompt_request_tracker[cache._build_prompt_tracker_key(key)]),
            floor + 1,
        )

    def test_tenant_promotion_transfers_ownership_to_global(self):
        """Rework invariant: after promotion, the promoted slots are owned by the
        global tree only.

        The triggering personal node is detached (without freeing its slots), and
        later requests fully served by the global tree must not duplicate those
        slots into a personal cache. Guards against double-owned KV slots that a
        non-ref-counted allocator would double-free under eviction.
        """
        cache = RadixCache.create_simulated(enable_multi_tenant_cache=True)
        floor = cache._get_min_to_raise()
        tokens = array("q", [1, 2, 3, 4])
        key = RadixKey(tokens)
        value = torch.tensor([11, 22, 33, 44], dtype=torch.int64)

        def miss_insert(user_id):
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id=user_id,
                    track_miss_for_promotion=True,
                )
            )

        # Force a winning roll so the floor+1-th user triggers the promotion.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            for i in range(floor + 1):
                miss_insert(f"user-{i}")

            # The triggering user's personal cache no longer holds a copy.
            self.assertEqual(
                cache.personal_caches[f"user-{floor}"]._total_size_helper(), 0
            )

            # A request fully served by the global tree must not duplicate the
            # slots into a personal cache (the personal cache is not created).
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id="user-later",
                    track_miss_for_promotion=False,
                )
            )
        self.assertNotIn("user-later", cache.personal_caches)

        # The promoted prompt is reachable through the global tier for any user.
        for user_id in ["user-3", "user-zzz"]:
            hit = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(tokens), user_id=user_id)
            )
            torch.testing.assert_close(hit.device_indices, value)

        # The global tree owns the content exactly once.
        self.assertEqual(len(cache.root_node.children), 1)

    def test_tenant_promotion_counts_truncated_global_match_inserts(self):
        """Regression: a prompt that only PARTIALLY matches the global tree must
        still be tracked toward promotion.

        When the global tree already owns a shared prefix (e.g. a chat template),
        a new prompt sharing that prefix is stored as a truncated insert (only the
        suffix reaches the personal cache) and the old full-miss-only tracker never
        counted it, so the prompt could not be promoted no matter how many distinct
        users requested it. The truncated insert path must feed the tracker and
        promote the suffix below the existing global prefix.
        """
        cache = RadixCache.create_simulated(enable_multi_tenant_cache=True)
        floor = cache._get_min_to_raise()

        # First, promote a "template + secret-A" prompt so the global tree owns a
        # shared prefix that later prompts will partially match.
        template_key = RadixKey(array("q", [1, 2, 3, 4, 5]))
        template_value = torch.tensor([11, 22, 33, 44, 55], dtype=torch.int64)
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            for i in range(floor + 1):
                cache.insert(
                    InsertParams(
                        key=template_key,
                        value=template_value,
                        user_id=f"template-{i}",
                        track_miss_for_promotion=True,
                    )
                )
        self.assertIn(
            cache._build_prompt_tracker_key(template_key), cache.promoted_prompt_keys
        )

        # A new prompt sharing the [1, 2, 3] prefix: every insert is truncated at
        # the shared prefix, so only the [6, 7] suffix reaches each personal cache.
        shared_key = RadixKey(array("q", [1, 2, 3, 6, 7]))
        shared_value = torch.tensor([11, 22, 33, 66, 77], dtype=torch.int64)

        def lookup(user_id):
            return cache.match_prefix(
                MatchPrefixParams(key=shared_key, user_id=user_id)
            ).device_indices

        # Truncated inserts must count toward promotion of the shared prompt.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            for i in range(floor + 1):
                result = cache.insert(
                    InsertParams(
                        key=shared_key,
                        value=shared_value,
                        user_id=f"user-{i}",
                        track_miss_for_promotion=True,
                    )
                )
                # Before promotion only the shared global prefix is matched.
                self.assertEqual(result.prefix_len, 3)

        # The floor+1-th distinct user triggered promotion of the shared prompt.
        self.assertIn(
            cache._build_prompt_tracker_key(shared_key), cache.promoted_prompt_keys
        )
        # The [6, 7] suffix now lives below the existing global prefix, so any
        # later user is fully served by the global tree with the shared slots.
        hit = lookup("user-later")
        torch.testing.assert_close(hit, shared_value)
        self.assertEqual(len(hit), 5)

    def test_tenant_promotion_guard_purged_on_global_eviction(self):
        """The promotion guard must be cleared when the promoted node's slots leave
        the global tree, so the prompt can be promoted again on future full misses.
        """
        cache = RadixCache.create_simulated(
            mock_allocator=unittest.mock.Mock(device=torch.device("cpu")),
            enable_multi_tenant_cache=True,
        )
        floor = cache._get_min_to_raise()
        tokens = array("q", [1, 2, 3, 4])
        key = RadixKey(tokens)
        value = torch.tensor([11, 22, 33, 44], dtype=torch.int64)

        # Force a winning roll so the floor+1-th user triggers the promotion.
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            for i in range(floor + 1):
                cache.insert(
                    InsertParams(
                        key=key,
                        value=value,
                        user_id=f"user-{i}",
                        track_miss_for_promotion=True,
                    )
                )
        prompt_key = cache._build_prompt_tracker_key(key)
        self.assertIn(prompt_key, cache.promoted_prompt_keys)

        # A fully-global hit (later user) must not mutate the guard.
        cache.insert(
            InsertParams(
                key=key,
                value=value,
                user_id="user-later",
                track_miss_for_promotion=False,
            )
        )
        self.assertIn(prompt_key, cache.promoted_prompt_keys)

        # Evict everything: the promoted node's slots are freed, so the guard is
        # purged and the prompt becomes eligible for promotion again.
        cache.evict(EvictParams(num_tokens=100))
        self.assertNotIn(prompt_key, cache.promoted_prompt_keys)

        # A fresh distinct user can now promote the prompt again (winning roll).
        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.random.random", return_value=0.0
        ):
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    user_id="user-new",
                    track_miss_for_promotion=True,
                )
            )
        self.assertIn(prompt_key, cache.promoted_prompt_keys)
        self.assertEqual(len(cache.promoted_prompt_node_ids), 1)

    def test_lock_ref_operations(self):
        """Test lock reference counting operations."""
        cache = RadixCache.create_simulated()

        # Insert sequence
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )

        # Get node
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])))
        )
        node = result.last_device_node

        initial_evictable = cache.evictable_size()
        initial_protected = cache.protected_size()

        # Lock the node
        cache.inc_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected + 3)
        self.assertEqual(cache.evictable_size(), initial_evictable - 3)

        # Unlock the node
        cache.dec_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected)
        self.assertEqual(cache.evictable_size(), initial_evictable)

    def test_evict_functionality(self):
        """Test eviction functionality."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)

        # Insert sequences
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2])),
                value=torch.tensor([10, 20], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [3, 4])),
                value=torch.tensor([30, 40], dtype=torch.int64),
            )
        )

        initial_size = cache.total_size()

        # Evict some tokens
        result = cache.evict(EvictParams(num_tokens=2))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(
            result.num_tokens_evicted,
            2,
            f"evicted {result.num_tokens_evicted} tokens, expected at least 2",
        )

        # Should have called free_segment and reduced size
        mock_allocator.free_segment.assert_called()
        self.assertLess(cache.total_size(), initial_size)

    def test_page_alignment_boundary(self):
        """Test page alignment with different sizes."""
        test_cases = [
            (1, 5),
            (2, 5),
            (4, 6),
        ]

        for page_size, sequence_length in test_cases:
            with self.subTest(page_size=page_size, sequence_length=sequence_length):
                cache = RadixCache.create_simulated(page_size=page_size)

                tokens = list(range(sequence_length))
                key = RadixKey(array("q", tokens))
                cache.insert(
                    InsertParams(
                        key=key,
                        value=torch.tensor(tokens, dtype=torch.int64)[: len(key)],
                    )
                )

                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens)))
                )
                self.assertGreater(len(result.device_indices), 0)

                # Match length should be page-aligned
                match_len = len(result.device_indices)
                self.assertEqual(match_len % page_size, 0)

    def test_advanced_prefix_match_with_node_splits(self):
        """Advanced prefix matching: splits inside nodes and across pages."""
        for page_size in [1, 2]:
            with self.subTest(page_size=page_size):
                cache = RadixCache.create_simulated(page_size=page_size)

                # Insert a long sequence that will be split later.
                seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
                val1 = torch.tensor([x * 10 for x in seq1], dtype=torch.int64)
                cache.insert(InsertParams(key=RadixKey(array("q", seq1)), value=val1))

                # Insert a diverging branch to create an internal node on the path.
                seq2 = [1, 2, 9, 10]
                val2 = torch.tensor([x * 10 for x in seq2], dtype=torch.int64)
                cache.insert(InsertParams(key=RadixKey(array("q", seq2)), value=val2))
                print(cache.pretty_print())

                baseline_total = cache.total_size()
                expected_total = 10  # 8 + 2
                self.assertEqual(baseline_total, expected_total)

                # Match that causes a split inside an existing node:
                # take first 4 tokens of seq1, then diverge.
                query1 = [1, 2, 3, 4, 999, 1000]
                result1 = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", query1)))
                )
                torch.testing.assert_close(result1.device_indices, val1[:4])
                # No data change after structural split during matching.
                self.assertEqual(cache.total_size(), baseline_total)

                # Full match of the long sequence still returns the full indices.
                result_full = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq1)))
                )
                torch.testing.assert_close(result_full.device_indices, val1)

                # Another split deeper on the path (after matching 6 tokens, then diverge).
                query2 = [1, 2, 3, 4, 5, 6, 777, 888]
                result2 = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", query2)))
                )
                torch.testing.assert_close(result2.device_indices, val1[:6])
                self.assertEqual(cache.total_size(), baseline_total)

                # Matching the short diverging branch should return exactly its indices.
                result_branch = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq2)))
                )
                torch.testing.assert_close(result_branch.device_indices, val2)

    def test_hash_value_storage(self):
        """Test that hash_value is stored correctly after insert operations."""
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence
        cache.insert(
            InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 7, 8])), value=None)
        )

        # Trigger event emission to compute hash_value lazily
        cache.take_events()

        # Find the inserted node (traverse from root)
        node = cache.root_node
        for i in range(0, 8, 4):  # page_size=4, so 2 pages
            child_key = tuple([1, 2, 3, 4][:4]) if i == 0 else tuple([5, 6, 7, 8][:4])
            if child_key in node.children:
                node = node.children[child_key]
                break

        # Verify hash_value is set (computed lazily during event emission)
        self.assertIsNotNone(node.hash_value)
        # Should have 2 pages (8 tokens / 4 page_size)
        self.assertEqual(len(node.hash_value), 2)

    def test_hash_value_repeating_tokens(self):
        """Test that repeating token patterns get different hash values."""
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence with repeating token pattern: [1,2,3,4, 1,2,3,4]
        cache.insert(
            InsertParams(key=RadixKey(array("q", [1, 2, 3, 4, 1, 2, 3, 4])), value=None)
        )

        events = cache.take_events()
        block_stored_events = [e for e in events if isinstance(e, BlockStored)]

        # Should have 2 blocks (2 pages of size 4)
        self.assertEqual(len(block_stored_events), 2)

        # Extract block hashes
        block_hash_1 = block_stored_events[0].block_hashes[0]
        block_hash_2 = block_stored_events[1].block_hashes[0]

        # The two blocks should have DIFFERENT hashes despite same content
        # because they are at different positions (sequence-aware hashing)
        self.assertNotEqual(
            block_hash_1,
            block_hash_2,
            "Repeating token patterns should get different sequence-aware hashes",
        )

        # First block should have no parent
        self.assertIsNone(block_stored_events[0].parent_block_hash)

        # Second block's parent should be the first block's hash
        self.assertEqual(block_stored_events[1].parent_block_hash, block_hash_1)

    def test_hash_value_split(self):
        """Test that hash_value is split correctly when nodes are split."""
        cache = RadixCache.create_simulated(
            page_size=2,
            enable_kv_cache_events=True,
        )

        # Insert a sequence that will cause a split
        cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 3, 4])), value=None))
        cache.take_events()  # Clear events and compute hash_value for first node

        # Insert a diverging sequence that will cause a split at page boundary
        cache.insert(InsertParams(key=RadixKey(array("q", [1, 2, 5, 6])), value=None))
        cache.take_events()  # Trigger event emission to compute hash_value

        # Find the split node
        node = cache.root_node
        child_key = tuple([1, 2])
        if child_key in node.children:
            node = node.children[child_key]
            # After split and event emission, hash_value should be computed
            # Note: If hash_value wasn't set before split, it will be computed lazily
            # during event emission. If it was set, it will be split.
            # Either way, after events are emitted, it should be set.
            self.assertIsNotNone(node.hash_value)
            # Should have 1 page (split at page_size=2)
            self.assertEqual(len(node.hash_value), 1)

    def test_memory_allocated(self):
        keys, values = [], []

        num_seqs = 10000
        vocab_size = 1000
        base_prefix_len = 10000
        suffix_len = 100

        torch_allocated_before = torch.get_device_module().memory_allocated()

        # build dataset with common prefix
        common_prefix = [
            random.randint(1, vocab_size - 1) for _ in range(base_prefix_len)
        ]
        for _ in range(num_seqs):
            suffix = [random.randint(1, vocab_size - 1) for _ in range(suffix_len)]
            seq = common_prefix + suffix
            keys.append(seq)
            values.append(torch.zeros(len(seq), device=get_device(), dtype=torch.int32))

        cache: RadixCache = RadixCache.create_simulated()

        for key, value in zip(keys, values):
            cache.insert(InsertParams(key=RadixKey(array("q", key)), value=value))

        del values

        torch_allocated = (
            torch.get_device_module().memory_allocated() - torch_allocated_before
        )
        cache_size_bytes = cache.total_size() * 4
        print(f"\nCache size (MB): {cache_size_bytes / (1024 * 1024)}")
        print(f"Torch allocated (MB): {torch_allocated / (1024 * 1024)}")

        # The cache size should be within reasonable bounds of the actual allocated memory.
        self.assertLess(torch_allocated, cache_size_bytes * 2)


if __name__ == "__main__":
    unittest.main()
