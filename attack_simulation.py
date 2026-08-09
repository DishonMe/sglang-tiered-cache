"""InputSnatch-style attack simulation against the multi-tenant radix cache.

Each round is self-contained and simulates a fresh victim secret:

  STAGE 1 (setup)   A RANDOM number of distinct users each request the SAME
                    secret prompt exactly once. Every request is a full cache
                    miss (a new user has no personal cache and the prompt is
                    not yet in the shared global cache), so every request is a
                    full prefill and is tracked for promotion.

  STAGE 2 (attack)  An "attacker" repeatedly requests that same secret prompt
                    from FRESH user ids until either
                      * he observes a fast (cache hit) latency, or
                      * MAX_ATTACK_ATTEMPTS (105) attempts have passed.
                    He calibrates the hit/miss threshold himself: he requests a
                    unique long string from a fresh user (full miss -> slow),
                    requests the SAME string again from that user (personal
                    cache hit -> fast) and sets
                        threshold = hit + FRACTION * (miss - hit)
                    so that, e.g., miss=0.6s / hit=0.3s gives a 0.45s
                    threshold (0.5s => miss, 0.4s => hit).

The suite runs ROUNDS rounds and writes two CSVs:

  * round_summary.csv   one row per round, with BOTH the attacker-visible view
                        (calibration latencies, threshold, per-round attempt
                        stats, what he detected) AND the ground truth we, the
                        testers, extract from the server [DBG] lines (number
                        of setup requests, the distinct-user count at which the
                        prompt was promoted to the global cache, and which
                        attack attempt actually hit the global cache).
  * attempt_details.csv one row per attack attempt across all rounds.

The server MUST be launched with:
    --enable-multi-tenant-cache --enable-radix-cache-debug-log
and its stdout log path passed via argv[1] / --log / $SERVER_LOG (the [DBG]
lines are what let us verify the cache mechanics deterministically).

Defense invariants asserted from the logs:
    * a prompt is NEVER promoted before MIN_TO_RAISE + 1 (6) distinct users;
    * the attacker can never hit the global cache before that promotion point
      (a fresh user has no personal cache, so the only fast path is global);
    * promotion is guaranteed by MIN_TO_RAISE + PROMOTION_ODDS_WINDOW (105)
      distinct users, so the attacker's own attempts eventually win.
"""

import argparse
import csv
import os
import random
import re
import secrets
import sys
import time
from openai import OpenAI

BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("SGLANG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Keep in sync with RadixCache in python/sglang/srt/mem_cache/radix_cache.py.
MIN_TO_RAISE = 5
PROMOTION_ODDS_WINDOW = 100
PROMOTION_GUARANTEED = MIN_TO_RAISE + PROMOTION_ODDS_WINDOW  # 105

# "A secret" the victim has processed. Deliberately LONG so a cache MISS is a
# slow full prefill while a cache HIT is near-instant, making the latency-based
# attack detectable. Each round salts it to make the prompt unique per round.
_SECRET_BLOCK = (
    "The confidential medical evaluation for patient John Doe was completed on "
    "August 4th and contains the following findings. The patient presented with "
    "persistent fatigue, elevated blood glucose levels, and a family history of "
    "metabolic disorders. Laboratory results confirm a diagnosis of Type 2 "
    "Diabetes Mellitus. Recommended treatment includes lifestyle modification, "
    "regular monitoring of hemoglobin A1c levels, and an individualized "
    "medication plan to be reviewed at the next follow-up appointment. This "
    "document is protected health information and must not be disclosed to "
    "unauthorized parties under any circumstances."
)

# English-like filler used to build the attacker's unique calibration probe at
# roughly the same length as the secret, so the miss prefill cost is comparable.
_FILLER = (
    "The quick brown fox jumped over the lazy dog near the river bank while "
    "watching the stars at night and drinking coffee by the warm fire. "
)

DBG_RE = re.compile(r"\[DBG\] user_id=(\S+) key_len=(\d+) global_match_len=(\d+)(.*)")


class ServerLog:
    """Incrementally reads the server's stdout log for [DBG] line parsing."""

    def __init__(self, path):
        self.path = path
        self.offset = 0

    def usable(self):
        return bool(self.path) and os.path.isfile(self.path)

    def snapshot(self):
        try:
            self.offset = os.path.getsize(self.path)
        except OSError:
            self.offset = 0

    def read_new(self):
        if not self.usable():
            return []
        with open(self.path, errors="replace") as f:
            f.seek(self.offset)
            data = f.read()
            self.offset = f.tell()
        return data.splitlines()


def make_secret(round_idx):
    salt = f"Confidential record for round {round_idx} serial {secrets.token_hex(4)}. "
    return " ".join([salt + _SECRET_BLOCK] * 19)


def make_probe(target_chars):
    """A unique (timestamp + random) long string, English-like, ~ target length."""
    salt = f"{time.time_ns()} {secrets.token_hex(8)}. "
    filler = ""
    while len(salt) + len(filler) < target_chars:
        filler += _FILLER
    return salt + filler


def parse_dbg(lines):
    recs = []
    for ln in lines:
        m = DBG_RE.search(ln)
        if m:
            recs.append(
                {
                    "user": m.group(1),
                    "key_len": int(m.group(2)),
                    "global_match_len": int(m.group(3)),
                    "hit_global": "hit_global=True" in m.group(4),
                    "truncated": "truncated=True" in m.group(4),
                }
            )
    return recs


# A request is treated as a global-cache hit when its DBG line shows a FULL
# match (hit_global=True) OR a large fraction of the prompt came from the global
# tree (truncated=True with global_match_len close to key_len). The tail fallback
# matters because a promoted node only ever holds the first (page-aligned) chunk
# of a long prompt, so a fully-cached request can still log as "truncated".
GLOBAL_HIT_FRACTION = 0.8


def _rec_is_hit(rec):
    if rec["hit_global"]:
        return True
    if rec["key_len"] > 0:
        return rec["global_match_len"] >= GLOBAL_HIT_FRACTION * rec["key_len"]
    return False


def analyze_log(lines, round_idx, max_attempts):
    """Tester-side ground truth for one round, derived from the [DBG] lines."""
    pref_setup = f"round{round_idx}_setup_"
    pref_atk = f"round{round_idx}_atk_"
    secret_recs = []
    atk_recs = []
    for rec in parse_dbg(lines):
        if rec["user"].startswith(pref_setup) or rec["user"].startswith(pref_atk):
            secret_recs.append(rec)
        if rec["user"].startswith(pref_atk):
            atk_recs.append(rec)

    setup_users_in_log = len({r["user"] for r in secret_recs if r["user"].startswith(pref_setup)})
    key_lens = sorted({r["key_len"] for r in secret_recs})

    # A single request emits several DBG lines (one per prefill chunk / growth
    # insert), so aggregate per user: a user "hit" if any of their lines did.
    user_first_idx = {}
    user_hit = {}
    for i, r in enumerate(secret_recs):
        user_first_idx.setdefault(r["user"], i)
        user_hit[r["user"]] = user_hit.get(r["user"], False) or _rec_is_hit(r)

    first_hit_idx = None
    first_hit_user = None
    for user, hit in user_hit.items():
        if hit and (first_hit_idx is None or user_first_idx[user] < first_hit_idx):
            first_hit_idx = user_first_idx[user]
            first_hit_user = user

    if first_hit_user is not None:
        promotion_users = len({r["user"] for r in secret_recs[:first_hit_idx]})
        # Promotion happened during the setup phase iff the promoting insert ran
        # while only setup users had requested (i.e. the promotion user count
        # does not exceed the number of distinct setup users).
        promoted_during_setup = promotion_users <= setup_users_in_log
    else:
        promotion_users = None
        promoted_during_setup = None

    log_first_hit_attempt = None
    atk_attempt_of_user = {}
    for i, r in enumerate(atk_recs):
        atk_attempt_of_user.setdefault(r["user"], i + 1)
    for user, hit in user_hit.items():
        if hit and user.startswith(pref_atk):
            attempt = atk_attempt_of_user[user]
            if log_first_hit_attempt is None or attempt < log_first_hit_attempt:
                log_first_hit_attempt = attempt

    return {
        "setup_users_in_log": setup_users_in_log,
        "key_lens": key_lens,
        "log_first_hit_attempt": log_first_hit_attempt,
        "log_promotion_users": promotion_users,
        "promoted_during_setup": promoted_during_setup,
        "log_secret_recs": len(secret_recs),
        "atk_users_in_log": len(atk_attempt_of_user),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--min-setup", type=int, default=1)
    ap.add_argument("--max-setup", type=int, default=60)
    ap.add_argument("--max-attempts", type=int, default=PROMOTION_GUARANTEED)
    ap.add_argument("--threshold-fraction", type=float, default=0.5,
                    help="threshold = hit + FRACTION*(miss-hit)")
    ap.add_argument("--calib-iterations", type=int, default=2,
                    help="miss/hit calibration sample pairs per round")
    ap.add_argument("--outdir", type=str, default="/home/banana/cache/tests/results")
    ap.add_argument("--log", type=str, default=None,
                    help="path to the server's stdout log (argv[1]/$SERVER_LOG fallback)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    server_log = ServerLog(args.log or (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SERVER_LOG", "")))
    if not server_log.usable():
        print("⚠️  No server log path given; attacker-visible results will be "
              "written but log-derived ground truth will be empty.")

    if args.seed is not None:
        random.seed(args.seed)

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

    def measure(user_id, prompt):
        t0 = time.perf_counter()
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            user=user_id,  # the tenant id -> per-user personal cache
            max_tokens=4,
        )
        return time.perf_counter() - t0

    # Sanity check that the server is up.
    try:
        client.models.list()
    except Exception as e:  # noqa: BLE001
        print(f"❌ Cannot reach server at {BASE_URL}: {e}")
        sys.exit(1)

    print("--- STARTING ATTACK SIMULATION ---")
    print(f"model={MODEL} rounds={args.rounds} setup=[{args.min_setup},{args.max_setup}] "
          f"max-attempts={args.max_attempts} threshold-fraction={args.threshold_fraction}")
    print("(server must be launched with --enable-multi-tenant-cache "
          "--enable-radix-cache-debug-log)\n")

    # Warm the GPU / CUDA cold-start before round 1.
    measure("warmup_user", "What is the capital of France?")
    print("[warmup] done\n")

    summary_rows = []
    attempt_rows = []

    for r in range(1, args.rounds + 1):
        secret = make_secret(r)
        server_log.snapshot()
        round_t0 = time.perf_counter()

        # ---- STAGE 1: setup ----
        n_setup = random.randint(args.min_setup, args.max_setup)
        setup_t0 = time.perf_counter()
        for i in range(n_setup):
            measure(f"round{r}_setup_{i}", secret)
        setup_time = time.perf_counter() - setup_t0

        # ---- attacker calibrates his latency threshold ----
        miss_samples, hit_samples = [], []
        for it in range(args.calib_iterations):
            probe = make_probe(len(secret))
            calib_user = f"round{r}_calib_{it}"
            miss_samples.append(measure(calib_user, probe))   # full miss (slow)
            hit_samples.append(measure(calib_user, probe))    # personal hit (fast)
        calib_miss = max(miss_samples)
        calib_hit = min(hit_samples)
        threshold = calib_hit + args.threshold_fraction * (calib_miss - calib_hit)

        # ---- STAGE 2: attack ----
        attack_t0 = time.perf_counter()
        attempt_lat = []
        attacker_hit_attempt = None
        attacker_hit_latency = None
        for j in range(1, args.max_attempts + 1):
            lat = measure(f"round{r}_atk_{j}", secret)
            attempt_lat.append(lat)
            hit = lat < threshold
            attempt_rows.append(
                {
                    "round": r,
                    "attempt": j,
                    "user": f"round{r}_atk_{j}",
                    "latency_s": round(lat, 6),
                    "attacker_verdict": "hit" if hit else "miss",
                }
            )
            if hit:
                attacker_hit_attempt = j
                attacker_hit_latency = lat
                break
        attack_time = time.perf_counter() - attack_t0

        # ---- tester-side ground truth from the server log ----
        log = analyze_log(server_log.read_new(), r, args.max_attempts)

        attempts_made = len(attempt_lat)
        miss_lats = [x for x in attempt_lat if x >= threshold]

        detection_matches_log = None
        if log["log_first_hit_attempt"] is not None and attacker_hit_attempt is not None:
            detection_matches_log = log["log_first_hit_attempt"] == attacker_hit_attempt

        # Defense invariant: never promoted before MIN_TO_RAISE + 1 distinct users.
        floor_respected = log["log_promotion_users"] is None or log["log_promotion_users"] >= MIN_TO_RAISE + 1
        # Attacker must eventually hit (guaranteed by 105 distinct users).
        attack_resolved = log["log_first_hit_attempt"] is not None

        summary_rows.append(
            {
                # --- round / tester choices ---
                "round": r,
                "setup_users": n_setup,
                "secret_chars": len(secret),
                # --- attacker-visible ---
                "calib_miss_s": round(calib_miss, 6),
                "calib_hit_s": round(calib_hit, 6),
                "attack_threshold_s": round(threshold, 6),
                "attacker_attempts": attempts_made,
                "attacker_hit_attempt": attacker_hit_attempt if attacker_hit_attempt is not None else "",
                "attacker_hit_latency_s": round(attacker_hit_latency, 6) if attacker_hit_latency is not None else "",
                "attacker_miss_min_s": round(min(miss_lats), 6) if miss_lats else "",
                "attacker_miss_median_s": round(sorted(miss_lats)[len(miss_lats) // 2], 6) if miss_lats else "",
                "attacker_miss_max_s": round(max(miss_lats), 6) if miss_lats else "",
                "setup_time_s": round(setup_time, 3),
                "attack_time_s": round(attack_time, 3),
                # --- tester ground truth (from [DBG] log) ---
                "log_setup_users_in_log": log["setup_users_in_log"],
                "log_promotion_users": log["log_promotion_users"] if log["log_promotion_users"] is not None else "",
                "promoted_during_setup": log["promoted_during_setup"] if log["promoted_during_setup"] is not None else "",
                "log_first_hit_attempt": log["log_first_hit_attempt"] if log["log_first_hit_attempt"] is not None else "",
                # --- verdicts ---
                "detection_matches_log": detection_matches_log if detection_matches_log is not None else "",
                "floor_respected": floor_respected,
                "attack_resolved": attack_resolved,
            }
        )

        status = []
        if not floor_respected:
            status.append("❌ FLOOR VIOLATION")
        if attack_resolved:
            status.append(f"hit@{log['log_first_hit_attempt']}")
        else:
            status.append("no global hit in log")
        if detection_matches_log is not None and not detection_matches_log:
            status.append("attacker/log mismatch")
        print(f"[round {r:>2}] setup={n_setup:>2} calib(miss/hit)={calib_miss:.3f}/{calib_hit:.3f}s "
              f"threshold={threshold:.3f}s attempts={attempts_made:>3} "
              f"attacker_hit={attacker_hit_attempt or '-'} log_hit={log['log_first_hit_attempt'] or '-'} "
              f"promote@{log['log_promotion_users'] or '-'}u ({'; '.join(status) or 'ok'}) "
              f"[{time.perf_counter() - round_t0:.1f}s]")

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, time.strftime("attack_%Y%m%d_%H%M%S"))
    summary_path = base + "_round_summary.csv"
    attempts_path = base + "_attempt_details.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(attempts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "attempt", "user", "latency_s", "attacker_verdict"])
        writer.writeheader()
        writer.writerows(attempt_rows)

    print(f"\n--- SUMMARY ({args.rounds} rounds) ---")
    ok = sum(1 for row in summary_rows if row["floor_respected"] and row["attack_resolved"])
    print(f"rounds with floor respected & attack resolved: {ok}/{args.rounds}")
    print(f"promotion distinct-user counts (log): {[r['log_promotion_users'] for r in summary_rows]}")
    print(f"attacker hit attempts (latency):        {[r['attacker_hit_attempt'] for r in summary_rows]}")
    print(f"log hit attempts (ground truth):        {[r['log_first_hit_attempt'] for r in summary_rows]}")
    print(f"\nCSV written:\n  {summary_path}\n  {attempts_path}")


if __name__ == "__main__":
    main()
