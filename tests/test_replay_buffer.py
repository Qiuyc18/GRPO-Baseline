"""
Verification script for ReplayBuffer.

Run inside the container (no GPU needed):
    docker exec rocm_verl python3 \
        /home/qinghua/qiuyc/tsinghua/GRPO-Baseline/tests/test_replay_buffer.py

Or from host (if verl deps are available on host Python):
    PYTHONPATH=verl-src python3 tests/test_replay_buffer.py

What is checked:
  1. put() serialises all expected fields to disk
  2. sample() deserialises and returns a valid DataProto
  3. Tensor values survive the numpy round-trip exactly
  4. LRU hot cache eviction works (hot_cache_size=2, put 3 batches)
  5. Disk eviction works (max_size=2, put 3 batches → oldest file deleted)
  6. should_replay() respects p_fresh=0.0 (always replay) and p_fresh=1.0 (never)
  7. Buffer is empty → should_replay() always False regardless of p_fresh
  8. Metric key 'rollout/from_cache' is present in ray_trainer patch
     (static check: just verifies the string exists in the patched file)
"""

import sys
import tempfile
import os

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO, "verl-src"))

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.replay_buffer import ReplayBuffer, _TENSOR_FIELDS

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_failures = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        _failures.append(name)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_batch(B: int = 4, n: int = 2, P: int = 8, R: int = 12, step: int = 1) -> DataProto:
    """Minimal DataProto that mimics a post-rollout batch."""
    BN = B * n
    PR = P + R
    td = TensorDict(
        {
            "input_ids":          torch.randint(0, 1000, (BN, PR)),
            "attention_mask":     torch.ones(BN, PR, dtype=torch.bool),
            "position_ids":       torch.arange(PR).unsqueeze(0).expand(BN, -1).clone(),
            "responses":          torch.randint(0, 1000, (BN, R)),
            "response_mask":      torch.ones(BN, R, dtype=torch.bool),
            "token_level_scores": torch.randn(BN, R),
        },
        batch_size=[BN],
    )
    # After batch.repeat(n, interleave=True) uid has length B*n
    uid = np.array([f"uid-{step}-{i}" for i in range(B) for _ in range(n)], dtype=object)
    dp = DataProto(batch=td, non_tensor_batch={"uid": uid})
    return dp


# ── test cases ────────────────────────────────────────────────────────────────

def test_roundtrip(tmpdir):
    print("\n[1] Tensor round-trip through disk")
    rb = ReplayBuffer(tmpdir, max_size=10, p_fresh=0.0, hot_cache_size=5)
    original = make_batch(step=1)
    rb.put(original, step=1)

    result = rb.sample(step=2)

    check("len(buffer)==1", len(rb) == 1)
    check("all tensor fields present", all(k in result.batch.keys() for k in _TENSOR_FIELDS))
    for k in _TENSOR_FIELDS:
        if k in original.batch.keys():
            ok = torch.equal(original.batch[k], result.batch[k])
            check(f"  {k} values match", ok)

    uid_ok = list(original.non_tensor_batch["uid"]) == list(result.non_tensor_batch["uid"])
    check("uid preserved", uid_ok)


def test_hot_cache_eviction(tmpdir):
    print("\n[2] LRU hot cache eviction (hot_cache_size=2)")
    rb = ReplayBuffer(tmpdir, max_size=10, p_fresh=0.0, hot_cache_size=2)
    for i in range(3):
        rb.put(make_batch(step=i), step=i)

    # Force load all 3 into hot cache; 3rd should evict 1st
    for _ in range(3):
        rb.sample(step=99)

    check("hot cache size ≤ 2", len(rb._hot) <= 2, f"actual={len(rb._hot)}")


def test_disk_eviction(tmpdir):
    print("\n[3] Disk eviction (max_size=2)")
    rb = ReplayBuffer(tmpdir, max_size=2, p_fresh=0.0, hot_cache_size=5)
    for i in range(3):
        rb.put(make_batch(step=i), step=i)

    check("index length == 2", len(rb._index) == 2, f"actual={len(rb._index)}")
    # First batch file should be gone
    first_file = os.path.join(tmpdir, "b0000000.npz")
    check("evicted file deleted", not os.path.exists(first_file))


def test_should_replay_flags(tmpdir):
    print("\n[4] should_replay() flag behaviour")
    rb_never = ReplayBuffer(tmpdir, max_size=10, p_fresh=1.0, hot_cache_size=5)
    rb_never.put(make_batch(step=1), step=1)
    results_never = [rb_never.should_replay() for _ in range(50)]
    check("p_fresh=1.0 → never replay", not any(results_never))

    rb_always = ReplayBuffer(tmpdir, max_size=10, p_fresh=0.0, hot_cache_size=5)
    rb_always.put(make_batch(step=1), step=1)
    results_always = [rb_always.should_replay() for _ in range(50)]
    check("p_fresh=0.0 → always replay", all(results_always))


def test_empty_buffer(tmpdir):
    print("\n[5] Empty buffer → should_replay() always False")
    rb = ReplayBuffer(tmpdir, max_size=10, p_fresh=0.0, hot_cache_size=5)
    results = [rb.should_replay() for _ in range(20)]
    check("empty buffer → no replay", not any(results))


def test_patch_metric_key():
    print("\n[6] ray_trainer.py patch — metric key 'rollout/from_cache' present")
    trainer_path = os.path.join(_REPO, "verl-src/verl/trainer/ppo/ray_trainer.py")
    with open(trainer_path) as f:
        src = f.read()
    check("'rollout/from_cache' in ray_trainer.py", "rollout/from_cache" in src)
    check("'from_cache' branch in ray_trainer.py", "from_cache" in src)
    check("ReplayBuffer import in ray_trainer.py", "replay_buffer" in src)


def test_index_persistence(tmpdir):
    print("\n[7] Index persists across ReplayBuffer instances")
    rb1 = ReplayBuffer(tmpdir, max_size=10, p_fresh=0.0, hot_cache_size=5)
    rb1.put(make_batch(step=5), step=5)
    rb1.put(make_batch(step=6), step=6)

    rb2 = ReplayBuffer(tmpdir, max_size=10, p_fresh=0.0, hot_cache_size=5)
    check("new instance sees existing index", len(rb2) == 2, f"actual={len(rb2)}")
    result = rb2.sample(step=99)
    check("sample from reloaded index returns DataProto", isinstance(result, DataProto))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ReplayBuffer verification")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d1:
        test_roundtrip(d1)
    with tempfile.TemporaryDirectory() as d2:
        test_hot_cache_eviction(d2)
    with tempfile.TemporaryDirectory() as d3:
        test_disk_eviction(d3)
    with tempfile.TemporaryDirectory() as d4:
        test_should_replay_flags(d4)
    with tempfile.TemporaryDirectory() as d5:
        test_empty_buffer(d5)
    test_patch_metric_key()
    with tempfile.TemporaryDirectory() as d6:
        test_index_persistence(d6)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)} test(s): {', '.join(_failures)}")
        sys.exit(1)
    else:
        print("All tests passed.")
