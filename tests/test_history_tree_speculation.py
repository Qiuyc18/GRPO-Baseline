import math
import random
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.history_tree_speculation import (
    DraftProposal,
    HistoryTreeSpeculativeRollout,
    PolicyVersionCache,
    TrajectoryTree,
    emitted_distribution_after_verify,
    residual_distribution_from_probs,
    stable_prompt_key,
)


class _LogProb:
    def __init__(self, logprob):
        self.logprob = logprob


class _Sample:
    def __init__(self, token_id, logprob):
        self.token_ids = [token_id]
        self.logprobs = [{token_id: _LogProb(logprob)}]


class _Output:
    def __init__(self, sample_token=None, sample_logprob=None, prompt_logprobs=None):
        self.outputs = []
        if sample_token is not None:
            self.outputs.append(_Sample(sample_token, sample_logprob))
        self.prompt_logprobs = prompt_logprobs


class _SamplingParams:
    def __init__(self):
        self.max_tokens = 1
        self.n = 1
        self.logprobs = 1
        self.prompt_logprobs = None


class _FakeEngine:
    def __init__(self, score_logprobs=None, sample_tokens=None):
        self.score_logprobs = score_logprobs or {}
        self.sample_tokens = list(sample_tokens or [])

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
        if getattr(sampling_params, "prompt_logprobs", None) is not None:
            outputs = []
            for prompt in prompts:
                token_id = prompt["prompt_token_ids"][-1]
                logp = self.score_logprobs[token_id]
                prompt_logprobs = [None] * (len(prompt["prompt_token_ids"]) - 1)
                prompt_logprobs.append({token_id: _LogProb(logp)})
                outputs.append(_Output(sample_token=0, sample_logprob=0.0, prompt_logprobs=prompt_logprobs))
            return outputs
        outputs = []
        for _ in prompts:
            token_id, logp = self.sample_tokens.pop(0)
            outputs.append(_Output(sample_token=token_id, sample_logprob=logp))
        return outputs


class _FixedRng:
    def __init__(self, random_values):
        self.random_values = list(random_values)

    def choices(self, population, weights, k):
        return [population[0]]

    def random(self):
        return self.random_values.pop(0)


def _enabled_rollout(**overrides):
    cfg = {
        "seed": 0,
        "history_tree_speculation": {
            "enabled": True,
            "max_depth": 1,
            "max_branch_width": 8,
            "min_visits": 1,
            "exact_residual": True,
            "candidate_prompt_logprobs": -1,
        },
    }
    cfg["history_tree_speculation"].update(overrides)
    return HistoryTreeSpeculativeRollout(cfg, pad_token_id=0, eos_token_id=2)


def test_residual_distribution_correctness():
    p = torch.tensor([0.10, 0.25, 0.40, 0.25])
    child_ids = torch.tensor([1, 3])
    q = torch.tensor([0.50, 0.50])
    residual = residual_distribution_from_probs(p, child_ids, q)
    emitted = emitted_distribution_after_verify(p, child_ids, q)
    assert torch.allclose(residual, torch.tensor([0.2, 0.0, 0.8, 0.0]), atol=1e-6)
    assert torch.allclose(emitted, p, atol=1e-6)


def test_q_tree_normalization_no_nan_or_inf():
    tree = TrajectoryTree()
    prompt_key = stable_prompt_key([11, 12])
    tree.observe(prompt_key, [5], old_logprobs=[math.log(0.4)], reward=1.0)
    tree.observe(prompt_key, [6], old_logprobs=[math.log(0.2)], reward=2.0)
    tree.observe(prompt_key, [6], old_logprobs=[math.log(0.3)], reward=2.0)
    root_id = tree.find_node(prompt_key, [])
    token_ids, probs = tree.child_distribution(
        root_id,
        count_alpha=1.0,
        use_reward_prior=True,
        reward_lambda=0.1,
        use_nll_prior=True,
        nll_eta=0.05,
    )
    assert set(token_ids) == {5, 6}
    assert abs(sum(probs) - 1.0) < 1e-8
    assert all(math.isfinite(p) and p > 0 for p in probs)


def test_accepted_draft_old_logprob_is_online_logprob_not_q():
    rollout = _enabled_rollout()
    prompt = [10, 11]
    prompt_key = stable_prompt_key(prompt)
    rollout.tree.observe(prompt_key, [3], old_logprobs=[math.log(0.5)])
    rollout.tree.observe(prompt_key, [4], old_logprobs=[math.log(0.5)])
    rollout.rng = _FixedRng([0.0])
    engine = _FakeEngine(score_logprobs={3: math.log(0.9)})
    result = rollout.generate_sequences(
        inference_engine=engine,
        vllm_inputs=[{"prompt_token_ids": prompt}],
        sampling_params=_SamplingParams(),
        response_length=1,
        eos_token_id=2,
    )
    assert result["responses"] == [[3]]
    assert abs(result["rollout_log_probs"][0][0] - math.log(0.9)) < 1e-8
    assert abs(result["rollout_log_probs"][0][0] - math.log(0.5)) > 1e-3


def test_residual_sample_old_logprob_is_online_logprob_not_q():
    rollout = _enabled_rollout()
    prompt = [10, 11]
    prompt_key = stable_prompt_key(prompt)
    root_id = rollout.tree.get_or_create_root(prompt_key)
    rollout.tree.observe(prompt_key, [3], old_logprobs=[math.log(0.5)])
    rollout.tree.observe(prompt_key, [4], old_logprobs=[math.log(0.5)])
    engine = _FakeEngine(sample_tokens=[(8, math.log(0.7))])
    token_id, logp = rollout._sample_residual_one(engine, prompt, root_id, None, _SamplingParams(), None)
    assert token_id == 8
    assert abs(logp - math.log(0.7)) < 1e-8


def test_disabled_and_empty_tree_fallback_behavior():
    disabled = HistoryTreeSpeculativeRollout({"history_tree_speculation": {"enabled": False}}, pad_token_id=0)
    assert disabled.generate_sequences(None, [], None, 1, eos_token_id=2) is None

    enabled = _enabled_rollout()
    result = enabled.generate_sequences(
        inference_engine=_FakeEngine(),
        vllm_inputs=[{"prompt_token_ids": [1]}],
        sampling_params=_SamplingParams(),
        response_length=1,
        eos_token_id=2,
    )
    assert result is None


def test_batched_variable_lengths_with_eos():
    rollout = _enabled_rollout()
    rollout.tree.observe(stable_prompt_key([1]), [2], old_logprobs=[math.log(0.4)])
    rollout.tree.observe(stable_prompt_key([3]), [7, 8], old_logprobs=[math.log(0.5), math.log(0.6)])
    rollout.rng = _FixedRng([0.0, 0.0, 0.0])
    engine = _FakeEngine(score_logprobs={2: math.log(0.4), 7: math.log(0.5), 8: math.log(0.6)})
    result = rollout.generate_sequences(
        inference_engine=engine,
        vllm_inputs=[{"prompt_token_ids": [1]}, {"prompt_token_ids": [3]}],
        sampling_params=_SamplingParams(),
        response_length=2,
        eos_token_id=2,
    )
    assert result["responses"][0] == [2]
    assert result["responses"][1] == [7, 8]


def test_policy_version_cache_invalidation():
    cache = PolicyVersionCache()
    cache.put("k", "prompt-a", SimpleNamespace(value=1))
    assert cache.get("k", "prompt-a").value == 1
    assert cache.get("k", "prompt-b") is None
    cache.bump_policy_version()
    assert cache.policy_version == 1
    assert cache.get("k", "prompt-a") is None


def test_update_tree_stores_only_prefix_tokens():
    rollout = _enabled_rollout(max_tokens_to_store=2)
    batch = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[10, 11]]),
                "responses": torch.tensor([[3, 4, 5, 6]]),
                "response_mask": torch.tensor([[1, 1, 1, 1]]),
                "old_log_probs": torch.tensor([[math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)]]),
                "token_level_scores": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            },
            batch_size=[1],
        ),
        non_tensor_batch={"uid": np.array(["u0"], dtype=object)},
    )
    metrics = rollout.update_tree_from_batch(batch)
    root_id = rollout.tree.find_node(stable_prompt_key([10, 11]), [])
    first_id = rollout.tree.nodes[root_id].children[3]
    second_id = rollout.tree.nodes[first_id].children[4]
    assert metrics["history_tree_updated_sequences"] == 1.0
    assert 5 not in rollout.tree.nodes[second_id].children
