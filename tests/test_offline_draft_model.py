"""CPU checks for offline draft-head initialization."""

import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO, "verl-src"))

from verl.trainer.ppo.draft_model import (
    OfflineDraftTrainingConfig,
    _gather_last_valid_hidden,
    train_offline_draft_model,
)


class TinyPolicy(nn.Module):
    def __init__(self, vocab_size=23, hidden_size=16):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        del attention_mask, use_cache
        hidden, _ = self.rnn(self.embed(input_ids))
        logits = self.lm_head(hidden)
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=(hidden,))
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=4,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=0,
        **kwargs,
    ):
        del attention_mask, pad_token_id, kwargs
        seq = input_ids.repeat_interleave(num_return_sequences, dim=0)
        for _ in range(max_new_tokens):
            logits = self(seq, output_hidden_states=False).logits[:, -1, :]
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=-1)
        return seq


def test_offline_draft_training_does_not_update_policy_and_freezes_draft():
    torch.manual_seed(0)
    policy = TinyPolicy()
    before = {name: param.detach().clone() for name, param in policy.named_parameters()}
    prompts = [
        {"input_ids": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5])},
        {"input_ids": torch.tensor([6, 7, 8, 9])},
    ]
    config = OfflineDraftTrainingConfig(
        max_new_tokens=3,
        num_rollouts_per_prompt=2,
        prompt_batch_size=2,
        draft_batch_size=2,
        num_train_epochs=2,
        do_sample=False,
        pad_token_id=0,
        learning_rate=1e-2,
    )

    draft_model = train_offline_draft_model(policy, prompts, config=config)

    for name, param in policy.named_parameters():
        assert torch.equal(before[name], param.detach())
        assert param.grad is None
    assert all(not param.requires_grad for param in draft_model.parameters())
    assert draft_model.offline_init_metrics["num_rollout_sequences"] == 6.0
    assert draft_model.offline_init_metrics["train_steps"] > 0


def test_trained_draft_model_proposes_candidate_tokens():
    torch.manual_seed(1)
    policy = TinyPolicy()
    prompts = [{"input_ids": torch.tensor([1, 2, 3])}, {"input_ids": torch.tensor([3, 2, 1])}]
    config = OfflineDraftTrainingConfig(
        max_new_tokens=2,
        num_train_epochs=1,
        do_sample=False,
        pad_token_id=0,
    )
    draft_model = train_offline_draft_model(policy, prompts, config=config)

    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    attention_mask = torch.ones_like(input_ids)
    candidate = draft_model.propose_next_token(
        policy,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_candidates=2,
    )

    assert candidate.token_ids.shape == (2, 2)
    assert candidate.log_probs.shape == (2, 2)
    assert candidate.logits.shape == (2, policy.config.vocab_size)


def test_last_valid_hidden_supports_left_padding():
    hidden_states = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])

    gathered = _gather_last_valid_hidden(hidden_states, attention_mask)

    assert torch.equal(gathered[0], hidden_states[0, 4])
    assert torch.equal(gathered[1], hidden_states[1, 2])
