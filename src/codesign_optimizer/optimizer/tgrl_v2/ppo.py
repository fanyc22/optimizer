from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import torch
import torch.nn.functional as F

from codesign_optimizer.optimizer.tgrl_v2.model import (
    TensorObservation,
    TGRLGNNPolicy,
    batch_tensor_observations,
    policy_logits_from_batched_tensor,
    tensorize_observation,
)
from codesign_optimizer.optimizer.tgrl_v2.observation import GraphObservation


@dataclass
class PPOTransition:
    observation: GraphObservation
    action_index: int
    old_logprob: float
    value: float
    reward: float
    done: bool
    candidate_signature: str
    episode_env: int
    rollout_step: int
    advantage: float = 0.0
    return_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "old_logprob": self.old_logprob,
            "value": self.value,
            "reward": self.reward,
            "done": self.done,
            "candidate_signature": self.candidate_signature,
            "episode_env": self.episode_env,
            "rollout_step": self.rollout_step,
            "advantage": self.advantage,
            "return_value": self.return_value,
            "action": self.observation.masked_actions[self.action_index].action.to_dict(),
        }


@dataclass(frozen=True)
class PPOConfig:
    ppo_epochs: int = 4
    minibatch_size: int = 16
    gamma: float = 0.95
    gae_lambda: float = 0.90
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_weight: float = 0.1
    learning_rate: float = 3e-4
    heuristic_weight: float = 1.0
    reward_clip: float = 5.0


@dataclass(frozen=True)
class _PreparedPPOTransition:
    tensor_observation: TensorObservation
    action: torch.Tensor
    old_logprob: torch.Tensor
    advantage: torch.Tensor
    return_value: torch.Tensor
    prior_probs: torch.Tensor
    action_count: int


def attach_gae(
    trajectories: list[list[PPOTransition]],
    *,
    gamma: float,
    gae_lambda: float,
) -> list[PPOTransition]:
    flattened: list[PPOTransition] = []
    for trajectory in trajectories:
        next_value = 0.0
        next_advantage = 0.0
        for transition in reversed(trajectory):
            mask = 0.0 if transition.done else 1.0
            delta = transition.reward + gamma * next_value * mask - transition.value
            advantage = delta + gamma * gae_lambda * mask * next_advantage
            transition.advantage = advantage
            transition.return_value = advantage + transition.value
            next_value = transition.value
            next_advantage = advantage
        flattened.extend(trajectory)
    return flattened


def ppo_update(
    *,
    model: TGRLGNNPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[PPOTransition],
    config: PPOConfig,
    device: torch.device,
    rng: random.Random,
) -> dict[str, float]:
    if not transitions:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_prior": 0.0,
            "loss": 0.0,
            "updates": 0.0,
        }
    advantages = torch.tensor([item.advantage for item in transitions], dtype=torch.float32)
    adv_mean = float(advantages.mean())
    adv_std = float(advantages.std(unbiased=False))
    if adv_std > 1e-9:
        for item in transitions:
            item.advantage = (item.advantage - adv_mean) / adv_std

    prepared = _prepare_transitions(transitions, device=device)
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl_prior": 0.0,
        "loss": 0.0,
        "updates": 0.0,
    }
    indices = list(range(len(transitions)))
    for _ in range(config.ppo_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), max(1, config.minibatch_size)):
            batch_indices = indices[start : start + max(1, config.minibatch_size)]
            optimizer.zero_grad()
            batch_losses: list[torch.Tensor] = []
            batch_policy_loss = 0.0
            batch_value_loss = 0.0
            batch_entropy = 0.0
            batch_kl = 0.0
            prepared_batch = [prepared[idx] for idx in batch_indices]
            batched_observation = batch_tensor_observations(
                [item.tensor_observation for item in prepared_batch]
            )
            batched_logits, batched_values = policy_logits_from_batched_tensor(
                model,
                batched_observation,
                heuristic_weight=config.heuristic_weight,
            )
            action_offset = 0
            for local_idx, prepared_transition in enumerate(prepared_batch):
                next_action_offset = action_offset + prepared_transition.action_count
                dist = torch.distributions.Categorical(
                    logits=batched_logits[action_offset:next_action_offset]
                )
                value = batched_values[local_idx]
                new_logprob = dist.log_prob(prepared_transition.action)
                ratio = torch.exp(new_logprob - prepared_transition.old_logprob)
                unclipped = ratio * prepared_transition.advantage
                clipped = (
                    torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                    * prepared_transition.advantage
                )
                policy_loss = -torch.min(unclipped, clipped)
                value_loss = F.mse_loss(value, prepared_transition.return_value)
                entropy = dist.entropy()
                policy_probs = dist.probs.clamp_min(1e-9)
                kl_prior = torch.sum(
                    policy_probs * (torch.log(policy_probs) - torch.log(prepared_transition.prior_probs))
                )
                action_offset = next_action_offset
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                    + config.kl_weight * kl_prior
                )
                batch_losses.append(loss)
                batch_policy_loss += float(policy_loss.detach())
                batch_value_loss += float(value_loss.detach())
                batch_entropy += float(entropy.detach())
                batch_kl += float(kl_prior.detach())
            if not batch_losses:
                continue
            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            denom = max(1, len(batch_losses))
            metrics["policy_loss"] += batch_policy_loss / denom
            metrics["value_loss"] += batch_value_loss / denom
            metrics["entropy"] += batch_entropy / denom
            metrics["kl_prior"] += batch_kl / denom
            metrics["loss"] += float(total_loss.detach())
            metrics["updates"] += 1.0

    updates = max(1.0, metrics["updates"])
    for key in ["policy_loss", "value_loss", "entropy", "kl_prior", "loss"]:
        metrics[key] /= updates
    if math.isnan(metrics["loss"]):
        raise RuntimeError("PPO update produced NaN loss")
    return metrics


def _prepare_transitions(transitions: list[PPOTransition], *, device: torch.device) -> list[_PreparedPPOTransition]:
    prepared: list[_PreparedPPOTransition] = []
    for transition in transitions:
        tensor_observation = tensorize_observation(transition.observation, device)
        prepared.append(
            _PreparedPPOTransition(
                tensor_observation=tensor_observation,
                action=torch.tensor(transition.action_index, dtype=torch.long, device=device),
                old_logprob=torch.tensor(transition.old_logprob, dtype=torch.float32, device=device),
                advantage=torch.tensor(transition.advantage, dtype=torch.float32, device=device),
                return_value=torch.tensor(transition.return_value, dtype=torch.float32, device=device),
                prior_probs=torch.softmax(tensor_observation.heuristic_logits, dim=0).clamp_min(1e-9),
                action_count=int(tensor_observation.action_features.shape[0]),
            )
        )
    return prepared
