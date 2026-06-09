from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from codesign_optimizer.optimizer.tgrl_v2.observation import (
    ACTION_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphObservation,
)


@dataclass(frozen=True)
class TensorObservation:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    global_features: torch.Tensor
    action_features: torch.Tensor
    action_target_indices: torch.Tensor
    heuristic_logits: torch.Tensor


def tensorize_observation(observation: GraphObservation, device: torch.device) -> TensorObservation:
    edge_count = len(observation.edge_features)
    action_count = len(observation.action_features)
    return TensorObservation(
        node_features=torch.tensor(observation.node_features, dtype=torch.float32, device=device),
        edge_index=torch.tensor(observation.edge_index, dtype=torch.long, device=device)
        if edge_count
        else torch.empty((2, 0), dtype=torch.long, device=device),
        edge_features=torch.tensor(observation.edge_features, dtype=torch.float32, device=device)
        if edge_count
        else torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device),
        global_features=torch.tensor(observation.global_features, dtype=torch.float32, device=device),
        action_features=torch.tensor(observation.action_features, dtype=torch.float32, device=device)
        if action_count
        else torch.empty((0, ACTION_FEATURE_DIM), dtype=torch.float32, device=device),
        action_target_indices=torch.tensor(observation.action_target_indices, dtype=torch.long, device=device)
        if action_count
        else torch.empty((0,), dtype=torch.long, device=device),
        heuristic_logits=torch.tensor(observation.heuristic_logits, dtype=torch.float32, device=device)
        if action_count
        else torch.empty((0,), dtype=torch.float32, device=device),
    )


class TGRLGNNPolicy(nn.Module):
    def __init__(
        self,
        *,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        action_dim: int = ACTION_FEATURE_DIM,
        hidden_dim: int = 96,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_encoders = nn.ModuleList(nn.Linear(edge_dim, hidden_dim) for _ in range(layers))
        self.message_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: TensorObservation) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.action_features.shape[0] == 0:
            raise ValueError("TGRLGNNPolicy requires at least one valid action")
        h = F.relu(self.node_encoder(observation.node_features))
        for layer_idx in range(self.layers):
            h = self._message_pass(
                h,
                observation.edge_index,
                observation.edge_features,
                edge_layer=self.edge_encoders[layer_idx],
                message_layer=self.message_layers[layer_idx],
                self_layer=self.self_layers[layer_idx],
            )
        graph_embedding = 0.5 * (h[0] + h.mean(dim=0))
        target_embeddings = h[observation.action_target_indices.clamp(min=0, max=h.shape[0] - 1)]
        action_embeddings = F.relu(self.action_encoder(observation.action_features))
        graph_for_actions = graph_embedding.unsqueeze(0).expand(action_embeddings.shape[0], -1)
        actor_input = torch.cat(
            [
                graph_for_actions,
                target_embeddings,
                action_embeddings,
                observation.heuristic_logits.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.actor(actor_input).squeeze(-1)
        value = self.critic(torch.cat([graph_embedding, observation.global_features], dim=-1)).squeeze(-1)
        return logits, value

    def _message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        *,
        edge_layer: nn.Linear,
        message_layer: nn.Linear,
        self_layer: nn.Linear,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return F.relu(self_layer(h))
        src = edge_index[0]
        dst = edge_index[1]
        messages = message_layer(h[src]) + edge_layer(edge_features)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
        agg = agg / degree.clamp_min(1.0)
        return F.relu(self_layer(h) + agg)


def policy_distribution(
    model: TGRLGNNPolicy,
    observation: GraphObservation,
    *,
    device: torch.device,
    heuristic_weight: float,
) -> tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor, TensorObservation]:
    tensor_observation = tensorize_observation(observation, device)
    dist, logits, value = policy_distribution_from_tensor(
        model,
        tensor_observation,
        heuristic_weight=heuristic_weight,
    )
    return dist, logits, value, tensor_observation


def policy_distribution_from_tensor(
    model: TGRLGNNPolicy,
    tensor_observation: TensorObservation,
    *,
    heuristic_weight: float,
) -> tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor]:
    actor_logits, value = model(tensor_observation)
    logits = actor_logits + heuristic_weight * tensor_observation.heuristic_logits
    dist = torch.distributions.Categorical(logits=logits)
    return dist, logits, value
