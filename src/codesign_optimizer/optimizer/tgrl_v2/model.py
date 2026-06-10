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


@dataclass(frozen=True)
class BatchedTensorObservation:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    global_features: torch.Tensor
    action_features: torch.Tensor
    action_target_indices: torch.Tensor
    heuristic_logits: torch.Tensor
    node_graph_index: torch.Tensor
    root_node_indices: torch.Tensor
    action_graph_index: torch.Tensor
    action_offsets: torch.Tensor

    @property
    def graph_count(self) -> int:
        return int(self.global_features.shape[0])


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


def batch_tensor_observations(observations: list[TensorObservation]) -> BatchedTensorObservation:
    if not observations:
        raise ValueError("batch_tensor_observations requires at least one observation")
    device = observations[0].node_features.device
    dtype = observations[0].node_features.dtype
    edge_index_parts: list[torch.Tensor] = []
    edge_feature_parts: list[torch.Tensor] = []
    action_feature_parts: list[torch.Tensor] = []
    action_target_parts: list[torch.Tensor] = []
    heuristic_parts: list[torch.Tensor] = []
    node_graph_parts: list[torch.Tensor] = []
    action_graph_parts: list[torch.Tensor] = []
    root_node_indices: list[int] = []
    action_offsets = [0]
    node_offset = 0
    action_offset = 0
    for graph_idx, observation in enumerate(observations):
        if observation.node_features.device != device:
            raise ValueError("all observations in a batch must be on the same device")
        node_count = int(observation.node_features.shape[0])
        action_count = int(observation.action_features.shape[0])
        if node_count <= 0:
            raise ValueError("batched graph observations require at least one node per graph")
        root_node_indices.append(node_offset)
        node_graph_parts.append(torch.full((node_count,), graph_idx, dtype=torch.long, device=device))
        if observation.edge_index.numel():
            edge_index_parts.append(observation.edge_index + node_offset)
            edge_feature_parts.append(observation.edge_features)
        if action_count:
            max_node = node_count - 1
            action_feature_parts.append(observation.action_features)
            action_target_parts.append(observation.action_target_indices.clamp(min=0, max=max_node) + node_offset)
            heuristic_parts.append(observation.heuristic_logits)
            action_graph_parts.append(torch.full((action_count,), graph_idx, dtype=torch.long, device=device))
        node_offset += node_count
        action_offset += action_count
        action_offsets.append(action_offset)

    node_features = torch.cat([observation.node_features for observation in observations], dim=0)
    global_features = torch.stack([observation.global_features for observation in observations], dim=0)
    edge_index = (
        torch.cat(edge_index_parts, dim=1)
        if edge_index_parts
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    edge_features = (
        torch.cat(edge_feature_parts, dim=0)
        if edge_feature_parts
        else torch.empty((0, EDGE_FEATURE_DIM), dtype=dtype, device=device)
    )
    action_features = (
        torch.cat(action_feature_parts, dim=0)
        if action_feature_parts
        else torch.empty((0, ACTION_FEATURE_DIM), dtype=dtype, device=device)
    )
    action_target_indices = (
        torch.cat(action_target_parts, dim=0)
        if action_target_parts
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    heuristic_logits = (
        torch.cat(heuristic_parts, dim=0)
        if heuristic_parts
        else torch.empty((0,), dtype=dtype, device=device)
    )
    action_graph_index = (
        torch.cat(action_graph_parts, dim=0)
        if action_graph_parts
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    return BatchedTensorObservation(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_features,
        action_features=action_features,
        action_target_indices=action_target_indices,
        heuristic_logits=heuristic_logits,
        node_graph_index=torch.cat(node_graph_parts, dim=0),
        root_node_indices=torch.tensor(root_node_indices, dtype=torch.long, device=device),
        action_graph_index=action_graph_index,
        action_offsets=torch.tensor(action_offsets, dtype=torch.long, device=device),
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

    def forward_batched(self, observation: BatchedTensorObservation) -> tuple[torch.Tensor, torch.Tensor]:
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
        graph_sums = torch.zeros(
            (observation.graph_count, h.shape[-1]),
            dtype=h.dtype,
            device=h.device,
        )
        graph_sums.index_add_(0, observation.node_graph_index, h)
        graph_counts = torch.bincount(
            observation.node_graph_index,
            minlength=observation.graph_count,
        ).to(dtype=h.dtype, device=h.device)
        graph_mean = graph_sums / graph_counts.clamp_min(1.0).unsqueeze(-1)
        graph_embedding = 0.5 * (h[observation.root_node_indices] + graph_mean)
        target_embeddings = h[observation.action_target_indices]
        action_embeddings = F.relu(self.action_encoder(observation.action_features))
        graph_for_actions = graph_embedding[observation.action_graph_index]
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
        values = self.critic(torch.cat([graph_embedding, observation.global_features], dim=-1)).squeeze(-1)
        return logits, values

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


def policy_logits_from_batched_tensor(
    model: TGRLGNNPolicy,
    tensor_observation: BatchedTensorObservation,
    *,
    heuristic_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    actor_logits, values = model.forward_batched(tensor_observation)
    logits = actor_logits + heuristic_weight * tensor_observation.heuristic_logits
    return logits, values
