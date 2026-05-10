from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class NodeTypeSpec(BaseModel):
    compute_teraflops_dense: float | None = None
    compute_teraflops_sparse: float | None = None
    local_memory_gb: float | None = None
    peak_tflops: float | None = None
    memory_bw_gbps: float | None = None
    memory_latency_ns: float | None = None
    slots: int | None = None
    max_parallelism: int | None = None
    comm_slots: int | None = None
    tdp_watts: float = Field(default=0.0, ge=0)
    cost_unit: float = Field(default=0.0, ge=0)
    role: str | None = None
    rack_units: float = Field(default=1.0, ge=0)
    capacity_gb: float | None = None
    radix: int | None = None
    area_mm2: float | None = None


class LinkTypeSpec(BaseModel):
    bandwidth_gbps: float = Field(ge=0)
    latency_ns: float = Field(ge=0)
    protocol: str
    cost_unit: float = Field(default=0.0, ge=0)
    lanes: int = Field(default=1, ge=1)
    technology: str | None = None
    level: str | None = None


class InstantiatedNode(BaseModel):
    node_id: str
    type: str


class InstantiatedLink(BaseModel):
    src: str
    dst: str
    link_type: str
    qty: int = Field(default=1, ge=1)


class ComponentLibrary(BaseModel):
    node_types: dict[str, NodeTypeSpec]
    link_types: dict[str, LinkTypeSpec]


class SystemInstantiation(BaseModel):
    nodes: list[InstantiatedNode]
    topology: list[InstantiatedLink]


class HardwareProposal(BaseModel):
    optimizer_version: str
    iteration: int = Field(ge=0)
    component_library: ComponentLibrary
    system_instantiation: SystemInstantiation

    @model_validator(mode="after")
    def validate_references(self) -> "HardwareProposal":
        node_type_names = set(self.component_library.node_types.keys())
        link_type_names = set(self.component_library.link_types.keys())
        node_ids = {n.node_id for n in self.system_instantiation.nodes}

        for node in self.system_instantiation.nodes:
            if node.type not in node_type_names:
                raise ValueError(f"Node type '{node.type}' not found in component library.")

        for edge in self.system_instantiation.topology:
            if edge.link_type not in link_type_names:
                raise ValueError(f"Link type '{edge.link_type}' not found in component library.")
            if edge.src not in node_ids or edge.dst not in node_ids:
                raise ValueError(f"Topology edge uses undefined node: {edge.src} -> {edge.dst}.")

        return self

    def total_node_cost(self) -> float:
        node_types = self.component_library.node_types
        return sum(node_types[n.type].cost_unit for n in self.system_instantiation.nodes)

    def total_link_cost(self) -> float:
        link_types = self.component_library.link_types
        return sum(link_types[e.link_type].cost_unit * e.qty for e in self.system_instantiation.topology)

    def total_estimated_cost(self) -> float:
        return self.total_node_cost() + self.total_link_cost()

    def estimated_peak_power(self) -> float:
        node_types = self.component_library.node_types
        return sum(node_types[n.type].tdp_watts for n in self.system_instantiation.nodes)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
