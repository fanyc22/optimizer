from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


def _as_fraction(value: str | float | int) -> float:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            return float(stripped[:-1]) / 100.0
        return float(stripped)
    return float(value)


class GlobalMetrics(BaseModel):
    makespan_us: float = Field(ge=0)
    total_energy_joules: float = Field(ge=0)
    peak_power_watts: float = Field(ge=0)
    thermal_violation: bool
    budget_utilization_percent: float = Field(ge=0)


class UtilizationStats(BaseModel):
    avg_utilization: float = Field(ge=0)
    bubble_time_percent: float = Field(ge=0)

    @field_validator("avg_utilization", "bubble_time_percent", mode="before")
    @classmethod
    def parse_percent(cls, value: str | float | int) -> float:
        return _as_fraction(value)


class CxlMemoryPoolStats(BaseModel):
    avg_access_latency_ns: float = Field(ge=0)
    conflict_rate: float = Field(ge=0)

    @field_validator("conflict_rate", mode="before")
    @classmethod
    def parse_percent(cls, value: str | float | int) -> float:
        return _as_fraction(value)


class MemoryProfile(BaseModel):
    local_hbm_bandwidth_util: float = Field(ge=0)
    cxl_memory_pool: CxlMemoryPoolStats

    @field_validator("local_hbm_bandwidth_util", mode="before")
    @classmethod
    def parse_percent(cls, value: str | float | int) -> float:
        return _as_fraction(value)


class LinkBottleneck(BaseModel):
    link_id: str
    utilization: float = Field(ge=0)
    queue_depth_pkts: int = Field(ge=0)
    dominant_traffic: str

    @field_validator("utilization", mode="before")
    @classmethod
    def parse_percent(cls, value: str | float | int) -> float:
        return _as_fraction(value)


class NetworkProfile(BaseModel):
    average_link_utilization: float = Field(ge=0)
    top_bottlenecks: list[LinkBottleneck]

    @field_validator("average_link_utilization", mode="before")
    @classmethod
    def parse_percent(cls, value: str | float | int) -> float:
        return _as_fraction(value)


class SimulationFeedback(BaseModel):
    simulation_id: str
    workload: str
    global_metrics: GlobalMetrics
    compute_profile: dict[str, UtilizationStats]
    memory_profile: MemoryProfile
    network_profile: NetworkProfile
