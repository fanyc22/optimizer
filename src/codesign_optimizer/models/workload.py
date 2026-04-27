from __future__ import annotations

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    task_id: str
    task_type: str
    flops_dense: float = Field(default=0.0, ge=0)
    flops_sparse: float = Field(default=0.0, ge=0)
    memory_gb: float = Field(default=0.0, ge=0)
    communication_weight: float = Field(default=1.0, ge=0)


class WorkloadSpec(BaseModel):
    workload_name: str
    tasks: list[TaskSpec]
