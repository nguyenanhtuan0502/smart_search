"""Public HTTP response models — shared across APIs."""

from pydantic import BaseModel, ConfigDict


class ProblemDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None
    code: str
    trace_id: str | None = None
    retryable: bool = False
