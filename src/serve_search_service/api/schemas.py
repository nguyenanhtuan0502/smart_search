"""Client-facing search DTOs without provider or infrastructure credentials."""

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4096)
    limit: int = Field(default=10, ge=1, le=100)
    filters: dict[str, str] = Field(default_factory=dict)
