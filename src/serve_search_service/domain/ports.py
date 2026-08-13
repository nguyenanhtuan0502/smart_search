"""Port interfaces for the search service."""

from typing import Protocol


class TenantResolver(Protocol):
    def resolve(self, headers: dict[str, str]) -> str: ...
