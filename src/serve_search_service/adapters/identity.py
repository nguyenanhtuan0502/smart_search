"""Temporary local tenant resolver; production must replace it with verified identity context."""


class DevelopmentHeaderTenantResolver:
    def __init__(self, default_tenant_id: str) -> None:
        self._default_tenant_id = default_tenant_id

    def resolve(self, headers: dict[str, str]) -> str:
        # This is only valid in local development behind no public listener.
        return headers.get("x-verified-tenant-id", self._default_tenant_id)
