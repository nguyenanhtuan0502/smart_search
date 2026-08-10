"""Public HTTP routes for the search service."""

from fastapi import APIRouter, Depends, Request, status

from ..application.search import SearchService
from ..contracts import SearchResponse, SearchResult
from ..domain.ports import TenantResolver
from ..domain.search import SearchQuery
from .errors import ApiError
from .schemas import SearchRequest

router = APIRouter()


def tenant_id(request: Request) -> str:
    resolver: TenantResolver = request.app.state.tenant_resolver
    return resolver.resolve(dict(request.headers))


@router.get("/health/live", tags=["platform"])
async def live() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready", tags=["platform"])
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@router.get("/health/startup", tags=["platform"])
async def startup() -> dict[str, str]:
    return {"status": "started"}


@router.get("/v1", tags=["platform"])
async def api_root(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    return {
        "service": "serve-search-service",
        "version": settings.service_version,
        "api_version": "v1",
    }


@router.post("/v1/search", response_model=SearchResponse, tags=["search"])
async def search(
    payload: SearchRequest,
    request: Request,
    verified_tenant_id: str = Depends(tenant_id),
) -> SearchResponse:
    search_service: SearchService | None = request.app.state.search_service
    if search_service is None:
        raise ApiError(
            status=status.HTTP_501_NOT_IMPLEMENTED,
            code="search_not_configured",
            detail="No retriever adapter is configured.",
        )
    hits = await search_service.search(
        SearchQuery(
            tenant_id=verified_tenant_id,
            text=payload.query,
            limit=payload.limit,
            filters=payload.filters,
        )
    )
    return SearchResponse(
        results=[
            SearchResult(
                document_id=hit.document_id,
                passage_id=hit.passage_id,
                text=hit.text,
                score=hit.score,
                source_uri=hit.source_uri,
                metadata=dict(hit.metadata),
            )
            for hit in hits
        ],
        trace_id=request.state.trace_id,
    )
