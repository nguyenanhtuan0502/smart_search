"""ASGI application for the search service."""

from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from .adapters.identity import DevelopmentHeaderTenantResolver
from .api.errors import ApiError, handle_api_error, handle_validation_error
from .api.router import router
from .application.search import SearchService
from .logging import bind_log_context, configure_logging, reset_log_context
from .settings import ServiceSettings
from .telemetry import configure_observability

SERVICE_NAME = "serve-search-service"


def create_app(
    settings: ServiceSettings | None = None, search_service: SearchService | None = None
) -> FastAPI:
    settings = settings or ServiceSettings()
    configure_logging(app_name=SERVICE_NAME, log_directory=settings.log_directory)
    configure_observability(
        service_name=SERVICE_NAME,
        service_version=settings.service_version,
        environment=settings.deployment_environment,
    )
    app = FastAPI(title="Serve Search Service", version=settings.service_version)
    app.state.settings = settings
    app.state.search_service = search_service
    app.state.tenant_resolver = DevelopmentHeaderTenantResolver(settings.dev_tenant_id)
    app.include_router(router)
    app.add_exception_handler(ApiError, handle_api_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)

    @app.middleware("http")
    async def set_request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        trace_id = request.headers.get("X-Trace-Id", str(uuid4()))
        tokens = bind_log_context(trace_id, SERVICE_NAME)
        request.state.trace_id = trace_id
        try:
            response = await call_next(request)
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            reset_log_context(tokens)

    return app


app = create_app()
