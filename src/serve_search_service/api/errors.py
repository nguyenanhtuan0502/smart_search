
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..logging import current_trace_id, get_logger

_logger = get_logger(__name__)


class ApiError(Exception):
    """Custom API error with support for retryable flag."""
    
    def __init__(
        self,
        *,
        status: int,
        code: str,
        detail: str,
        retryable: bool = False,
    ) -> None:
        self.status = status
        self.code = code
        self.detail = detail
        self.retryable = retryable
        super().__init__(detail)


def error_response(
    request: Request,
    *,
    status: int,
    code: str,
    detail: str,
    retryable: bool = False,
) -> JSONResponse:
    """Build unified error response.
    
    Args:
        request: FastAPI request object
        status: HTTP status code
        code: Error code identifier
        detail: Human-readable error detail
        retryable: Whether the error is retryable
    """
    body = {
        "status": status,
        "error": {
            "code": code,
            "detail": detail,
            "instance": request.url.path,
            "trace_id": current_trace_id(),
            "retryable": retryable,
        },
    }
    return JSONResponse(status_code=status, content=body)


async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    """Handle ApiError exceptions."""
    _logger.error(
        "API error on %s: status=%d, code=%s, detail=%s, retryable=%s",
        request.url.path,
        exc.status,
        exc.code,
        exc.detail,
        exc.retryable,
    )
    return error_response(
        request,
        status=exc.status,
        code=exc.code,
        detail=exc.detail,
        retryable=exc.retryable,
    )


async def handle_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    # Extract the first error message for simplicity
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        detail = first_error.get("msg", "Validation failed.")
    else:
        detail = "Validation failed."

    # Log all validation errors with field locations for debugging
    # (e.g. weights must sum to 1.0, limit out of range, etc.)
    _logger.warning(
        "Validation error on %s: %d error(s): %s",
        request.url.path,
        len(errors),
        [
            {"loc": err.get("loc"), "msg": err.get("msg")}
            for err in errors
        ],
    )

    return error_response(
        request,
        status=422,
        code="validation_error",
        detail=detail,
        retryable=False,
    )
