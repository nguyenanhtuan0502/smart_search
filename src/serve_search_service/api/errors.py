
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..logging import current_trace_id


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
    
    return error_response(
        request,
        status=422,
        code="validation_error",
        detail=detail,
        retryable=False,
    )
