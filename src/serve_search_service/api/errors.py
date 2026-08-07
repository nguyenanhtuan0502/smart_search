"""RFC 9457-compatible errors for the search API."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..contracts import ProblemDetail
from ..logging import current_trace_id


class ApiError(Exception):
    def __init__(self, *, status: int, code: str, detail: str, retryable: bool = False) -> None:
        self.status = status
        self.code = code
        self.detail = detail
        self.retryable = retryable


def problem_response(
    request: Request, *, status: int, code: str, detail: str, retryable: bool = False
) -> JSONResponse:
    detail_model = ProblemDetail(
        type=f"https://serve-search-service.dev/problems/{code}",
        title="Request failed",
        status=status,
        detail=detail,
        instance=str(request.url.path),
        code=code,
        trace_id=current_trace_id(),
        retryable=retryable,
    )
    return JSONResponse(status_code=status, content=detail_model.model_dump(mode="json"))


async def handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return problem_response(
        request, status=exc.status, code=exc.code, detail=exc.detail, retryable=exc.retryable
    )


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return problem_response(
        request, status=422, code="validation_error", detail="Request validation failed."
    )
