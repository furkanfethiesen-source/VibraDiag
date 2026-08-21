"""
VibraDiag API — Global Exception & Error Handlers.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.schemas.common import ErrorDetail, ErrorResponse, ResponseStatus


def setup_exception_handlers(app: FastAPI) -> None:
    """Registers standard JSON exception handlers on the FastAPI application."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}")
        error_resp = ErrorResponse(
            status=ResponseStatus.ERROR,
            error=ErrorDetail(
                code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                details=getattr(exc, "headers", None),
            ),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
        error_resp = ErrorResponse(
            status=ResponseStatus.ERROR,
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="İstek parametreleri veya JSON şeması doğrulanamadı.",
                details=exc.errors(),
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
        error_resp = ErrorResponse(
            status=ResponseStatus.ERROR,
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message=f"Sunucu tarafında beklenmeyen bir hata oluştu: {exc!s}",
                details=None,
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_resp.model_dump(),
        )
