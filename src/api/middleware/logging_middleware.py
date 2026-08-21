"""
VibraDiag API — Logging & Request ID Middleware.
"""

from __future__ import annotations

import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds X-Request-ID and X-Process-Time headers, and logs request lifecycle."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:10]}"
        request.state.request_id = request_id

        start_time = time.perf_counter()
        logger.info(f"--> [{request_id}] {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            process_time_ms = (time.perf_counter() - start_time) * 1000.0

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"

            logger.info(
                f"<-- [{request_id}] {request.method} {request.url.path} "
                f"Status={response.status_code} Time={process_time_ms:.2f}ms"
            )
            return response
        except Exception as e:
            process_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"<-- [{request_id}] {request.method} {request.url.path} "
                f"FAILED: {e} Time={process_time_ms:.2f}ms"
            )
            raise
