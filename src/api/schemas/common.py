"""
VibraDiag API — Common Schemas & Enums.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResponseStatus(StrEnum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class PlotFormat(StrEnum):
    JSON = "json"
    HTML = "html"


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine readable error code")
    message: str = Field(..., description="Human readable error message")
    details: Any = Field(
        default=None, description="Detailed diagnostic or validation errors"
    )


class ErrorResponse(BaseModel):
    status: ResponseStatus = ResponseStatus.ERROR
    error: ErrorDetail
