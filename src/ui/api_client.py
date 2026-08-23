"""
API Client for VibraDiag backend.
Provides a synchronous httpx wrapper for Streamlit applications.
"""
import json
from typing import Any

import httpx
import streamlit as st
from loguru import logger


class VibraDiagAPIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message: str, status_code: int | None = None, detail: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail

class VibraDiagClient:
    """Synchronous HTTP client for VibraDiag backend API."""

    def __init__(self, base_url: str = 'http://localhost:8000', timeout: float = 120.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the underlying HTTP client session."""
        self.client.close()

    def _handle_error(self, response: httpx.Response):
        """Helper to handle HTTP errors uniformly."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except ValueError:
                detail = e.response.text

            logger.error(f"API Error {e.response.status_code}: {detail}")
            raise VibraDiagAPIError(
                message=f"API Request failed: {e.response.status_code}",
                status_code=e.response.status_code,
                detail=detail
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error: {e!s}")
            raise VibraDiagAPIError(message=f"Request failed: {e!s}") from e

    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Internal helper to make a request and handle errors."""
        try:
            response = self.client.request(method, endpoint, **kwargs)
            self._handle_error(response)
            return response.json()
        except Exception as e:
            if not isinstance(e, VibraDiagAPIError):
                logger.error(f"Failed to call {method} {endpoint}: {e}")
                raise VibraDiagAPIError(message=str(e)) from e
            raise

    # 1. Health Check
    def get_healthz(self) -> dict[str, Any]:
        return self._request('GET', '/healthz')

    # 2. Readiness Check
    def get_readyz(self) -> dict[str, Any]:
        return self._request('GET', '/readyz')

    # 3. Upload Signal
    def upload_signal(
        self,
        file_content: bytes,
        filename: str,
        fs: float | None = None,
        rpm: float | None = None
    ) -> dict[str, Any]:
        files = {'file': (filename, file_content)}
        data = {}
        if fs is not None:
            data['fs'] = str(fs)
        if rpm is not None:
            data['rpm'] = str(rpm)

        return self._request('POST', '/api/v1/signals/upload', files=files, data=data)

    # 4. Analyze Signal
    def analyze_signal(
        self,
        signal_id: str,
        signal_file_path: str,
        machine_metadata: dict[str, Any],
        plot_format: str = 'html'
    ) -> dict[str, Any]:
        payload = {
            'signal_id': signal_id,
            'signal_file_path': signal_file_path,
            'machine_metadata': machine_metadata,
            'plot_format': plot_format
        }
        return self._request('POST', '/api/v1/signals/analyze', json=payload)

    # 5. Get Signal Plots
    def get_signal_plots(
        self,
        signal_id: str,
        signal_file_path: str,
        machine_metadata: dict[str, Any],
        plot_format: str = 'html'
    ) -> dict[str, Any]:
        payload = {
            'signal_id': signal_id,
            'signal_file_path': signal_file_path,
            'machine_metadata': machine_metadata,
            'plot_format': plot_format
        }
        return self._request('POST', '/api/v1/signals/plots', json=payload)

    # 6. Diagnose (full pipeline)
    def diagnose(
        self,
        query: str,
        signal_id: str | None = None,
        signal_file_path: str | None = None,
        session_id: str | None = None,
        machine_metadata: dict[str, Any] | None = None,
        plot_format: str = 'html'
    ) -> dict[str, Any]:
        payload = {
            'query': query,
            'plot_format': plot_format
        }
        if signal_id:
            payload['signal_id'] = signal_id
        if signal_file_path:
            payload['signal_file_path'] = signal_file_path
        if session_id:
            payload['session_id'] = session_id
        if machine_metadata:
            payload['machine_metadata'] = machine_metadata

        return self._request('POST', '/api/v1/diagnose', json=payload)

    # 7. Diagnose Direct (multipart)
    def diagnose_direct(
        self,
        file_content: bytes,
        filename: str,
        query: str,
        session_id: str | None = None,
        fs: float | None = None,
        rpm: float | None = None,
        machine_type: str | None = None,
        machine_class: str | None = None,
        machine_metadata: dict[str, Any] | None = None,
        plot_format: str = 'html'
    ) -> dict[str, Any]:
        files = {'file': (filename, file_content)}
        data = {
            'query': query,
            'plot_format': plot_format
        }
        if session_id:
            data['session_id'] = session_id
        if fs is not None:
            data['fs'] = str(fs)
        if rpm is not None:
            data['rpm'] = str(rpm)
        if machine_type:
            data['machine_type'] = machine_type
        if machine_class:
            data['machine_class'] = machine_class
        if machine_metadata:
            data['machine_metadata_json'] = json.dumps(machine_metadata)

        return self._request('POST', '/api/v1/diagnose/direct', files=files, data=data)

    # 8. List Recent Sessions
    def list_recent_sessions(self, limit: int = 5) -> list[dict[str, Any]]:
        try:
            res = self._request('GET', f'/api/v1/sessions?limit={limit}')
            return res.get('sessions', [])
        except Exception:
            return []

    # 9. Get Session
    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request('GET', f'/api/v1/sessions/{session_id}')

    # 10. Get Session Messages
    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._request('GET', f'/api/v1/sessions/{session_id}/messages')

    # 11. Delete Session
    def delete_session(self, session_id: str) -> dict[str, Any]:
        return self._request('DELETE', f'/api/v1/sessions/{session_id}')

    # 11. Get Config Thresholds
    def get_config_thresholds(self) -> dict[str, Any]:
        return self._request('GET', '/api/v1/config/thresholds')

    # 12. Get Config Info
    def get_config_info(self) -> dict[str, Any]:
        return self._request('GET', '/api/v1/config/info')


@st.cache_resource
def get_client(base_url: str = 'http://localhost:8000') -> VibraDiagClient:
    """
    Returns a cached singleton instance of VibraDiagClient.
    """
    return VibraDiagClient(base_url=base_url)
