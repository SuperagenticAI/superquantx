"""SuperQuantX Client - Main interface for the SuperQuantX platform
"""

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, Field


class SuperQuantXConfig(BaseModel):
    """Configuration for SuperQuantX client"""

    api_key: str = Field(..., description="API key for authentication")
    base_url: str = Field(default="https://api.superquantx.ai", description="Base API URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")


class QuantumJob(BaseModel):
    """Represents a quantum computation job"""

    job_id: str
    status: str
    created_at: str
    circuit_id: str | None = None
    backend: str | None = None
    shots: int | None = None
    results: dict[str, Any] | None = None
    error: str | None = None


class SuperQuantXClient:
    """Main client for interacting with the SuperQuantX platform

    This client provides access to quantum computing resources, quantum algorithms,
    and quantum machine learning capabilities through the SuperQuantX API.
    """

    def __init__(self, config: SuperQuantXConfig | str | dict[str, Any]):
        """Initialize SuperQuantX client

        Args:
            config: Configuration object, API key string, or config dictionary

        """
        if isinstance(config, str):
            self.config = SuperQuantXConfig(api_key=config)
        elif isinstance(config, dict):
            self.config = SuperQuantXConfig(**config)
        else:
            self.config = config

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "SuperQuantX-Python-SDK/0.1.0"
            }
        )

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._client.aclose()

    def __enter__(self):
        """Sync context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        asyncio.run(self._client.aclose())

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make an authenticated request to the SuperQuantX API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            httpx.HTTPError: If request fails

        """
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=endpoint,
                    json=data,
                    params=params
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPError:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def health_check(self) -> dict[str, Any]:
        """Check API health and connectivity

        Returns:
            Health status information

        """
        return await self._request("GET", "/health")

    async def get_account_info(self) -> dict[str, Any]:
        """Get account information and usage statistics

        Returns:
            Account information including quotas and usage

        """
        return await self._request("GET", "/account")

    async def list_backends(self) -> list[dict[str, Any]]:
        """List available quantum backends

        Returns:
            List of available backends with their specifications

        """
        response = await self._request("GET", "/backends")
        return response.get("backends", [])

    async def get_backend_info(self, backend_name: str) -> dict[str, Any]:
        """Get detailed information about a specific backend

        Args:
            backend_name: Name of the backend

        Returns:
            Backend information including capabilities and status

        """
        return await self._request("GET", f"/backends/{backend_name}")

    async def submit_job(
        self,
        circuit_data: dict[str, Any],
        backend: str = "simulator",
        shots: int = 1024,
        optimization_level: int = 1,
        **kwargs
    ) -> QuantumJob:
        """Submit a quantum circuit for execution

        Args:
            circuit_data: Quantum circuit representation
            backend: Target backend for execution
            shots: Number of measurement shots
            optimization_level: Circuit optimization level (0-3)
            **kwargs: Additional execution parameters

        Returns:
            QuantumJob object representing the submitted job

        """
        job_data = {
            "circuit": circuit_data,
            "backend": backend,
            "shots": shots,
            "optimization_level": optimization_level,
            **kwargs
        }

        response = await self._request("POST", "/jobs", data=job_data)
        return QuantumJob(**response)

    async def get_job(self, job_id: str) -> QuantumJob:
        """Get job information and results

        Args:
            job_id: Job identifier

        Returns:
            QuantumJob object with current status and results

        """
        response = await self._request("GET", f"/jobs/{job_id}")
        return QuantumJob(**response)

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a running job

        Args:
            job_id: Job identifier

        Returns:
            Cancellation confirmation

        """
        return await self._request("DELETE", f"/jobs/{job_id}")

    async def list_jobs(
        self,
        limit: int = 100,
        status: str | None = None,
        backend: str | None = None
    ) -> list[QuantumJob]:
        """List user's jobs with optional filtering

        Args:
            limit: Maximum number of jobs to return
            status: Filter by job status
            backend: Filter by backend

        Returns:
            List of QuantumJob objects

        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        if backend:
            params["backend"] = backend

        response = await self._request("GET", "/jobs", params=params)
        return [QuantumJob(**job) for job in response.get("jobs", [])]

    async def wait_for_job(
        self,
        job_id: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> QuantumJob:
        """Wait for job completion with polling

        Args:
            job_id: Job identifier
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Completed QuantumJob object

        Raises:
            TimeoutError: If job doesn't complete within timeout

        """
        start_time = asyncio.get_event_loop().time()

        while True:
            job = await self.get_job(job_id)

            if job.status in ["completed", "failed", "cancelled"]:
                return job

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            await asyncio.sleep(poll_interval)

    # Synchronous wrappers for common operations
    def health_check_sync(self) -> dict[str, Any]:
        """Synchronous version of health_check"""
        return asyncio.run(self.health_check())

    def submit_job_sync(
        self,
        circuit_data: dict[str, Any],
        backend: str = "simulator",
        shots: int = 1024,
        **kwargs
    ) -> QuantumJob:
        """Synchronous version of submit_job"""
        return asyncio.run(self.submit_job(circuit_data, backend, shots, **kwargs))

    def get_job_sync(self, job_id: str) -> QuantumJob:
        """Synchronous version of get_job"""
        return asyncio.run(self.get_job(job_id))

    def wait_for_job_sync(self, job_id: str, timeout: int = 300) -> QuantumJob:
        """Synchronous version of wait_for_job"""
        return asyncio.run(self.wait_for_job(job_id, timeout))
