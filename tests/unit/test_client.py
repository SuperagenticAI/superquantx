"""
Unit tests for SuperQuantX client
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from superquantx.client import QuantumJob, SuperQuantXClient, SuperQuantXConfig


class TestSuperQuantXConfig:
    """Test SuperQuantX configuration"""

    def test_config_creation(self):
        """Test configuration creation"""
        config = SuperQuantXConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.base_url == "https://api.superquantx.ai"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.verify_ssl is True

    def test_config_with_custom_values(self):
        """Test configuration with custom values"""
        config = SuperQuantXConfig(
            api_key="test-key",
            base_url="https://custom.api.com",
            timeout=60,
            max_retries=5,
            verify_ssl=False
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.com"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.verify_ssl is False


class TestQuantumJob:
    """Test QuantumJob model"""

    def test_job_creation(self):
        """Test job creation"""
        job = QuantumJob(
            job_id="test-123",
            status="running",
            created_at="2023-01-01T00:00:00Z"
        )

        assert job.job_id == "test-123"
        assert job.status == "running"
        assert job.created_at == "2023-01-01T00:00:00Z"
        assert job.circuit_id is None
        assert job.backend is None
        assert job.shots is None
        assert job.results is None
        assert job.error is None

    def test_job_with_results(self):
        """Test job with results"""
        job = QuantumJob(
            job_id="test-123",
            status="completed",
            created_at="2023-01-01T00:00:00Z",
            results={"counts": {"00": 512, "11": 512}}
        )

        assert job.results == {"counts": {"00": 512, "11": 512}}


class TestSuperQuantXClient:
    """Test SuperQuantX client"""

    def test_client_creation_with_string(self):
        """Test client creation with API key string"""
        client = SuperQuantXClient("test-api-key")

        assert client.config.api_key == "test-api-key"
        assert client.config.base_url == "https://api.superquantx.ai"

    def test_client_creation_with_dict(self):
        """Test client creation with config dict"""
        config_dict = {
            "api_key": "test-key",
            "base_url": "https://custom.api.com",
            "timeout": 60
        }

        client = SuperQuantXClient(config_dict)

        assert client.config.api_key == "test-key"
        assert client.config.base_url == "https://custom.api.com"
        assert client.config.timeout == 60

    def test_client_creation_with_config_object(self):
        """Test client creation with config object"""
        config = SuperQuantXConfig(api_key="test-key")
        client = SuperQuantXClient(config)

        assert client.config.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check endpoint"""
        config = SuperQuantXConfig(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = {"status": "healthy"}
            mock_response.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response

            client = SuperQuantXClient(config)
            result = await client.health_check()

            assert result == {"status": "healthy"}
            mock_client.request.assert_called_once_with(
                method="GET",
                url="/health",
                json=None,
                params=None
            )

    @pytest.mark.asyncio
    async def test_request_with_retry(self):
        """Test request retry mechanism"""
        config = SuperQuantXConfig(api_key="test-key", max_retries=2)

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # First call fails, second succeeds
            mock_client.request.side_effect = [
                httpx.HTTPError("Network error"),
                Mock(json=lambda: {"success": True}, raise_for_status=lambda: None)
            ]

            with patch('asyncio.sleep') as mock_sleep:
                client = SuperQuantXClient(config)
                result = await client._request("GET", "/test")

                assert result == {"success": True}
                assert mock_client.request.call_count == 2
                mock_sleep.assert_called_once_with(2)  # Exponential backoff

    @pytest.mark.asyncio
    async def test_submit_job(self):
        """Test job submission"""
        config = SuperQuantXConfig(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = {
                "job_id": "test-job-123",
                "status": "submitted",
                "created_at": "2023-01-01T00:00:00Z"
            }
            mock_response.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response

            client = SuperQuantXClient(config)
            circuit_data = {"gates": [{"name": "H", "qubits": [0]}]}

            job = await client.submit_job(circuit_data, backend="simulator", shots=1024)

            assert isinstance(job, QuantumJob)
            assert job.job_id == "test-job-123"
            assert job.status == "submitted"

            # Verify request was made correctly
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "/jobs"
            assert call_args[1]["json"]["circuit"] == circuit_data
            assert call_args[1]["json"]["backend"] == "simulator"
            assert call_args[1]["json"]["shots"] == 1024

    def test_sync_wrappers(self):
        """Test synchronous wrapper methods"""
        config = SuperQuantXConfig(api_key="test-key")

        with patch('asyncio.run') as mock_run:
            mock_run.return_value = {"status": "healthy"}

            client = SuperQuantXClient(config)
            result = client.health_check_sync()

            assert result == {"status": "healthy"}
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_job_completion(self):
        """Test waiting for job completion"""
        config = SuperQuantXConfig(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock responses: running, running, completed
            mock_responses = [
                Mock(json=lambda: {"job_id": "test-123", "status": "running", "created_at": "2023-01-01T00:00:00Z"}),
                Mock(json=lambda: {"job_id": "test-123", "status": "running", "created_at": "2023-01-01T00:00:00Z"}),
                Mock(json=lambda: {"job_id": "test-123", "status": "completed", "created_at": "2023-01-01T00:00:00Z", "results": {"counts": {"0": 512, "1": 512}}})
            ]

            for response in mock_responses:
                response.raise_for_status.return_value = None

            mock_client.request.side_effect = mock_responses

            with patch('asyncio.sleep') as mock_sleep:
                client = SuperQuantXClient(config)
                job = await client.wait_for_job("test-123", timeout=300, poll_interval=1)

                assert job.status == "completed"
                assert job.results == {"counts": {"0": 512, "1": 512}}
                assert mock_client.request.call_count == 3
                assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_wait_for_job_timeout(self):
        """Test job wait timeout"""
        config = SuperQuantXConfig(api_key="test-key")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Always return running status
            mock_response = Mock()
            mock_response.json.return_value = {"job_id": "test-123", "status": "running", "created_at": "2023-01-01T00:00:00Z"}
            mock_response.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response

            with patch('asyncio.sleep'):
                with patch('asyncio.get_event_loop') as mock_loop:
                    # Mock time progression
                    mock_loop.return_value.time.side_effect = [0, 0.5, 1.0, 1.5, 2.0]  # Exceeds 1 second timeout

                    client = SuperQuantXClient(config)

                    with pytest.raises(TimeoutError):
                        await client.wait_for_job("test-123", timeout=1, poll_interval=0.1)
