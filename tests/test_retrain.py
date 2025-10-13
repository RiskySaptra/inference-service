import pytest
from httpx import AsyncClient
from main import app
from config import settings
import io
import zipfile

@pytest.mark.asyncio
async def test_upload_and_retrain_success():
    """
    Test successful upload and start of retraining.
    """
    # Create a dummy zip file
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("data.yaml", "train: ../train/images")
    zip_bytes.seek(0)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/retrain/upload/",
            files={"file": ("test.zip", zip_bytes, "application/zip")},
            headers={"X-API-Key": settings.API_KEY}
        )
    assert response.status_code == 200
    assert "task_id" in response.json()
    assert response.json()["status"] == "Retraining started"

@pytest.mark.asyncio
async def test_get_retraining_status_not_found():
    """
    Test getting the status of a non-existent retraining job.
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(
            "/retrain/status/non-existent-task-id",
            headers={"X-API-Key": settings.API_KEY}
        )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_retrain_unauthorized():
    """
    Test unauthorized access to the retrain endpoint.
    """
    # Create a dummy zip file
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w') as zf:
        zf.writestr("data.yaml", "train: ../train/images")
    zip_bytes.seek(0)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/retrain/upload/",
            files={"file": ("test.zip", zip_bytes, "application/zip")},
            headers={"X-API-Key": "wrong-key"}
        )
    assert response.status_code == 403