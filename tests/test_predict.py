import pytest
from httpx import AsyncClient
from main import app
from config import settings
from PIL import Image
import io

@pytest.mark.asyncio
async def test_predict_success():
    """
    Test successful prediction.
    """
    # Create a dummy image
    image = Image.new('RGB', (100, 100), color = 'red')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/predict/",
            files={"file": ("test.png", image_bytes, "image/png")},
            headers={"X-API-Key": settings.API_KEY}
        )
    assert response.status_code == 200
    assert "predictions" in response.json()

@pytest.mark.asyncio
async def test_predict_unauthorized():
    """
    Test unauthorized access to the predict endpoint.
    """
    # Create a dummy image
    image = Image.new('RGB', (100, 100), color = 'red')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/predict/",
            files={"file": ("test.png", image_bytes, "image/png")},
            headers={"X-API-Key": "wrong-key"}
        )
    assert response.status_code == 403

@pytest.mark.asyncio
async def test_predict_no_model(monkeypatch):
    """
    Test the predict endpoint when the model is not loaded.
    """
    # Mock the model to be None
    monkeypatch.setattr("routers.predict.model", None)

    # Create a dummy image
    image = Image.new('RGB', (100, 100), color = 'red')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/predict/",
            files={"file": ("test.png", image_bytes, "image/png")},
            headers={"X-API-Key": settings.API_KEY}
        )
    assert response.status_code == 503