# YOLOv8 Inference API

FastAPI service for object detection using YOLOv8, with Docker support and API key authentication.

## Project Structure

```
.
├── main.py              # FastAPI app, startup, exception handlers
├── config.py            # Pydantic settings (env-based)
├── security.py          # API key validation
├── routers/
│   └── predict.py       # /predict/ endpoint
├── models/
│   └── best.pt          # YOLOv8 model weights
├── inference_images/    # Saved inference images
├── tests/
│   └── test_predict.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Setup

1. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` — set `API_KEY` and adjust `MODEL_PATH` / `DEVICE` as needed.

2. **Place your model** in `models/` (default: `models/best.pt`).

3. **Run with Docker:**
   ```bash
   docker compose up --build
   ```
   Or in the background:
   ```bash
   docker compose up --build -d
   ```

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API available at `http://127.0.0.1:8000`.

## API Reference

All endpoints require the `X-API-Key` header.

### Health Check

```
GET /
```

### Predict

```
POST /predict/
```

**Headers:** `X-API-Key: <your-key>`

**Body:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | Image to process |

**Query Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `confidence` | float | 0.25 | 0.0–1.0 | Confidence threshold |
| `overlap` | float | 0.45 | 0.0–1.0 | IoU threshold (NMS) |
| `imgsz` | int | 640 | 320–1280 | Inference image size |
| `max_det` | int | 50 | 1–200 | Max detections per image |
| `augment` | bool | False | — | Test-time augmentation |
| `agnostic_nms` | bool | False | — | Class-agnostic NMS |

**Example:**

```bash
curl -X POST "http://localhost:8000/predict/?confidence=0.3&augment=true" \
  -H "X-API-Key: your-secret-api-key" \
  -F "file=@image.jpg"
```

**Response:**

```json
{
  "predictions": [
    {
      "x": 320.5,
      "y": 240.2,
      "width": 100.0,
      "height": 80.0,
      "confidence": 0.92,
      "class": "person",
      "class_id": 0,
      "detection_id": "uuid"
    }
  ],
  "count": 1
}
```

## Configuration

Set via `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best.pt` | Path to YOLOv8 weights |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `INFERENCE_IMAGES_PATH` | `inference_images` | Directory for saved images |
| `API_KEY` | `your-secret-api-key` | API authentication key |

## Testing

```bash
pytest
```
