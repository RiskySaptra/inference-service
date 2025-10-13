# YOLOv8 Inference API

This project provides a simple and scalable API for performing object detection using a custom YOLOv8 model. The API is built with FastAPI and can be easily deployed as a Docker container.

## Project Structure

```
.
├── Dockerfile
├── main.py
├── models/
│   └── your_model.pt  <-- Place your custom model here
└── requirements.txt
```

## Getting Started

### Prerequisites

- Docker
- Python 3.9+

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Place your model:**
    - Put your custom YOLOv8 model file (e.g., `your_model.pt`) into the `models/` directory.
    - Update the model path in `main.py` if your filename is different.

3.  **Run with Docker Compose:**
    ```bash
    docker compose up --build
    ```
    This command will build the Docker image and start the API service. To run it in the background, add the `-d` flag.

## API Usage

### Predict Endpoint

- **URL:** `/predict/`
- **Method:** `POST`
- **Body:** `multipart/form-data`
  - `file`: The image file to be processed.
- **Query Parameters:**
  - `confidence`: (optional) The confidence threshold for predictions (default: 0.5).
  - `overlap`: (optional) The overlap (IoU) threshold for non-maximum suppression (default: 0.5).

#### Example Request (using cURL)

```bash
curl -X POST "http://localhost:80/predict/?confidence=0.7&overlap=0.6" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

#### Example Response

```json
{
  "predictions": [
    {
      "x": 229,
      "y": 297,
      "width": 180,
      "height": 228,
      "confidence": 0.95,
      "class": "0",
      "class_id": 1,
      "detection_id": "03d12bb9-7809-4493-bddc-ab0b3fb13666"
    }
  ]
}
```

## Automated Retraining API

This project includes an API for automated retraining of the YOLOv8 model.

### Upload and Retrain Endpoint

- **URL:** `/retrain/upload/`
- **Method:** `POST`
- **Body:** `multipart/form-data`
  - `file`: A ZIP file containing the new dataset in YOLOv8 format.

This endpoint kicks off the retraining pipeline in the background and returns a `task_id`.

#### Example Request (using cURL)

```bash
curl -X POST "http://localhost:80/retrain/upload/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/dataset.zip"
```

### Check Retraining Status Endpoint

- **URL:** `/retrain/status/{task_id}`
- **Method:** `GET`

This endpoint allows you to check the status of a retraining job.

#### Example Request (using cURL)

```bash
curl -X GET "http://localhost:80/retrain/status/your-task-id" -H "accept: application/json"
```

## Local Development (Without Docker)

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.