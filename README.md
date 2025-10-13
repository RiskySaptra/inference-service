# YOLOv8 Inference and Retraining API

This project provides a robust and scalable API for performing object detection using a custom YOLOv8 model. The API is built with FastAPI and includes features such as automated retraining, persistent job storage, and API key security.

## Project Structure

```
.
├── Dockerfile
├── main.py
├── config.py
├── database.py
├── security.py
├── retraining_worker.py
├── routers/
│   ├── predict.py
│   ├── retrain.py
│   └── models.py
├── tests/
│   ├── test_predict.py
│   └── test_retrain.py
├── models/
│   └── your_model.pt
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

2.  **Configure the application:**
    - Create a `.env` file in the root directory and add the following environment variables:
      ```
      MODEL_PATH=models/your_model.pt
      API_KEY=your-secret-api-key
      ```
    - Place your custom YOLOv8 model file (e.g., `your_model.pt`) into the `models/` directory.

3.  **Run with Docker Compose:**
    ```bash
    docker compose up --build
    ```
    This command will build the Docker image and start the API service. To run it in the background, add the `-d` flag.

## API Usage

All endpoints are protected by an API key. You must include the `X-API-Key` header in your requests.

### Predict Endpoint

- **URL:** `/predict/`
- **Method:** `POST`
- **Headers:**
  - `X-API-Key`: Your secret API key.
- **Body:** `multipart/form-data`
  - `file`: The image file to be processed.
- **Query Parameters:**
  - `confidence`: (optional) The confidence threshold for predictions (default: 0.5).
  - `overlap`: (optional) The overlap (IoU) threshold for non-maximum suppression (default: 0.5).

#### Example Request (using cURL)

```bash
curl -X POST "http://localhost:80/predict/?confidence=0.7&overlap=0.6" -H "accept: application/json" -H "X-API-Key: your-secret-api-key" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

### Automated Retraining API

#### Upload and Retrain Endpoint

- **URL:** `/retrain/upload/`
- **Method:** `POST`
- **Headers:**
  - `X-API-Key`: Your secret API key.
- **Body:** `multipart/form-data`
  - `file`: A ZIP file containing the new dataset in YOLOv8 format.

#### Example Request (using cURL)

```bash
curl -X POST "http://localhost:80/retrain/upload/" -H "accept: application/json" -H "X-API-Key: your-secret-api-key" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/dataset.zip"
```

#### Check Retraining Status Endpoint

- **URL:** `/retrain/status/{task_id}`
- **Method:** `GET`
- **Headers:**
  - `X-API-Key`: Your secret API key.

#### Example Request (using cURL)

```bash
curl -X GET "http://localhost:80/retrain/status/your-task-id" -H "accept: application/json" -H "X-API-Key: your-secret-api-key"
```

##### Example Response

```json
{
"task_id": "your-task-id",
"status": "training model",
"progress": "45.20%"
}
```

### Model Management API

#### List Models Endpoint

- **URL:** `/models/`
- **Method:** `GET`
- **Headers:**
- `X-API-Key`: Your secret API key.

##### Example Request (using cURL)

```bash
curl -X GET "http://localhost:80/models/" -H "accept: application/json" -H "X-API-Key: your-secret-api-key"
```

##### Example Response

```json
{
"models": [
"task-id-1",
"task-id-2"
]
}
```

#### Set Active Model Endpoint

- **URL:** `/models/set_active/`
- **Method:** `POST`
- **Headers:**
- `X-API-Key`: Your secret API key.
- **Body:** `application/json`
- `task_id`: The ID of the model to set as active.

##### Example Request (using cURL)

```bash
curl -X POST "http://localhost:80/models/set_active/" -H "accept: application/json" -H "X-API-Key: your-secret-api-key" -H "Content-Type: application/json" -d '{"task_id": "your-task-id"}'
```

##### Example Response

```json
{
"message": "Model your-task-id set as active and reloaded"
}
```

## Testing

To run the tests, use the following command:

```bash
pytest
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