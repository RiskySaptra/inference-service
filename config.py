from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/best.pt"
    DEVICE: str = "cpu"
    INFERENCE_IMAGES_PATH: str = "inference_images"
    API_KEY: str = "your-secret-api-key"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
