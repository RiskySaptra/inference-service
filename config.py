from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings.
    """
    MODEL_PATH: str = "models/v1.pt"
    DEVICE: str = "cpu"
    INFERENCE_IMAGES_PATH: str = "inference_images"
    DATASETS_PATH: str = "datasets"
    API_KEY: str = "your-secret-api-key"

    class Config:
        """
        Pydantic settings configuration.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()