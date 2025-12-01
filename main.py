import uvicorn
from src.api.app import create_app
from src.config import get_settings


def main():
    settings = get_settings()
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
