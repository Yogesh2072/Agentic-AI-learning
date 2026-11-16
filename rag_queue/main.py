from dotenv import load_dotenv
from .server import app
import uvicorn

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)