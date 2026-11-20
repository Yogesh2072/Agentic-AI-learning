from dotenv import load_dotenv
from .server import app
import uvicorn

load_dotenv()

# use export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES 
# and then run the below command to start the server
# python -m rag_queue.main to start the server on http://127.0.0.1:8000

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



