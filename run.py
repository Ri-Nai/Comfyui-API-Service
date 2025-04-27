import uvicorn
from app import create_app
import sys

app = create_app()

if __name__ == "__main__":
    # --port=8000
    port = 8000
    args = sys.argv
    if len(args) > 1:
        port = int(args[1])
    uvicorn.run("run:app", host="0.0.0.0", port=port, reload=True)

