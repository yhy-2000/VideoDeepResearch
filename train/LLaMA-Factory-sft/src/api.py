
import os

import uvicorn

from llamafactory.api.app import create_app
from llamafactory.chat import ChatModel


def main():
    chat_model = ChatModel()
    app = create_app(chat_model)
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    print(f"Visit http://localhost:{api_port}/docs for API document.")
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    main()
