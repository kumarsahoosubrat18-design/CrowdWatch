# CrowdWatch

CrowdWatch includes a Vite frontend and a FastAPI/YOLO backend. Start them together so the dashboard can connect to live backend telemetry.

## Run locally

1. Install Node.js and Python.
2. Install the backend requirements into the local virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   ```

3. Install frontend dependencies and launch the app:

   ```powershell
   npm install
   npm run dev
   ```

The launcher waits for the backend health endpoint before opening the frontend. Press `Ctrl+C` in the terminal to stop both services. If you use a different Python interpreter, set the `PYTHON` environment variable before running `npm run dev`.

The frontend opens at `http://127.0.0.1:5173`; the backend API is available at `http://127.0.0.1:8000`.

Configure camera sources by setting `CROWDWATCH_FEED_1_SOURCE` through `CROWDWATCH_FEED_6_SOURCE` before starting the app, or update a feed from the dashboard settings. Keep any URLs containing access tokens private and out of Git. Without configured camera sources, the backend remains available and the dashboard uses its fallback telemetry.
