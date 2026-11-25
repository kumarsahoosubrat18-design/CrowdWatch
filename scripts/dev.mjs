import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const backendUrl = "http://127.0.0.1:8000";
const localPython = path.join(root, "venv", "Scripts", "python.exe");
const python = process.env.PYTHON || (
  existsSync(localPython) ? localPython : process.platform === "win32" ? "python" : "python3"
);
const children = [];
let stopping = false;

function stopAll(code) {
  if (stopping) return;
  stopping = true;
  process.exitCode = code;
  for (const child of children) {
    if (!child.killed) child.kill();
  }
}

async function backendIsReady() {
  try {
    const response = await fetch(`${backendUrl}/api/stats`, {
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitForBackend(backend) {
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    if (backend.spawnError) throw backend.spawnError;
    if (backend.exitCode !== null) {
      throw new Error(`Backend exited before becoming ready (code ${backend.exitCode}).`);
    }
    if (await backendIsReady()) return;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Backend did not become ready within 3 minutes.");
}

process.on("SIGINT", () => stopAll(130));
process.on("SIGTERM", () => stopAll(143));

try {
  if (await backendIsReady()) {
    console.log("Using the already-running CrowdWatch backend.");
  } else {
    console.log("Starting the CrowdWatch backend...");
    const backend = spawn(
      python,
      ["-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", "8000"],
      { cwd: root, stdio: "inherit" },
    );
    backend.spawnError = null;
    backend.once("error", (error) => {
      backend.spawnError = error;
    });
    children.push(backend);
    backend.once("exit", (code) => {
      if (!stopping) {
        console.error(`Backend stopped${code === null ? "" : ` with code ${code}`}.`);
        stopAll(code ?? 1);
      }
    });
    await waitForBackend(backend);
    console.log("CrowdWatch backend is ready at http://127.0.0.1:8000.");
  }

  const vite = spawn(
    process.execPath,
    [path.join(root, "node_modules", "vite", "bin", "vite.js"), "--host", "127.0.0.1", "--open", "/"],
    { cwd: root, stdio: "inherit" },
  );
  children.push(vite);
  vite.once("error", (error) => {
    console.error(`Could not start the frontend: ${error.message}`);
    stopAll(1);
  });
  vite.once("exit", (code) => {
    if (!stopping) stopAll(code ?? 1);
  });
} catch (error) {
  console.error(error.message);
  stopAll(1);
}
