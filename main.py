import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = PROJECT_ROOT / "models"


def _call(cmd: list[str], cwd: Path) -> int:
    return subprocess.call(cmd, cwd=str(cwd))


def ensure_model_artifacts() -> None:
    required = [
        MODELS_DIR / "retrieval.pkl",
        MODELS_DIR / "ranker.pkl",
        MODELS_DIR / "user_embeddings.pkl",
    ]
    if all(path.exists() for path in required):
        return

    print("Model artifacts missing. Running build_index.py and train_ranker.py ...")
    if _call([sys.executable, "build_index.py"], PROJECT_ROOT) != 0:
        raise RuntimeError("build_index.py failed")
    if _call([sys.executable, "train_ranker.py"], PROJECT_ROOT) != 0:
        raise RuntimeError("train_ranker.py failed")


def ensure_frontend_dependencies() -> None:
    next_bin = FRONTEND_DIR / "node_modules" / ".bin" / ("next.cmd" if os.name == "nt" else "next")
    if next_bin.exists():
        return

    print("Frontend dependencies missing. Running npm install in ./frontend ...")
    install_cmd = ["cmd", "/c", "npm", "install"] if os.name == "nt" else ["npm", "install"]
    if _call(install_cmd, FRONTEND_DIR) != 0:
        raise RuntimeError("npm install failed")


def run_backend(host: str, port: int) -> int:
    cmd = [sys.executable, "-m", "uvicorn", "api.app:app", "--host", host, "--port", str(port), "--reload"]
    return _call(cmd, PROJECT_ROOT)


def run_frontend(frontend_port: int) -> int:
    ensure_frontend_dependencies()
    cmd = (
        ["cmd", "/c", "npm", "run", "dev", "--", "--port", str(frontend_port)]
        if os.name == "nt"
        else ["npm", "run", "dev", "--", "--port", str(frontend_port)]
    )
    return _call(cmd, FRONTEND_DIR)


def run_fullstack(host: str, port: int, frontend_port: int) -> int:
    ensure_model_artifacts()
    ensure_frontend_dependencies()

    backend_cmd = [sys.executable, "-m", "uvicorn", "api.app:app", "--host", host, "--port", str(port), "--reload"]
    frontend_cmd = (
        ["cmd", "/c", "npm", "run", "dev", "--", "--port", str(frontend_port)]
        if os.name == "nt"
        else ["npm", "run", "dev", "--", "--port", str(frontend_port)]
    )

    backend_proc = subprocess.Popen(backend_cmd, cwd=str(PROJECT_ROOT))
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR))

    print("Starting full stack...")
    print(f"Backend:  http://{host}:{port}")
    print(f"Frontend: http://127.0.0.1:{frontend_port}")
    print("Press Ctrl+C to stop both processes.")

    try:
        while True:
            if backend_proc.poll() is not None:
                if frontend_proc.poll() is None:
                    frontend_proc.terminate()
                return backend_proc.returncode or 1
            if frontend_proc.poll() is not None:
                if backend_proc.poll() is None:
                    backend_proc.terminate()
                return frontend_proc.returncode or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (backend_proc, frontend_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (backend_proc, frontend_proc):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backend", "frontend", "fullstack"], default="fullstack")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=3000)
    args = parser.parse_args()

    if args.mode == "backend":
        raise SystemExit(run_backend(args.host, args.port))
    if args.mode == "frontend":
        raise SystemExit(run_frontend(args.frontend_port))
    raise SystemExit(run_fullstack(args.host, args.port, args.frontend_port))


if __name__ == "__main__":
    main()
