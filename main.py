import argparse
import subprocess
import sys


def run_backend(host: str, port: int) -> int:
    cmd = [sys.executable, "-m", "uvicorn", "api.app:app", "--host", host, "--port", str(port), "--reload"]
    return subprocess.call(cmd)


def run_frontend(frontend_port: int) -> int:
    cmd = ["npm", "run", "dev", "--", "--port", str(frontend_port)]
    return subprocess.call(cmd, cwd="frontend")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backend", "frontend"], default="backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=3000)
    args = parser.parse_args()

    if args.mode == "backend":
        raise SystemExit(run_backend(args.host, args.port))
    raise SystemExit(run_frontend(args.frontend_port))


if __name__ == "__main__":
    main()

