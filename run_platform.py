"""
Unified Platform Entrypoint and Service Launcher.
"""

import sys
import subprocess
import argparse

def launch_dashboard():
    cmd = [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.headless=true"]
    subprocess.run(cmd)

def launch_api():
    cmd = [sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Financial Fraud Intelligence Platform Launcher")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "api"],
        default="dashboard",
        help="Execution mode: dashboard (Streamlit UI) or api (FastAPI Microservice)"
    )
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        launch_dashboard()
    elif args.mode == "api":
        launch_api()

if __name__ == "__main__":
    main()
