"""
Language Recovery OS - Master Execution Entrypoint
Refuses to boot a broken configuration silently (missing API key).
"""

import os
import sys

import uvicorn
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()


def fail_fast_on_missing_config():
    if not os.getenv("GEMINI_API_KEY"):
        print("\nGEMINI_API_KEY is not set.")
        print("   Copy .env.example to .env and add your key from https://aistudio.google.com/\n")
        sys.exit(1)


def main():
    print("""
    ==================================================================
      LANGUAGE RECOVERY OS - EVIDENCE BEFORE CONFIDENCE
      Google Cloud "All Things Agentic" Hackathon Submission
    ==================================================================
    """)

    fail_fast_on_missing_config()

    web_port = int(os.getenv("PORT") or os.getenv("WEB_APP_PORT", "8000"))
    # 0.0.0.0 in production - Cloud Run cannot healthcheck 127.0.0.1 (a
    # listener bound only to loopback is unreachable from outside the
    # container). Defaults to 0.0.0.0 here too so local `python run.py`
    # matches what actually ships, instead of drifting from the Docker path.
    web_host = os.getenv("WEB_APP_HOST", "0.0.0.0")
    print(f"Launching Language Recovery OS on http://{web_host}:{web_port} ...\n")

    uvicorn.run("web_app.app:app", host=web_host, port=web_port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
