"""Capture dashboard screenshots for the README (demo mode, no Alpaca keys).

    python -m src.scripts.capture_screenshots

Starts the dashboard in ``--demo`` mode on a free port, drives a headless
Chromium via Playwright, and writes three PNGs to ``docs/screenshots/``:

  * ``dashboard-desktop.png``    — 1440x900, normal view
  * ``dashboard-mobile.png``     — 390x844, mobile view
  * ``dashboard-kill-switch.png``— 1440x900, kill-switch-active view (``?kill=1``)

Playwright is a *dev-only* dependency (see ``requirements-dev.txt``) — it is not
needed to run the bot or the dashboard. The script is idempotent (it overwrites
existing PNGs) and always tears down the dashboard subprocess on exit.
"""
from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO_ROOT / "docs" / "screenshots"

_SHOTS = [
    {"name": "dashboard-desktop.png", "path": "/", "width": 1440, "height": 900},
    {"name": "dashboard-mobile.png", "path": "/", "width": 390, "height": 844},
    {"name": "dashboard-kill-switch.png", "path": "/?kill=1", "width": 1440, "height": 900},
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(base_url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # not up yet
            last_err = exc
            time.sleep(0.3)
    raise RuntimeError(f"dashboard did not come up at {base_url}: {last_err}")


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "ERROR: Playwright is not installed. It is a dev-only dependency:\n"
            "  .venv/bin/pip install -r requirements-dev.txt\n"
            "  .venv/bin/playwright install chromium",
            file=sys.stderr,
        )
        return 2

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.scripts.run_dashboard",
         "--demo", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup)

    try:
        _wait_until_up(base_url)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                for shot in _SHOTS:
                    page = browser.new_page(
                        viewport={"width": shot["width"], "height": shot["height"]},
                        device_scale_factor=2,
                    )
                    page.goto(base_url + shot["path"], wait_until="networkidle")
                    out = _OUT_DIR / shot["name"]
                    page.screenshot(path=str(out))
                    page.close()
                    print(f"[screenshots] wrote {out.relative_to(_REPO_ROOT)} "
                          f"({shot['width']}x{shot['height']})")
            finally:
                browser.close()
    finally:
        _cleanup()

    print(f"[screenshots] done -> {_OUT_DIR.relative_to(_REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
