"""
LangChain Tool for Puffy — Local-First Media Extraction Engine.

Usage:
    from langchain_puffy import PuffyExtractTool

    tool = PuffyExtractTool()
    result = tool.invoke({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})

Requires Puffy running locally:
    puffy serve
"""

from __future__ import annotations

from typing import Any, Optional, Type

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


HEALTH_URL = "http://127.0.0.1:41480/api/health"
EXTRACT_URL = "http://127.0.0.1:41480/api/extract"
TIMEOUT_HEALTH = 3
TIMEOUT_EXTRACT = 300


class PuffyExtractInput(BaseModel):
    """Input schema for PuffyExtractTool."""

    url: str = Field(description="Media URL to extract (YouTube, TikTok, Bilibili, X, etc.)")
    save_dir: Optional[str] = Field(default=None, description="Optional directory to save extracted assets")


class PuffyExtractTool(BaseTool):
    """Extract video, audio, and transcripts from media URLs via the local Puffy daemon.

    Puffy must be running locally on 127.0.0.1:41480.
    Install from: https://github.com/susu-pro/puffy

    Start in headless mode:
        puffy serve
    """

    name: str = "puffy_extract"
    description: str = (
        "Download and extract media assets (video, audio, transcript, subtitles) "
        "from a URL using the local Puffy engine. Supports YouTube, TikTok (Douyin), "
        "Bilibili, X (Twitter), Kuaishou, and more."
    )
    args_schema: Type[BaseModel] = PuffyExtractInput

    # Configurable endpoint
    endpoint: str = "http://127.0.0.1:41480"

    def _check_health(self) -> bool:
        try:
            resp = requests.get(f"{self.endpoint}/api/health", timeout=TIMEOUT_HEALTH)
            return resp.status_code == 200
        except Exception:
            return False

    def _run(
        self,
        url: str,
        save_dir: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        if not self._check_health():
            return {
                "error": (
                    "Puffy daemon is not reachable at "
                    f"{self.endpoint}. "
                    "Please install and start Puffy: "
                    "https://github.com/susu-pro/puffy"
                )
            }

        payload: dict[str, Any] = {"url": url}
        if save_dir:
            payload["saveDir"] = save_dir

        try:
            resp = requests.post(
                f"{self.endpoint}/api/extract",
                json=payload,
                timeout=TIMEOUT_EXTRACT,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            return {"error": "Extraction timed out after 5 minutes."}
        except Exception as exc:
            return {"error": f"Extraction request failed: {exc}"}
