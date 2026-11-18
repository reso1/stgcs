from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib import resources
from urllib.parse import urlencode


class ViewerStaticAssets:
    HTML_ROUTE = "/viewer.html"
    HTML_FILENAME = "viewer.html"
    PACKAGE_NAME = "visualization.viewer"

    @classmethod
    def html_bytes(cls) -> bytes:
        return resources.files(cls.PACKAGE_NAME).joinpath(cls.HTML_FILENAME).read_bytes()

    @classmethod
    def viewer_url(cls, manifest_url: str = "/api/manifest") -> str:
        return (
            f"{cls.HTML_ROUTE}?"
            f"{urlencode({'manifest': manifest_url})}"
        )

    @classmethod
    def is_viewer_path(cls, path: str) -> bool:
        return path == cls.HTML_ROUTE

    @classmethod
    def write_html_response(cls, handler: BaseHTTPRequestHandler) -> None:
        html_bytes = cls.html_bytes()
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(html_bytes)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(html_bytes)
