"""The local review page: one HTML file, one script, one stylesheet, served by the backend.

Serving them from the backend keeps the page on an origin the API already allows, so the session
cookie stays same-origin and the existing Origin and CSRF checks apply to it unchanged. The page
holds no secret: it is the same for everyone, and everything it shows comes from calls that need a
signed-in session. A strict Content-Security-Policy allows only these same-origin files, no inline
script, no framing, and nothing to be sent anywhere else.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter(tags=["review-page"], include_in_schema=False)

WEB = Path(__file__).resolve().parents[2] / "web" / "review"
HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
ASSETS = {
    "/voice-review": ("index.html", "text/html; charset=utf-8"),
    "/voice-review/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/voice-review/app.css": ("app.css", "text/css; charset=utf-8"),
}


def _serve(name: str, media_type: str):
    def handler() -> Response:
        return Response((WEB / name).read_bytes(), media_type=media_type, headers=HEADERS)

    return handler


for _path, (_name, _type) in ASSETS.items():
    router.add_api_route(_path, _serve(_name, _type), methods=["GET"])
