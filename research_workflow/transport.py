"""Small JSON HTTP transport with explicit 429/503 handling."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookiejar import CookieJar
import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener


class RemoteError(Exception):
    def __init__(self, status: int, message: str, retry_after: int | None = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(message)


def retry_seconds(raw: str | None, default: int = 60) -> int:
    if not raw:
        return default
    try:
        return max(1, int(float(raw)))
    except ValueError:
        try:
            return max(1, int((parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            return default


class JsonTransport:
    def __init__(self, timeout: int = 30):
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
        self.timeout = timeout

    def request(self, method: str, url: str, payload: dict | None = None, headers: dict | None = None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(url, data=body, headers={"Accept": "application/json", **({"Content-Type": "application/json"} if body else {}), **(headers or {})}, method=method)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read()
                return response.status, response.headers, json.loads(raw) if raw.strip() else {}
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")[:500]
            header = exc.headers.get("Retry-After") if exc.headers else None
            raise RemoteError(exc.code, raw, retry_seconds(header) if header else None) from exc
        except (URLError, TimeoutError) as exc:
            raise RemoteError(0, str(exc)) from exc
