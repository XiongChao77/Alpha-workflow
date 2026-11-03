"""Narrow BRAIN adapter. A POST with an unknown outcome is never retried blindly."""

from base64 import b64encode
from urllib.parse import urljoin

from .config import Config
from .transport import JsonTransport, RemoteError, retry_seconds


class Brain:
    def __init__(self, config: Config, transport: JsonTransport | None = None):
        self.config = config
        self.http = transport or JsonTransport()
        self.authenticated = False

    def _url(self, path: str) -> str:
        return urljoin(self.config.brain_url + "/", path.lstrip("/"))

    def authenticate(self):
        if self.authenticated:
            return
        if not self.config.email or not self.config.password:
            raise ValueError("WQ_BRAIN_EMAIL and WQ_BRAIN_PASSWORD are required")
        credentials = b64encode(f"{self.config.email}:{self.config.password}".encode()).decode()
        _, _, data = self.http.request("POST", self._url("authentication"), headers={"Authorization": "Basic " + credentials})
        if "inquiry" in data:
            raise RemoteError(401, "BRAIN requires interactive account verification")
        self.authenticated = True

    def _request(self, method: str, path: str, payload: dict | None = None):
        self.authenticate()
        try:
            return self.http.request(method, self._url(path), payload)
        except RemoteError as exc:
            if exc.status == 401:
                self.authenticated = False
            raise

    def start_simulation(self, expression: str, settings: dict) -> tuple[str, int]:
        _, headers, _ = self._request("POST", "simulations", {"type": "REGULAR", "settings": settings, "regular": expression})
        location = headers.get("Location")
        if not location:
            raise RemoteError(0, "Simulation accepted without a Location header; outcome requires reconciliation")
        return urljoin(self.config.brain_url + "/", location), retry_seconds(headers.get("Retry-After"), 10)

    def poll_simulation(self, location: str) -> tuple[dict, int]:
        self.authenticate()
        if not location.startswith(self.config.brain_url + "/"):
            raise ValueError("Unexpected simulation URL")
        _, headers, data = self.http.request("GET", location)
        status = str(data.get("status", "")).upper()
        if status in {"DONE", "COMPLETE"}:
            alpha = str(data.get("alpha") or "").rstrip("/").split("/")[-1]
            if alpha and not data.get("is"):
                detail = self.get_alpha(alpha)
                data["is"] = detail.get("is", {})
                data["oos"] = detail.get("oos", {})
            data["alpha_id"] = alpha or None
        return data, retry_seconds(headers.get("Retry-After"), 10)

    def get_alpha(self, alpha_id: str) -> dict:
        return self._request("GET", f"alphas/{alpha_id}")[2]

    def submit(self, alpha_id: str) -> dict:
        status, _, data = self._request("POST", f"alphas/{alpha_id}/submit")
        return {"http_status": status, "body": data}


def is_checks_passed(is_data: dict) -> bool:
    checks = is_data.get("checks") or []
    return bool(checks) and all(str(item.get("result", "")).upper() == "PASS" for item in checks)
