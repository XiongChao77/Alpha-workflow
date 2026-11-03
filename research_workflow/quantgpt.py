"""Optional QuantGPT HTTP adapter; QuantGPT is never modified or imported."""

from .config import Config
from .transport import JsonTransport


class QuantGPT:
    def __init__(self, config: Config, transport: JsonTransport | None = None):
        self.config = config
        self.http = transport or JsonTransport(timeout=30)

    def _headers(self) -> dict:
        return {"Authorization": "Bearer " + self.config.quantgpt_token} if self.config.quantgpt_token else {}

    def start(self, expression: str) -> str:
        _, _, data = self.http.request("POST", self.config.quantgpt_url + "/api/v1/auto_backtest",
                                       {"prompt": expression, "universe": self.config.quantgpt_universe}, self._headers())
        return data["task_id"]

    def poll(self, task_id: str) -> dict:
        return self.http.request("GET", self.config.quantgpt_url + "/api/v1/tasks/" + task_id,
                                 headers=self._headers())[2]
