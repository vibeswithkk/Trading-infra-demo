from __future__ import annotations
import json, logging
from datetime import datetime, timezone

class JsonLogger:
    def __init__(self, name: str = __name__):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            h = logging.StreamHandler()
            self._logger.addHandler(h)
        self._logger.setLevel(logging.INFO)

    def _fmt(self, level: str, msg: str, **kv):
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "msg": msg}
        payload.update({k: (str(v)) for k,v in kv.items()})
        return json.dumps(payload, separators=(",",":"))

    def info(self, msg: str, **kv):
        self._logger.info(self._fmt("info", msg, **kv))

    def warning(self, msg: str, **kv):
        self._logger.warning(self._fmt("warn", msg, **kv))

    def error(self, msg: str, **kv):
        self._logger.error(self._fmt("error", msg, **kv))