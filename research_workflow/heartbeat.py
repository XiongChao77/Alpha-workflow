"""Periodic, read-only progress counters for the long-running worker."""

import logging
from threading import Event, Thread

from .library import eligible_counts


log = logging.getLogger(__name__)


def start_heartbeat(store, interval_seconds=10):
    stop = Event()

    def worker():
        while not stop.is_set():
            try:
                counts = eligible_counts(store)
                log.info("因子统计：符合条件总数=%s 已提交因子数=%s 今日提交因子数=%s",
                         counts["eligible_total"], counts["submitted_total"], counts["submitted_today"])
            except Exception:
                log.exception("因子统计读取失败")
            stop.wait(interval_seconds)

    thread = Thread(target=worker, name="alpha-statistics", daemon=True)
    thread.start()
    return stop, thread
