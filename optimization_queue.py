from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class QueuedOptimization:
    queue_id: str
    queued_at: str
    ready_at: str
    elapsed_seconds: float
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        base["payload"] = self.payload
        return base


class OptimizationQueue:
    def __init__(self, queue_path: Path | str = "optimization_queue.json"):
        self.queue_path = Path(queue_path)

    def _load_existing(self) -> List[Dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        try:
            data = json.loads(self.queue_path.read_text())
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _persist(self, entries: List[Dict[str, Any]]) -> None:
        self.queue_path.write_text(json.dumps(entries, indent=2))

    def enqueue(self, queued_at: datetime, ready_at: datetime, elapsed_seconds: float, payload: Dict[str, Any]) -> Dict[str, Any]:
        queue_item = QueuedOptimization(
            queue_id=str(int(queued_at.timestamp())),
            queued_at=queued_at.isoformat() + "Z",
            ready_at=ready_at.isoformat() + "Z",
            elapsed_seconds=round(elapsed_seconds, 3),
            payload=payload,
        ).to_dict()

        existing = self._load_existing()
        existing.append(queue_item)
        self._persist(existing)
        return queue_item
