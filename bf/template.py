"""Template: the blank master page plus the answer zones on it.

A template is a JSON file next to the master image:
{
  "name": "...",
  "master_image": "master.png",          # path relative to the JSON file
  "items": [
    {"id": "q01", "label": "I feel calm", "type": "likert",
     "options": [{"value": 1, "box": [x, y, w, h]}, ...]},
    ...
  ]
}
Boxes are in master-image pixel coordinates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Option:
    value: object
    box: tuple[int, int, int, int]


@dataclass
class Item:
    id: str
    label: str
    type: str
    options: list[Option] = field(default_factory=list)


@dataclass
class Template:
    name: str
    master_path: Path
    items: list[Item]
    _master: np.ndarray | None = None

    @property
    def master(self) -> np.ndarray:
        if self._master is None:
            img = cv2.imread(str(self.master_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(self.master_path)
            self._master = img
        return self._master

    @classmethod
    def load(cls, path: str | Path) -> "Template":
        path = Path(path)
        data = json.loads(path.read_text())
        items = [
            Item(
                id=it["id"],
                label=it.get("label", it["id"]),
                type=it.get("type", "likert"),
                options=[Option(o["value"], tuple(o["box"])) for o in it["options"]],
            )
            for it in data["items"]
        ]
        return cls(data["name"], path.parent / data["master_image"], items)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "name": self.name,
            "master_image": str(Path(self.master_path).name),
            "items": [
                {
                    "id": it.id,
                    "label": it.label,
                    "type": it.type,
                    "options": [{"value": o.value, "box": list(o.box)} for o in it.options],
                }
                for it in self.items
            ],
        }
        path.write_text(json.dumps(data, indent=2))
