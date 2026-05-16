"""토러스 공간 해시 — neighbor lookup을 O(N) 평균으로."""
from typing import List, Iterable

from .entities import WORLD_W, WORLD_H


CELL_SIZE = 80


class SpatialHash:
    """매 tick마다 build()로 재구축. query()로 반경 내 entity iterator."""

    def __init__(self):
        self.cells: dict = {}
        self.cols = int(WORLD_W / CELL_SIZE) + 1
        self.rows = int(WORLD_H / CELL_SIZE) + 1

    def build(self, entities: Iterable):
        self.cells = {}
        for e in entities:
            if hasattr(e, 'alive') and not e.alive:
                continue
            cx = int(e.x / CELL_SIZE) % self.cols
            cy = int(e.y / CELL_SIZE) % self.rows
            self.cells.setdefault((cx, cy), []).append(e)

    def query(self, x: float, y: float, radius: float) -> List:
        """radius 내 entity 후보들 (false-positive 가능, 거리 검사는 호출자가)."""
        rng_cells = max(1, int(radius / CELL_SIZE) + 1)
        cx0 = int(x / CELL_SIZE)
        cy0 = int(y / CELL_SIZE)
        out = []
        for dx in range(-rng_cells, rng_cells + 1):
            for dy in range(-rng_cells, rng_cells + 1):
                key = ((cx0 + dx) % self.cols, (cy0 + dy) % self.rows)
                bucket = self.cells.get(key)
                if bucket:
                    out.extend(bucket)
        return out
