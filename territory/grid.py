"""격자 상태 관리"""
import numpy as np

EMPTY = 0


class Grid:
    def __init__(self, width: int = 60, height: int = 60):
        self.width = width
        self.height = height
        self.cells = np.zeros((height, width), dtype=np.int8)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def claim(self, x: int, y: int, agent_id: int) -> int:
        previous = int(self.cells[y, x])
        self.cells[y, x] = agent_id
        return previous

    def get_owner(self, x: int, y: int) -> int:
        return int(self.cells[y, x])

    def get_areas(self, num_agents: int) -> dict:
        return {i: int((self.cells == i).sum()) for i in range(1, num_agents + 1)}

    def total_cells(self) -> int:
        return self.width * self.height

    def damage_territory(self, agent_id: int, fraction: float, rng) -> int:
        """해당 에이전트의 영토 중 일부를 무작위로 빈 칸으로 되돌림.

        반환값: 잃은 칸 수
        """
        ys, xs = np.where(self.cells == agent_id)
        if len(xs) == 0:
            return 0
        n_loss = max(1, int(len(xs) * fraction))
        idx = rng.sample(range(len(xs)), min(n_loss, len(xs)))
        for i in idx:
            self.cells[ys[i], xs[i]] = EMPTY
        return len(idx)
