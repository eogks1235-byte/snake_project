"""맵 프리셋 — 각 함수는 (width, height, rng)을 받아 벽 좌표 리스트를 반환.

매 라운드 무작위 선택. 벽이 있는 칸은 점령/이동/아이템 스폰 불가.
"""
import random


def map_open(w: int, h: int, rng: random.Random):
    """벽 없는 오픈 필드."""
    return []


def map_cross(w: int, h: int, rng: random.Random):
    """가운데 +자 벽으로 4분면 분할 (중앙 통로 1, 각 변 통로 X)."""
    walls = []
    cx, cy = w // 2, h // 2
    gap = 4
    for x in range(w):
        if abs(x - cx) > gap:
            walls.append((x, cy))
    for y in range(h):
        if abs(y - cy) > gap:
            walls.append((cx, y))
    return walls


def map_pillars(w: int, h: int, rng: random.Random):
    """규칙적 2x2 기둥 격자."""
    walls = []
    spacing = 10
    pad = spacing // 2
    for y in range(pad, h - pad - 1, spacing):
        for x in range(pad, w - pad - 1, spacing):
            for dy in range(2):
                for dx in range(2):
                    walls.append((x + dx, y + dy))
    return walls


def map_arena(w: int, h: int, rng: random.Random):
    """가운데 사각 링 — 각 변 중앙에 통로."""
    walls = []
    margin = 12
    gap = 5
    cx, cy = w // 2, h // 2
    for x in range(margin, w - margin):
        if abs(x - cx) > gap:
            walls.append((x, margin))
            walls.append((x, h - margin - 1))
    for y in range(margin + 1, h - margin - 1):
        if abs(y - cy) > gap:
            walls.append((margin, y))
            walls.append((w - margin - 1, y))
    return walls


def map_diagonal(w: int, h: int, rng: random.Random):
    """X자 대각선 벽 (가운데 통로)."""
    walls = set()
    cx, cy = w // 2, h // 2
    gap = 4
    for x in range(w):
        for thick in (0, 1):
            y1 = x + thick
            y2 = (h - 1 - x) + thick
            if 0 <= y1 < h and not (abs(x - cx) <= gap and abs(y1 - cy) <= gap):
                walls.add((x, y1))
            if 0 <= y2 < h and not (abs(x - cx) <= gap and abs(y2 - cy) <= gap):
                walls.add((x, y2))
    return list(walls)


def map_stripes(w: int, h: int, rng: random.Random):
    """가로 또는 세로 줄무늬 벽 (방향 랜덤, 통로 위치 랜덤)."""
    walls = []
    horizontal = rng.random() < 0.5
    spacing = 8
    gap_len = 6
    if horizontal:
        for y in range(spacing, h - 2, spacing):
            gap_x = rng.randint(2, max(2, w - gap_len - 2))
            for x in range(w):
                if not (gap_x <= x < gap_x + gap_len):
                    walls.append((x, y))
    else:
        for x in range(spacing, w - 2, spacing):
            gap_y = rng.randint(2, max(2, h - gap_len - 2))
            for y in range(h):
                if not (gap_y <= y < gap_y + gap_len):
                    walls.append((x, y))
    return walls


def map_islands(w: int, h: int, rng: random.Random):
    """무작위 사각 덩어리 8~12개."""
    walls = set()
    n = rng.randint(8, 12)
    for _ in range(n):
        sw = rng.randint(2, 4)
        sh = rng.randint(2, 4)
        x0 = rng.randint(4, max(4, w - sw - 5))
        y0 = rng.randint(4, max(4, h - sh - 5))
        for dy in range(sh):
            for dx in range(sw):
                walls.add((x0 + dx, y0 + dy))
    return list(walls)


def map_donut(w: int, h: int, rng: random.Random):
    """가운데 원형 벽 띠 (중심부는 비어있음)."""
    walls = []
    cx, cy = w // 2, h // 2
    r = min(w, h) // 5
    inner = (r - 1) ** 2
    outer = (r + 1) ** 2
    for y in range(h):
        for x in range(w):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if inner <= d2 <= outer:
                walls.append((x, y))
    return walls


def map_corridors(w: int, h: int, rng: random.Random):
    """수평 2층 벽 — 격자 1/3, 2/3 지점에 통로 다수."""
    walls = []
    y1 = h // 3
    y2 = 2 * h // 3
    cx = w // 2
    center_gap = 4
    for x in range(w):
        if abs(x - cx) <= center_gap:
            continue
        if x % 9 < 3:  # 주기적 통로
            continue
        walls.append((x, y1))
        walls.append((x, y2))
    return walls


MAP_PRESETS = [
    ('open', map_open),
    ('cross', map_cross),
    ('pillars', map_pillars),
    ('arena', map_arena),
    ('diagonal', map_diagonal),
    ('stripes', map_stripes),
    ('islands', map_islands),
    ('donut', map_donut),
    ('corridors', map_corridors),
]

MAP_LOOKUP = dict(MAP_PRESETS)


def random_map(rng: random.Random):
    """(name, build_fn) 튜플 반환."""
    return rng.choice(MAP_PRESETS)


def resolve_map(name: str, rng: random.Random):
    """이름으로 맵 조회. 없으면 랜덤."""
    if name and name in MAP_LOOKUP:
        return name, MAP_LOOKUP[name]
    return random_map(rng)
