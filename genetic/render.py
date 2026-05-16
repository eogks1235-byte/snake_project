"""유전 진화 시뮬 시각화."""
import pygame
from typing import List

from .world import World, WORLD_W, WORLD_H


BG_COLOR = (15, 17, 21)
PANEL_COLOR = (22, 25, 31)
WORLD_BG = (20, 26, 22)
GRID_LINE = (28, 36, 30)
FOOD_COLOR = (140, 200, 90)
TEXT_PRIMARY = (235, 235, 240)
TEXT_SECONDARY = (140, 145, 160)
ACCENT_GOLD = (255, 215, 90)


class Renderer:
    HEADER_H = 80
    SIDEBAR_W = 280
    MARGIN = 24

    def __init__(self):
        pygame.init()
        self.window_w = WORLD_W + self.SIDEBAR_W + self.MARGIN * 3
        self.window_h = WORLD_H + self.HEADER_H + self.MARGIN * 2
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('Evolution — Genetic Algorithm')
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont('arial,malgungothic', 28, bold=True)
        self.font_sub = pygame.font.SysFont('arial,malgungothic', 13)
        self.font_stat = pygame.font.SysFont('arial,malgungothic', 14, bold=True)
        self.font_pin = pygame.font.SysFont('arial,malgungothic', 12, bold=True)
        self.font_label = pygame.font.SysFont('arial,malgungothic', 11)

        self.world_x = self.MARGIN
        self.world_y = self.HEADER_H + self.MARGIN

        # 그래프 데이터 캐시
        self.graph_cache: List[dict] = []
        self.graph_max_points = 200

    # ── 외부 진입점 ─────────────────────────────────────

    def draw(self, world: World, fast_forward: bool):
        self.screen.fill(BG_COLOR)
        self._draw_header(world, fast_forward)
        self._draw_world(world)
        self._draw_sidebar(world)
        pygame.display.flip()

    # ── 헤더 ──────────────────────────────────────────

    def _draw_header(self, world: World, fast: bool):
        title = self.font_title.render('EVOLUTION', True, TEXT_PRIMARY)
        self.screen.blit(title, (self.MARGIN, 22))

        sub = (f'gen {world.max_generation()}  ·  pop {world.population()}  ·  '
               f'tick {world.tick}  ·  seed {world.seed}')
        sub_surf = self.font_sub.render(sub, True, TEXT_SECONDARY)
        self.screen.blit(sub_surf, (self.MARGIN, 56))

        if fast:
            label = self.font_pin.render('FAST FORWARD', True, BG_COLOR)
            pad_x, pad_y = 10, 6
            rect = pygame.Rect(
                self.window_w - self.MARGIN - label.get_width() - pad_x * 2, 32,
                label.get_width() + pad_x * 2, label.get_height() + pad_y * 2,
            )
            pygame.draw.rect(self.screen, ACCENT_GOLD, rect, border_radius=10)
            self.screen.blit(label, (rect.x + pad_x, rect.y + pad_y))

    # ── 월드 ──────────────────────────────────────────

    def _draw_world(self, world: World):
        pad = 6
        panel = pygame.Rect(self.world_x - pad, self.world_y - pad,
                            WORLD_W + pad * 2, WORLD_H + pad * 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=10)
        pygame.draw.rect(self.screen, WORLD_BG,
                         (self.world_x, self.world_y, WORLD_W, WORLD_H))

        # 음식
        for f in world.foods:
            pygame.draw.circle(self.screen, FOOD_COLOR,
                               (int(self.world_x + f.x), int(self.world_y + f.y)), 2)

        # 개체
        for c in world.creatures:
            cx = int(self.world_x + c.x)
            cy = int(self.world_y + c.y)
            r = max(2, int(c.genome.size_px()))
            color = c.color()
            # 시야 (반투명)
            if c.energy > 0:
                vis_r = int(c.genome.vision_px())
                vision_surf = pygame.Surface((vis_r * 2, vis_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(vision_surf, (color[0], color[1], color[2], 18),
                                   (vis_r, vis_r), vis_r)
                self.screen.blit(vision_surf, (cx - vis_r, cy - vis_r))

            # 본체
            pygame.draw.circle(self.screen, color, (cx, cy), r)
            # 에너지바 (개체 위)
            from .world import MAX_ENERGY
            bar_w = max(8, r * 2)
            ratio = max(0.0, min(1.0, c.energy / MAX_ENERGY))
            pygame.draw.rect(self.screen, (40, 44, 54),
                             (cx - bar_w // 2, cy - r - 6, bar_w, 2))
            pygame.draw.rect(self.screen, (220, 220, 220),
                             (cx - bar_w // 2, cy - r - 6, int(bar_w * ratio), 2))

    # ── 사이드바 ─────────────────────────────────────

    def _draw_sidebar(self, world: World):
        x = self.world_x + WORLD_W + self.MARGIN
        y = self.world_y - 6
        w = self.SIDEBAR_W
        h = WORLD_H + 12
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h), border_radius=10)

        inner_x = x + 16
        inner_w = w - 32
        cur_y = y + 16

        # 타이틀
        title = self.font_stat.render('POPULATION', True, TEXT_SECONDARY)
        self.screen.blit(title, (inner_x, cur_y))
        cur_y += 22

        big_pop = pygame.font.SysFont('arial,malgungothic', 38, bold=True)
        pop_surf = big_pop.render(str(world.population()), True, ACCENT_GOLD)
        self.screen.blit(pop_surf, (inner_x, cur_y))
        cur_y += 50

        # 통계 행
        if world.history:
            last = world.history[-1]
            self._draw_stat(inner_x, cur_y, inner_w, 'avg speed',
                            f'{last["avg_speed"]:.2f}'); cur_y += 22
            self._draw_stat(inner_x, cur_y, inner_w, 'avg vision',
                            f'{last["avg_vision"]:.0f}'); cur_y += 22
            self._draw_stat(inner_x, cur_y, inner_w, 'avg size',
                            f'{last["avg_size"]:.2f}'); cur_y += 22
            self._draw_stat(inner_x, cur_y, inner_w, 'avg metab',
                            f'{last.get("avg_metab", 0):.3f}'); cur_y += 22
            self._draw_stat(inner_x, cur_y, inner_w, 'max generation',
                            str(world.max_generation())); cur_y += 30

        self._draw_stat(inner_x, cur_y, inner_w, 'births', str(world.births))
        cur_y += 22
        self._draw_stat(inner_x, cur_y, inner_w, 'deaths', str(world.deaths))
        cur_y += 30

        # 그래프 — population over time
        graph_h = 80
        cur_y += 12
        title2 = self.font_stat.render('POPULATION', True, TEXT_SECONDARY)
        self.screen.blit(title2, (inner_x, cur_y))
        cur_y += 18
        self._draw_graph(inner_x, cur_y, inner_w, graph_h, 'count', world.history,
                         color=(160, 200, 240))
        cur_y += graph_h + 18

        # avg_speed/vision/size 트렌드 — 작은 미니 그래프 3개
        title3 = self.font_stat.render('TRAITS — speed / vision / size', True, TEXT_SECONDARY)
        self.screen.blit(title3, (inner_x, cur_y))
        cur_y += 18
        mini_h = 50
        self._draw_graph(inner_x, cur_y, inner_w, mini_h, 'avg_speed', world.history,
                         color=(120, 220, 140), normalize=(0.4, 3.5))
        cur_y += mini_h + 4
        self._draw_graph(inner_x, cur_y, inner_w, mini_h, 'avg_vision', world.history,
                         color=(220, 140, 140), normalize=(10, 90))
        cur_y += mini_h + 4
        self._draw_graph(inner_x, cur_y, inner_w, mini_h, 'avg_size', world.history,
                         color=(140, 160, 220), normalize=(3.0, 9.0))

    def _draw_stat(self, x: int, y: int, w: int, label: str, value: str):
        lab = self.font_label.render(label, True, TEXT_SECONDARY)
        val = self.font_stat.render(value, True, TEXT_PRIMARY)
        self.screen.blit(lab, (x, y + 2))
        self.screen.blit(val, (x + w - val.get_width(), y))

    def _draw_graph(self, x: int, y: int, w: int, h: int, key: str,
                    history: List[dict], color: tuple,
                    normalize: tuple = None):
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h), border_radius=4)
        if not history:
            return

        # 최근 graph_max_points만
        data = history[-self.graph_max_points:]
        values = [d.get(key, 0) for d in data]
        if not values:
            return

        if normalize:
            lo, hi = normalize
        else:
            lo = 0
            hi = max(values) * 1.1 + 1

        if hi <= lo:
            return

        # 폴리라인
        pts = []
        for i, v in enumerate(values):
            px = x + i * w / max(1, len(values) - 1)
            py = y + h - (v - lo) / (hi - lo) * h
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 2)

    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()
