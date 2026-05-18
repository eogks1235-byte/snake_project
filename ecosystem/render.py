"""3계층 생태계 시각화 — 식물/초식/육식 + 인구 그래프 (먹이사슬 진동)."""
import pygame
from typing import List

from .world import World
from .entities import (
    WORLD_W, WORLD_H,
    HERB_COLOR, HERB_COLOR_DARK, HERB_SIZE, HERB_VISION, HERB_MAX_E,
    CARN_COLOR, CARN_COLOR_DARK, CARN_SIZE, CARN_VISION, CARN_MAX_E,
)


BG_COLOR = (15, 17, 21)
PANEL_COLOR = (22, 25, 31)
WORLD_BG = (16, 22, 18)
TEXT_PRIMARY = (235, 235, 240)
TEXT_SECONDARY = (140, 145, 160)
ACCENT_GOLD = (255, 215, 90)
ACCENT_SILVER = (220, 225, 235)
PLANT_GLOW = (50, 80, 50)

# 낮/밤 배경 양 끝점 — sin 보간으로 부드럽게 흐른다
WORLD_BG_NIGHT = (10, 14, 22)
WORLD_BG_DAY = (24, 38, 26)

CARRION_COLOR = (70, 60, 50)        # 옅은 흙빛 — 점점 사라짐
FEAR_COLOR = (180, 70, 70)          # 옅은 붉은 잔향


class Renderer:
    HEADER_H = 80
    SIDEBAR_W = 300
    MARGIN = 24

    def __init__(self):
        pygame.init()
        self.window_w = WORLD_W + self.SIDEBAR_W + self.MARGIN * 3
        self.window_h = WORLD_H + self.HEADER_H + self.MARGIN * 2
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('Ecosystem — plant / herbivore / carnivore')
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont('arial,malgungothic', 28, bold=True)
        self.font_sub = pygame.font.SysFont('arial,malgungothic', 13)
        self.font_stat = pygame.font.SysFont('arial,malgungothic', 14, bold=True)
        self.font_pin = pygame.font.SysFont('arial,malgungothic', 12, bold=True)
        self.font_label = pygame.font.SysFont('arial,malgungothic', 11)
        self.font_big_count = pygame.font.SysFont('arial,malgungothic', 30, bold=True)

        self.world_x = self.MARGIN
        self.world_y = self.HEADER_H + self.MARGIN

    def draw(self, world: World, fast_forward: bool):
        self.screen.fill(BG_COLOR)
        self._draw_header(world, fast_forward)
        self._draw_world(world)
        self._draw_sidebar(world)
        pygame.display.flip()

    def _draw_header(self, w: World, fast: bool):
        title = self.font_title.render('ECOSYSTEM', True, TEXT_PRIMARY)
        self.screen.blit(title, (self.MARGIN, 22))
        pops = w.populations()
        sub = (f'plants {pops["plants"]}  ·  herb {pops["herb"]}  ·  '
               f'carn {pops["carn"]}  ·  tick {w.tick}  ·  seed {w.seed}')
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

    def _draw_world(self, world: World):
        pad = 6
        panel = pygame.Rect(self.world_x - pad, self.world_y - pad,
                            WORLD_W + pad * 2, WORLD_H + pad * 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=10)

        # 낮/밤 — 배경색이 천천히 흐른다 (영원회귀)
        light = world.day_light()
        bg = self._lerp_color(WORLD_BG_NIGHT, WORLD_BG_DAY, light)
        pygame.draw.rect(self.screen, bg,
                         (self.world_x, self.world_y, WORLD_W, WORLD_H))

        # 공포의 풍경 — 사냥의 흔적은 산 자들에게 더 오래 남는다
        self._draw_fear_marks(world)

        # 시체 → 흙 — 어두운 점이 옅어지며 흙으로 돌아간다
        self._draw_carrions(world)

        # 식물 (글로우 효과 — 다 자란 식물에 약간 밝게)
        for p in world.plants:
            cx = int(self.world_x + p.x)
            cy = int(self.world_y + p.y)
            r = p.radius()
            if p.growth >= 20:
                pygame.draw.circle(self.screen, PLANT_GLOW, (cx, cy), r + 2)
            pygame.draw.circle(self.screen, p.color(), (cx, cy), r)

        # 초식 (꼬리 포함)
        for h in world.herbivores:
            cx = int(self.world_x + h.x)
            cy = int(self.world_y + h.y)
            tail_x = cx - int(h.vx * 4)
            tail_y = cy - int(h.vy * 4)
            pygame.draw.line(self.screen, HERB_COLOR_DARK, (cx, cy), (tail_x, tail_y), 2)
            pygame.draw.circle(self.screen, HERB_COLOR, (cx, cy), HERB_SIZE)

        # 육식 (테두리 더 강조)
        for c in world.carnivores:
            cx = int(self.world_x + c.x)
            cy = int(self.world_y + c.y)
            tail_x = cx - int(c.vx * 5)
            tail_y = cy - int(c.vy * 5)
            pygame.draw.line(self.screen, CARN_COLOR_DARK, (cx, cy), (tail_x, tail_y), 3)
            pygame.draw.circle(self.screen, CARN_COLOR_DARK, (cx, cy), CARN_SIZE + 1)
            pygame.draw.circle(self.screen, CARN_COLOR, (cx, cy), CARN_SIZE)

        # 시조 & 장로 표식 — 메멘토 모리. 결국 그들도 죽는다
        self._draw_founder_elder(world)

        # 좌상단 작은 해/달 — 사이클이 어디쯤인지
        self._draw_sun_moon(world)

    # ── 헬퍼: 시각 효과 ─────────────────────────────────

    @staticmethod
    def _lerp_color(a, b, t: float):
        t = max(0.0, min(1.0, t))
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )

    def _draw_carrions(self, world: World):
        for c in world.carrions:
            life = 1.0 - c.age / c.lifespan
            if life <= 0:
                continue
            cx = int(self.world_x + c.x)
            cy = int(self.world_y + c.y)
            r = max(2, int(3 * life + 1))
            # 흙빛 어두운 점 — fade를 위해 배경과 보간
            shade = self._lerp_color(WORLD_BG_DAY, CARRION_COLOR, life)
            pygame.draw.circle(self.screen, shade, (cx, cy), r)

    def _draw_fear_marks(self, world: World):
        # 옅은 붉은 영역 — alpha blending 으로 잔향 표현
        surf = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
        for fm in world.fear_marks:
            life = 1.0 - fm.age / fm.lifespan
            if life <= 0:
                continue
            alpha = int(55 * life)
            r = int(fm.radius)
            pygame.draw.circle(surf, (FEAR_COLOR[0], FEAR_COLOR[1], FEAR_COLOR[2], alpha),
                               (int(fm.x), int(fm.y)), r)
        self.screen.blit(surf, (self.world_x, self.world_y))

    def _draw_founder_elder(self, world: World):
        from .entities import Carnivore as _C
        f = world.founder
        e = world.elder
        # 시조 == 장로인 경우 — 두 후광이 겹쳐 보임
        if f is not None and f.alive:
            cx = int(self.world_x + f.x)
            cy = int(self.world_y + f.y)
            size = CARN_SIZE if isinstance(f, _C) else HERB_SIZE
            pygame.draw.circle(self.screen, ACCENT_GOLD, (cx, cy), size + 5, 1)
            pygame.draw.circle(self.screen, ACCENT_GOLD, (cx, cy - size - 6), 2)
        if e is not None and e.alive and e is not f:
            cx = int(self.world_x + e.x)
            cy = int(self.world_y + e.y)
            size = CARN_SIZE if isinstance(e, _C) else HERB_SIZE
            pygame.draw.circle(self.screen, ACCENT_SILVER, (cx, cy), size + 4, 1)
        elif e is not None and e is f and e.alive:
            # 시조이자 장로 — 외곽에 은빛 링 하나 더
            cx = int(self.world_x + e.x)
            cy = int(self.world_y + e.y)
            size = CARN_SIZE if isinstance(e, _C) else HERB_SIZE
            pygame.draw.circle(self.screen, ACCENT_SILVER, (cx, cy), size + 9, 1)

    def _draw_sun_moon(self, world: World):
        light = world.day_light()
        phase = world.day_phase()
        x = self.world_x + 18
        y = self.world_y + 18
        r = 9
        # 해(낮) → 달(밤) 색조 보간
        sun = (255, 220, 130)
        moon = (200, 210, 230)
        color = self._lerp_color(moon, sun, light)
        pygame.draw.circle(self.screen, color, (x, y), r)
        # 밤이 가까울수록 우측에 그림자
        if light < 0.5:
            shadow_r = int(r * (1 - light * 2))
            shadow_color = WORLD_BG_NIGHT
            pygame.draw.circle(self.screen, shadow_color,
                               (x + r - shadow_r, y), shadow_r)
        # phase 텍스트
        if phase < 0.125 or phase >= 0.875:
            tag = 'dawn'
        elif phase < 0.375:
            tag = 'noon'
        elif phase < 0.625:
            tag = 'dusk'
        else:
            tag = 'night'
        label = self.font_label.render(tag, True, TEXT_SECONDARY)
        self.screen.blit(label, (x + r + 8, y - 6))

    def _draw_sidebar(self, world: World):
        x = self.world_x + WORLD_W + self.MARGIN
        y = self.world_y - 6
        w = self.SIDEBAR_W
        h = WORLD_H + 12
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h), border_radius=10)

        inner_x = x + 16
        inner_w = w - 32
        cur_y = y + 18

        # 인구 카운트 3개 (색칩)
        pops = world.populations()
        for label, count, color in [
            ('PLANTS',  pops['plants'], (140, 200, 90)),
            ('HERBS',   pops['herb'],   HERB_COLOR),
            ('CARNS',   pops['carn'],   CARN_COLOR),
        ]:
            pygame.draw.rect(self.screen, color, (inner_x, cur_y + 4, 6, 22), border_radius=2)
            lab = self.font_stat.render(label, True, TEXT_SECONDARY)
            self.screen.blit(lab, (inner_x + 14, cur_y + 6))
            num = self.font_big_count.render(str(count), True, TEXT_PRIMARY)
            self.screen.blit(num, (inner_x + inner_w - num.get_width(), cur_y))
            cur_y += 38

        cur_y += 12

        # 인구 그래프 (3개 라인 한 그래프)
        title = self.font_stat.render('POPULATIONS OVER TIME', True, TEXT_SECONDARY)
        self.screen.blit(title, (inner_x, cur_y)); cur_y += 18
        graph_h = 180
        self._draw_multi_graph(inner_x, cur_y, inner_w, graph_h, world.history)
        cur_y += graph_h + 18

        # 출생/사망 통계
        title2 = self.font_stat.render('BIRTHS / DEATHS', True, TEXT_SECONDARY)
        self.screen.blit(title2, (inner_x, cur_y)); cur_y += 22
        b = world.births
        d = world.deaths
        for label, val in [
            ('herb births',   b['herb']),
            ('carn births',   b['carn']),
            ('herb killed',   d['herb_killed']),
            ('herb starved',  d['herb_starve']),
            ('carn starved',  d['carn_starve']),
        ]:
            self._draw_stat(inner_x, cur_y, inner_w, label, str(val))
            cur_y += 20

        cur_y += 8
        # 시조/장로/사이클 — 메멘토 모리
        title3 = self.font_stat.render('LINEAGE & CYCLE', True, TEXT_SECONDARY)
        self.screen.blit(title3, (inner_x, cur_y)); cur_y += 22

        from .entities import Carnivore as _C
        f = world.founder
        e = world.elder
        if f is not None and f.alive:
            kind = 'carn' if isinstance(f, _C) else 'herb'
            self._draw_stat(inner_x, cur_y, inner_w,
                            'founder', f'{kind} #{f.id} · {f.children} kids')
        else:
            self._draw_stat(inner_x, cur_y, inner_w, 'founder', '—')
        cur_y += 20
        if e is not None and e.alive:
            kind = 'carn' if isinstance(e, _C) else 'herb'
            self._draw_stat(inner_x, cur_y, inner_w,
                            'elder', f'{kind} #{e.id} · age {e.age}')
        else:
            self._draw_stat(inner_x, cur_y, inner_w, 'elder', '—')
        cur_y += 20

        # 사이클 진행도 바
        light = world.day_light()
        phase = world.day_phase()
        bar_x = inner_x
        bar_y = cur_y + 2
        bar_w = inner_w
        bar_h = 6
        pygame.draw.rect(self.screen, (40, 44, 52), (bar_x, bar_y, bar_w, bar_h),
                         border_radius=3)
        fill_w = int(bar_w * phase)
        bar_color = self._lerp_color((90, 120, 180), (230, 200, 120), light)
        if fill_w > 0:
            pygame.draw.rect(self.screen, bar_color,
                             (bar_x, bar_y, fill_w, bar_h), border_radius=3)
        cur_y += 14
        cycle_txt = self.font_label.render(
            f'day cycle  {int(phase * 100)}%  ·  light {light:.2f}',
            True, TEXT_SECONDARY)
        self.screen.blit(cycle_txt, (inner_x, cur_y))
        cur_y += 16
        # 시체/공포 카운트 (간결하게)
        meta_txt = self.font_label.render(
            f'carrion {len(world.carrions)}  ·  fear {len(world.fear_marks)}',
            True, TEXT_SECONDARY)
        self.screen.blit(meta_txt, (inner_x, cur_y))

    def _draw_stat(self, x, y, w, label, value):
        lab = self.font_label.render(label, True, TEXT_SECONDARY)
        val = self.font_stat.render(value, True, TEXT_PRIMARY)
        self.screen.blit(lab, (x, y + 2))
        self.screen.blit(val, (x + w - val.get_width(), y))

    def _draw_multi_graph(self, x: int, y: int, w: int, h: int,
                          history: List[dict]):
        pygame.draw.rect(self.screen, (28, 32, 40), (x, y, w, h), border_radius=4)
        if not history:
            return
        data = history[-300:]
        if not data:
            return

        # 각 시리즈의 max로 정규화 (각 라인은 자기 max 기준)
        max_p = max((d['plants'] for d in data), default=1) or 1
        max_h = max((d['herb'] for d in data), default=1) or 1
        max_c = max((d['carn'] for d in data), default=1) or 1
        # 식물은 보통 너무 많아서 그대로면 다른 라인 안 보임 — 같은 스케일 묶기
        unified_max = max(max_p, max_h * 8, max_c * 30) * 1.1

        def line(key: str, color: tuple, scale: float = 1.0):
            pts = []
            for i, d in enumerate(data):
                px = x + i * w / max(1, len(data) - 1)
                v = d[key] * scale
                py = y + h - (v / unified_max) * h
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, color, False, pts, 2)

        # 식물은 그대로, 초식 ×8, 육식 ×30 (스케일 강조)
        line('plants', (140, 200, 90))
        line('herb', HERB_COLOR, 8.0)
        line('carn', CARN_COLOR, 30.0)

        # 범례
        legend_y = y + 8
        for label, color in [
            ('plants',   (140, 200, 90)),
            ('herb ×8',  HERB_COLOR),
            ('carn ×30', CARN_COLOR),
        ]:
            pygame.draw.circle(self.screen, color, (x + 14, legend_y + 6), 4)
            txt = self.font_label.render(label, True, TEXT_SECONDARY)
            self.screen.blit(txt, (x + 22, legend_y))
            legend_y += 14

    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()
