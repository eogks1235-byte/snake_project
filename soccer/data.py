"""2026 FIFA 월드컵 — 데이터는 teams.json에서 로드.

48팀 / 12개 조 / 각 조 4팀.
개최국 고정: 멕시코=A1, 캐나다=B1, 미국=D1.

토너먼트 규칙:
  각 조 1·2위 24팀 + 3위 8팀(상위 점수 기준) = 32강 진출.
  R32 매치업은 BRACKET_MATCHES에 정의.

teams.json 편집:
  필드 추가/수정 시 자동 반영. 포메이션이나 능력치 입력값을 자유롭게 조정 가능.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TeamData:
    name: str
    code: str           # 3-letter ISO
    iso2: str           # 2-letter ISO (flag emoji 매핑용)
    fifa_points: float
    confederation: str  # UEFA / CONMEBOL / AFC / CAF / CONCACAF / OFC
    formation: str      # 4-3-3 / 4-2-3-1 / ...
    kit_primary: tuple  # (r, g, b) 홈 유니폼 메인 색
    kit_secondary: tuple # (r, g, b) 액센트 색
    style_tag: str      # possession / counter / parking-bus / balanced
    wc_titles: int      # 월드컵 우승 횟수
    secondary_formation: str = ''  # 보조 포메이션 (강팀 만나면 사용)
    altitude_native: bool = False  # 고도 적응 (멕시코·에콰도르·콜롬비아)
    heat_native: bool = False      # 더위 적응 (중동·아프리카·열대 남미·동남아)


def _hex_to_rgb(s: str) -> tuple:
    """'#RRGGBB' → (r, g, b). #이 없거나 잘못된 입력은 회색 fallback."""
    s = s.strip().lstrip('#')
    if len(s) != 6:
        return (180, 180, 180)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (180, 180, 180)


def flag_emoji(iso2: str) -> str:
    """'KR' → '🇰🇷' (regional indicator symbol 페어). 폰트가 지원해야 보임."""
    if not iso2 or len(iso2) != 2 or not iso2.isalpha():
        return ''
    a, b = iso2.upper()
    return chr(0x1F1E6 + ord(a) - ord('A')) + chr(0x1F1E6 + ord(b) - ord('A'))


def _load_groups() -> dict:
    json_path = Path(__file__).parent / 'teams.json'
    with json_path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    groups: dict = {letter: [] for letter in 'ABCDEFGHIJKL'}
    for entry in raw['teams']:
        td = TeamData(
            name=entry['name'],
            code=entry['code'],
            iso2=entry.get('iso2', ''),
            fifa_points=float(entry['fifa_points']),
            confederation=entry['confederation'],
            formation=entry.get('formation', '4-2-3-1'),
            kit_primary=_hex_to_rgb(entry.get('kit_primary', '#B4B4B4')),
            kit_secondary=_hex_to_rgb(entry.get('kit_secondary', '#404040')),
            style_tag=entry.get('style_tag', 'balanced'),
            wc_titles=int(entry.get('wc_titles', 0)),
            secondary_formation=entry.get('secondary_formation', ''),
            altitude_native=bool(entry.get('altitude_native', False)),
            heat_native=bool(entry.get('heat_native', False)),
        )
        groups[entry['group']].append(td)
    return groups


GROUPS: dict = _load_groups()


# ── 선수 스쿼드 (squads.json) ─────────────────────────────────

@dataclass(frozen=True)
class PlayerData:
    name: str
    role: str           # GK / DEF / MID / FWD (4분류 — 엔진 호환)
    rating: int
    is_star: bool = False
    pk_taker: bool = False
    position: str = ''  # 세부: GK/CB/LB/RB/DM/CM/AM/LW/RW/ST. positions.py 에서 자동 산출


def _load_squads() -> dict:
    """code → list[PlayerData] (16명/팀). positions.py 에서 세부 포지션 산출."""
    from .positions import infer_position_for_squad
    path = Path(__file__).parent / 'squads.json'
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    out: dict = {}
    for code, plist in raw.get('squads', {}).items():
        # 1차: position 비워둔 채로 임시 PlayerData 생성 (rating/role 필요)
        tmp_players = [
            PlayerData(
                name=p['name'],
                role=p['role'],
                rating=int(p['rating']),
                is_star=bool(p.get('is_star', False)),
                pk_taker=bool(p.get('pk_taker', False)),
            )
            for p in plist
        ]
        # 2차: 팀별 휴리스틱으로 세부 포지션 매핑
        pos_map = infer_position_for_squad(tmp_players)
        # 3차: position 채워서 다시 만듦 (frozen dataclass 라 replace 사용)
        from dataclasses import replace
        out[code] = [replace(p, position=pos_map.get(p.name, p.role))
                     for p in tmp_players]
    return out


SQUADS: dict = _load_squads()


# ──────────────────────────────────────────────────────────────
# 토너먼트 매치 정의 (대진표 그대로)
# ──────────────────────────────────────────────────────────────
# R32: 슬롯 코드는 다음 형식:
#   '1A' = A조 1위, '2B' = B조 2위
#   '3ABCDF' = A,B,C,D,F 조 중 한 곳에서 올라온 3위 (FIFA 매트릭스로 결정)
BRACKET_MATCHES = {
    # PATHWAY 1
    'M73': ('2A',     '2B'),
    'M74': ('1E',     '3ABCDF'),
    'M75': ('1F',     '2C'),
    'M77': ('1I',     '3CDFGH'),
    'M81': ('1D',     '3BEFIJ'),
    'M82': ('1G',     '3AEHIJ'),
    'M83': ('2K',     '2L'),
    'M84': ('1H',     '2J'),
    # PATHWAY 2
    'M76': ('1C',     '2F'),
    'M78': ('2E',     '2I'),
    'M79': ('1A',     '3CEFHI'),
    'M80': ('1L',     '3EHIJK'),
    'M85': ('1B',     '3EFGIJ'),
    'M86': ('1J',     '2H'),
    'M87': ('1K',     '3DEIJL'),
    'M88': ('2D',     '2G'),
}

# R16 ~ Final
KO_FOLLOWUPS = [
    # R16 (PATHWAY 1)
    ('M89', 'W74', 'W77'),
    ('M90', 'W73', 'W75'),
    ('M93', 'W83', 'W84'),
    ('M94', 'W81', 'W82'),
    # R16 (PATHWAY 2)
    ('M91', 'W76', 'W78'),
    ('M92', 'W79', 'W80'),
    ('M95', 'W86', 'W88'),
    ('M96', 'W85', 'W87'),
    # QF
    ('M97', 'W89', 'W90'),
    ('M98', 'W93', 'W94'),
    ('M99', 'W91', 'W92'),
    ('M100', 'W95', 'W96'),
    # SF
    ('M101', 'W97', 'W98'),
    ('M102', 'W99', 'W100'),
    # 3·4위전 + 결승
    ('M103', 'L101', 'L102'),
    ('M104', 'W101', 'W102'),
]


# 3위 슬롯 → 후보 조 (FIFA 매트릭스 단순화: 후보군 내에서 그리디 매칭)
THIRD_PLACE_SLOTS = {
    '3ABCDF':  set('ABCDF'),
    '3CDFGH':  set('CDFGH'),
    '3CEFHI':  set('CEFHI'),
    '3EHIJK':  set('EHIJK'),
    '3BEFIJ':  set('BEFIJ'),
    '3AEHIJ':  set('AEHIJ'),
    '3EFGIJ':  set('EFGIJ'),
    '3DEIJL':  set('DEIJL'),
}


def all_team_data() -> list:
    out = []
    for letter, teams in GROUPS.items():
        out.extend(teams)
    return out


# 매치 단계 라벨 (UI 표시용)
STAGE_LABELS = {
    **{f'M{i}': 'R32' for i in range(73, 89)},
    **{f'M{i}': 'R16' for i in range(89, 97)},
    **{f'M{i}': 'QF'  for i in (97, 98, 99, 100)},
    'M101': 'SF', 'M102': 'SF',
    'M103': '3rd', 'M104': 'FINAL',
}
