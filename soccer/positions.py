"""선수별 세분화 포지션 — 수동 + 휴리스틱.

squads.json 의 role (GK/DEF/MID/FWD) 만으로는 4-3-3 슬롯 구분이 안 됨.
이 모듈:
  1) KNOWN_POSITIONS — rating 80+ 약 180명 수동 매핑 (이름 → 세부 포지션)
  2) infer_position_for_squad — 나머지는 팀별 rating 순으로 자동 분류

세부 포지션 코드 (10종):
  GK    골키퍼
  CB    센터백
  LB    레프트백
  RB    라이트백
  DM    수비형 미드 (수미)
  CM    중앙 미드 (중미)
  AM    공격형 미드 (공미)
  LW    레프트 윙어
  RW    라이트 윙어
  ST    스트라이커

멀티 포지션 선수는 '가장 자주 뛰는 포지션 1개' 로 정함.
수정/추가: KNOWN_POSITIONS 에 한 줄 추가만 하면 됨.
"""

# ──────────────────────────────────────────────────────────────────
# 수동 매핑 — rating 80+ (~180명). 모르거나 다른 거면 자유 수정.
# 키: squads.json 의 'name' 필드와 정확히 일치 (대소문자/공백 포함)
# ──────────────────────────────────────────────────────────────────
KNOWN_POSITIONS: dict = {
    # ── 90+ ──────────────────────────────
    'Kylian Mbappe':         'LW',
    'Erling Haaland':        'ST',
    'Vinicius Jr':           'LW',
    'Lionel Messi':          'RW',

    # ── 89 ──────────────────────────────
    'Alisson':               'GK',
    'Virgil van Dijk':       'CB',
    'Thibaut Courtois':      'GK',
    'Kevin De Bruyne':       'AM',
    'Mohamed Salah':         'RW',
    'Rodri':                 'DM',
    'Cristiano Ronaldo':     'ST',
    'Harry Kane':            'ST',

    # ── 88 ──────────────────────────────
    'Lee Kang-in':           'AM',     # 손흥민과 윙어 다투기보다 공미가 잦음

    # ── 87 ──────────────────────────────
    'Rodrygo':               'RW',
    'Manuel Neuer':          'GK',
    'Jamal Musiala':         'AM',
    'Frenkie de Jong':       'CM',
    'Lamine Yamal':          'RW',
    'Emiliano Martinez':     'GK',
    'Ruben Dias':            'CB',
    'Bruno Fernandes':       'AM',
    'Jude Bellingham':       'CM',

    # ── 86 ──────────────────────────────
    'Kim Min-jae':           'CB',
    'Marquinhos':            'CB',
    'Joshua Kimmich':        'CM',
    'Florian Wirtz':         'AM',
    'Romelu Lukaku':         'ST',
    'Pedri':                 'CM',
    'Federico Valverde':     'CM',
    'Antoine Griezmann':     'AM',
    'Martin Odegaard':       'AM',
    'Lautaro Martinez':      'ST',
    'Bernardo Silva':        'RW',
    'Luis Diaz':             'LW',
    'Phil Foden':            'AM',
    'Bukayo Saka':           'RW',
    'Luka Modric':           'CM',

    # ── 85 ──────────────────────────────
    'Son Heung-min':         'LW',
    'Casemiro':              'DM',
    'Achraf Hakimi':         'RB',
    'Antonio Rudiger':       'CB',
    'Mike Maignan':          'GK',
    'William Saliba':        'CB',

    # ── 84 ──────────────────────────────
    'Alphonso Davies':       'LB',
    'Gabriel Magalhaes':     'CB',
    'Eder Militao':          'CB',
    'Bruno Guimaraes':       'CM',
    'Raphinha':              'RW',
    'Andy Robertson':        'LB',
    'Hakan Calhanoglu':      'CM',
    'Kai Havertz':           'ST',
    'Leroy Sane':            'RW',
    'Matthijs de Ligt':      'CB',
    'Memphis Depay':         'ST',
    'Cody Gakpo':            'LW',
    'Alexander Isak':        'ST',
    'Nico Williams':         'LW',
    'Ronald Araujo':         'CB',
    'Aurelien Tchouameni':   'DM',
    'Ousmane Dembele':       'RW',
    'Sadio Mane':            'LW',
    'Cristian Romero':       'CB',
    'Enzo Fernandez':        'CM',
    'Alexis Mac Allister':   'CM',
    'Julian Alvarez':        'ST',
    'Riyad Mahrez':          'RW',
    'David Alaba':           'CB',
    'Vitinha':               'CM',
    'Rafael Leao':           'LW',
    'Diogo Jota':            'LW',
    'John Stones':           'CB',
    'Trent Alexander-Arnold':'RB',
    'Declan Rice':           'DM',
    'Cole Palmer':           'AM',

    # ── 83 ──────────────────────────────
    'Oh Hyeon-gyu':          'ST',
    'Kaoru Mitoma':          'LW',
    'Dayot Upamecano':       'CB',
    'Lisandro Martinez':     'CB',

    # ── 82 ──────────────────────────────
    'Jonathan David':        'ST',
    'Granit Xhaka':          'CM',
    'Lucas Paqueta':         'AM',
    'Yassine Bounou':        'GK',
    'Hakim Ziyech':          'RW',
    'Christian Pulisic':     'RW',
    'Jonathan Tah':          'CB',
    'Serge Gnabry':          'RW',
    'Franck Kessie':         'CM',
    'Moises Caicedo':        'DM',
    'Nathan Ake':            'CB',
    'Xavi Simons':           'AM',
    'Takefusa Kubo':         'RW',
    'Viktor Gyokeres':       'ST',
    'Jeremy Doku':           'LW',
    'Unai Simon':            'GK',
    'Daniel Carvajal':       'RB',
    'Fabian Ruiz':           'CM',
    'Jose Maria Gimenez':    'CB',
    'Darwin Nunez':          'ST',
    'Theo Hernandez':        'LB',
    'Jules Kounde':          'RB',     # CB 도 자주 뛰지만 프리메라/대표팀 RB
    'Eduardo Camavinga':     'CM',
    'Marcus Thuram':         'ST',
    'Kalidou Koulibaly':     'CB',
    'Nicolas Otamendi':      'CB',
    'Rodrigo De Paul':       'CM',
    'Angel Di Maria':        'RW',
    'Diogo Costa':            'GK',
    'Joao Cancelo':          'RB',
    'Nuno Mendes':           'LB',
    'James Rodriguez':       'AM',
    'Jordan Pickford':       'GK',
    'Kyle Walker':           'RB',
    'Dominik Livakovic':     'GK',
    'Josko Gvardiol':        'CB',
    'Mateo Kovacic':         'CM',

    # ── 81 ──────────────────────────────
    'Edson Alvarez':         'DM',
    'Yann Sommer':           'GK',
    'Manuel Akanji':         'CB',
    'Danilo':                'RB',
    'Tijjani Reijnders':     'CM',
    'Loïs Openda':           'ST',
    'Aymeric Laporte':       'CB',
    'Alvaro Morata':         'ST',
    'Ibrahima Konate':       'CB',

    # ── 80 ──────────────────────────────
    'Santiago Gimenez':      'ST',
    'Hwang Hee-chan':        'LW',
    'Patrik Schick':         'ST',
    'Endrick':               'ST',
    'Richarlison':           'ST',
    'Lucas Beraldo':         'CB',
    'Pedro':                 'ST',
    'Sofyan Amrabat':        'DM',
    'Youssef En-Nesyri':     'ST',
    'Scott McTominay':       'CM',
    'Arda Guler':            'AM',
    'Nico Schlotterbeck':    'CB',
    'Leon Goretzka':         'CM',
    'Karim Adeyemi':         'LW',
    'Wilfried Zaha':         'LW',
    'Sebastien Haller':      'ST',
    'Enner Valencia':        'ST',
    'Denzel Dumfries':       'RB',
    'Stefan de Vrij':        'CB',
    'Wataru Endo':           'DM',
    'Daichi Kamada':         'AM',
    'Dejan Kulusevski':      'RW',
    'Youri Tielemans':       'CM',
    'Amadou Onana':          'DM',
    'Leandro Trossard':      'LW',
    'Mehdi Taremi':          'ST',
    'Robin Le Normand':      'CB',
    'Pau Cubarsi':           'CB',
    'Martin Zubimendi':      'DM',
    'Mikel Oyarzabal':       'LW',
    'Ferran Torres':         'RW',
    'Inigo Martinez':        'CB',
    'Manuel Ugarte':         'DM',
    'Rodrigo Bentancur':     'CM',
    'Luis Suarez':           'ST',
    "N'Golo Kante":          'DM',
    'Adrien Rabiot':         'CM',
    'Bradley Barcola':       'LW',
    'Edouard Mendy':         'GK',
    'Alexander Sorloth':     'ST',
    'Nahuel Molina':         'RB',
    'Ismael Bennacer':       'CM',
    'Konrad Laimer':         'CM',
    'Pepe':                  'CB',
    'Joao Palhinha':         'DM',
    'Joao Felix':            'AM',
    'Goncalo Ramos':         'ST',
    'David Ospina':          'GK',
    'Davinson Sanchez':      'CB',
    'Marc Guehi':            'CB',
    'Luke Shaw':             'LB',
    'Anthony Gordon':        'LW',
    'Ollie Watkins':         'ST',
    'Lewis Dunk':            'CB',
    'Andrej Kramaric':       'ST',
    'Mohammed Kudus':        'RW',
    'Thomas Partey':         'DM',
}


# ──────────────────────────────────────────────────────────────────
# 휴리스틱 — KNOWN_POSITIONS 에 없는 선수
#
# 팀 squad 안에서 같은 role 끼리 rating 순으로 정렬 후 슬롯 분배.
# 모든 팀이 4-3-3 처럼 본다고 가정 (실제 다른 포메이션은 추후 확장).
# ──────────────────────────────────────────────────────────────────

# DEF 분배 — rating 내림차순 정렬 후 인덱스별
#   0,1: CB / 2: LB / 3: RB / 4: CB (백5) / 5+: CB
_DEF_BY_RANK = ['CB', 'CB', 'LB', 'RB', 'CB', 'CB', 'CB']

# MID 분배 — 인덱스별
#   0: CM / 1: AM / 2: DM / 3: CM / 4+: CM
_MID_BY_RANK = ['CM', 'AM', 'DM', 'CM', 'CM', 'CM']

# FWD 분배 — 인덱스별
#   0: ST / 1: LW / 2: RW / 3+: ST
_FWD_BY_RANK = ['ST', 'LW', 'RW', 'ST', 'LW', 'RW']


def infer_position_for_squad(squad: list) -> dict:
    """팀 squad 의 각 선수에게 세부 포지션 매핑.
    KNOWN_POSITIONS 우선, 없으면 role 안에서 rating 순으로 휴리스틱.
    반환: {name: position}"""
    out: dict = {}

    # role 별로 분리 + rating 내림차순 정렬
    role_groups: dict = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
    for p in squad:
        if p.role in role_groups:
            role_groups[p.role].append(p)
    for role in role_groups:
        role_groups[role].sort(key=lambda x: -x.rating)

    # GK 는 전부 'GK'
    for p in role_groups['GK']:
        out[p.name] = KNOWN_POSITIONS.get(p.name, 'GK')

    # DEF / MID / FWD — 휴리스틱 테이블 사용. KNOWN_POSITIONS 우선.
    # 휴리스틱은 'KNOWN 에 없는 선수만' 순서로 채움 (KNOWN 선수가
    # 슬롯을 빼앗으면 어색해지니까, KNOWN 은 그냥 자기 자리 가짐).
    for role, table in (('DEF', _DEF_BY_RANK), ('MID', _MID_BY_RANK),
                          ('FWD', _FWD_BY_RANK)):
        unknown_idx = 0
        for p in role_groups[role]:
            if p.name in KNOWN_POSITIONS:
                out[p.name] = KNOWN_POSITIONS[p.name]
            else:
                pos = table[unknown_idx] if unknown_idx < len(table) else table[-1]
                out[p.name] = pos
                unknown_idx += 1

    return out
