"""48팀 국기 PNG 미리 다운로드 — 오프라인 녹화 대비.

사용:  python soccer/prefetch_flags.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer.render import _resolve_flag_code, _ensure_flag_png, FLAG_DIR


def main():
    teams_path = Path(__file__).resolve().parent / 'teams.json'
    with teams_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    teams = data.get('teams', data)

    print(f'[prefetch] target: {len(teams)} teams  →  {FLAG_DIR}')
    new = 0
    cached = 0
    failed = []
    for t in teams:
        code = t['code']
        iso2 = t['iso2']
        flag_code = _resolve_flag_code(iso2, code)
        path = FLAG_DIR / f'{flag_code}.png'
        if path.exists():
            cached += 1
            continue
        result = _ensure_flag_png(flag_code)
        if result is None:
            failed.append((code, flag_code))
        else:
            new += 1

    print(f'[prefetch] done.  new={new}  cached={cached}  failed={len(failed)}')
    if failed:
        print('[prefetch] failed entries:')
        for code, fc in failed:
            print(f'  - {code} ({fc})')


if __name__ == '__main__':
    main()
