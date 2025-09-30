#!/usr/bin/env python3

import os
from pathlib import Path

def find_and_print_files(root_dir="."):
    """
    마크다운과 파이썬 파일들을 찾아서 내용과 함께 출력
    """
    root_path = Path(root_dir).resolve()
    files = []

    # .venv 디렉토리를 제외하고 마크다운과 파이썬 파일 검색
    for pattern in ["**/*.md", "**/*.py"]:
        for file_path in root_path.glob(pattern):
            # .venv, __pycache__ 경로가 포함되어 있으면 스킵
            relative_path = str(file_path.relative_to(root_path))
            if ".venv" not in relative_path and "__pycache__" not in relative_path:
                files.append(file_path)

    files = sorted(files)

    if not files:
        print("파일을 찾을 수 없습니다.")
        return

    # 확장자별 개수 세기
    md_count = sum(1 for f in files if f.suffix == '.md')
    py_count = sum(1 for f in files if f.suffix == '.py')

    print(f"총 {len(files)}개의 파일을 찾았습니다.")
    print(f"- 마크다운 파일: {md_count}개")
    print(f"- 파이썬 파일: {py_count}개\n")
    print("=" * 80)

    for file_path in files:
        # 상대 경로 계산
        try:
            relative_path = file_path.relative_to(root_path)
        except ValueError:
            relative_path = file_path

        # 파일 타입에 따른 아이콘
        icon = "📝" if file_path.suffix == '.md' else "🐍"

        print(f"\n{icon} 파일 위치: {relative_path}")
        print("-" * 80)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    print(content)
                else:
                    print("[빈 파일]")
        except Exception as e:
            print(f"[읽기 오류: {e}]")

        print("\n" + "=" * 80)

if __name__ == "__main__":
    print("프로젝트 파일 수집 시작...")
    print("=" * 80)
    find_and_print_files()
    print("\n파일 출력 완료!")