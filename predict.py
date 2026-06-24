#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py — 학습된 모델(wc_model.pkl)을 불러와 "두 팀 -> 경기 결과"를 바로 예측합니다.

사용법(둘 다 가능):
  1) 커맨드라인:
       python predict.py "Czech Republic" "Mexico"
       python predict.py "South Korea" "Brazil" --home   # 첫 팀을 홈으로 (기본은 중립 경기)
  2) 다른 코드에서 함수로:
       from predict import predict_match
       print(predict_match("Czech Republic", "Mexico"))

준비물: wc_model.pkl  (노트북 worldcup_2026_czech_fixed.ipynb 를 실행하면 자동 생성됩니다)
필요 패키지: numpy, scipy, scikit-learn  (requirements.txt 참고)
"""
import sys
import pickle
import argparse
import difflib

import numpy as np
from scipy.stats import poisson

MODEL_PATH = "wc_model.pkl"


def load_model(path=MODEL_PATH):
    """pkl 파일에서 학습된 모델 묶음을 불러옵니다."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        sys.exit(f"[오류] '{path}' 파일이 없습니다. 먼저 노트북을 실행해 모델을 저장하세요.")


def resolve_team(name, elo, name_map):
    """사용자가 입력한 팀 이름을 모델이 아는 정식 이름으로 바꿔 줍니다.
    1) 이름 사전(NAME)으로 변환  2) 대소문자 무시 매칭  3) 비슷한 이름 추천."""
    # 1) 표기 사전 적용 (예: Czechia -> Czech Republic)
    name = name_map.get(name, name)
    if name in elo:
        return name
    # 2) 대소문자/공백 무시하고 찾기
    low = {k.lower(): k for k in elo}
    if name.lower() in low:
        return low[name.lower()]
    # 3) 못 찾으면 비슷한 이름을 추천하고 종료
    near = difflib.get_close_matches(name, list(elo.keys()), n=5, cutoff=0.5)
    msg = f"[오류] '{name}' 팀을 모델에서 찾을 수 없습니다."
    if near:
        msg += "\n  혹시 이 팀인가요? -> " + ", ".join(near)
    sys.exit(msg)


def score_matrix(L, M, rho, maxg):
    """예상 골(L, M)로부터 모든 스코어(0:0, 1:0, ...)의 확률표를 만듭니다. (Dixon-Coles 보정 포함)"""
    i = np.arange(maxg + 1)
    P = np.outer(poisson.pmf(i, L), poisson.pmf(i, M))
    P[0, 0] *= 1 - L * M * rho
    P[0, 1] *= 1 + L * rho
    P[1, 0] *= 1 + M * rho
    P[1, 1] *= 1 - rho
    return P / P.sum()


def win_draw_loss(P):
    """스코어 확률표를 (홈/팀A 승, 무, 원정/팀B 승) 확률로 합칩니다."""
    return np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()


def predict_match(teamA, teamB, neutral=True, model=None):
    """팀A vs 팀B 예측 -> 결과를 dict 로 돌려줍니다.
    neutral=True 면 중립 경기(월드컵 기본), False 면 teamA 가 홈."""
    if model is None:
        model = load_model()
    elo, m, rho, maxg, name_map = (
        model["ELO_NOW"], model["model"], model["rho"], model["MAXG"], model["NAME"],
    )
    a = resolve_team(teamA, elo, name_map)
    b = resolve_team(teamB, elo, name_map)

    ed = (elo[a] - elo[b]) / 100          # 두 팀 실력차
    ish = 0 if neutral else 1             # 홈 보너스 여부
    L = float(m.predict([[ed, ish]])[0])  # 팀A 예상 골
    M = float(m.predict([[-ed, 0]])[0])   # 팀B 예상 골
    P = score_matrix(L, M, rho, maxg)
    pw, pd_, pl = win_draw_loss(P)

    # 가장 확률이 높은 스코어 찾기
    gi, gj = np.unravel_index(P.argmax(), P.shape)
    return {
        "teamA": a, "teamB": b, "neutral": neutral,
        "expected_goals": (round(L, 2), round(M, 2)),
        "prob_A_win": round(float(pw), 4),
        "prob_draw": round(float(pd_), 4),
        "prob_B_win": round(float(pl), 4),
        "most_likely_score": (int(gi), int(gj)),
    }


def _print_result(r):
    a, b = r["teamA"], r["teamB"]
    venue = "중립" if r["neutral"] else f"{a} 홈"
    L, M = r["expected_goals"]
    gi, gj = r["most_likely_score"]
    print(f"\n{a} vs {b}  ({venue})")
    print(f"  예상 골:  {L:.2f} / {M:.2f}")
    print(f"  {a} 승 {r['prob_A_win']*100:4.1f}%  /  무 {r['prob_draw']*100:4.1f}%  /  {b} 승 {r['prob_B_win']*100:4.1f}%")
    print(f"  가장 유력한 스코어: {gi} - {gj}\n")


def main():
    ap = argparse.ArgumentParser(description="두 팀 -> 경기 결과 예측 (wc_model.pkl 사용)")
    ap.add_argument("teamA", help="팀 A 이름 (예: \"Czech Republic\")")
    ap.add_argument("teamB", help="팀 B 이름 (예: \"Mexico\")")
    ap.add_argument("--home", action="store_true", help="팀 A 를 홈팀으로 (기본: 중립 경기)")
    ap.add_argument("--model", default=MODEL_PATH, help="모델 pkl 경로 (기본: wc_model.pkl)")
    args = ap.parse_args()
    model = load_model(args.model)
    r = predict_match(args.teamA, args.teamB, neutral=not args.home, model=model)
    _print_result(r)


if __name__ == "__main__":
    main()
