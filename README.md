# 월드컵 2026 경기 예측 (Elo + Poisson 모델)

두 팀을 넣으면 **예상 골 수 / 승·무·패 확률 / 가장 유력한 스코어**를 바로 예측합니다.
학습된 모델은 `wc_model.pkl`(pickle) 로 저장되어 있어, 평가자 PC에서 그대로 재현·실행할 수 있습니다.

## 구성 파일

| 파일 | 설명 |
|------|------|
| `worldcup_2026_czech_fixed.ipynb` | **학습 노트북.** 실행하면 모델을 학습해 `wc_model.pkl` 을 생성합니다. |
| `predict.py` | **예측 스크립트.** `wc_model.pkl` 을 불러와 "두 팀 → 경기 결과" 를 출력합니다. |
| `wc_model.pkl` | 학습이 끝난 모델(팀 Elo + Poisson 회귀 + 보정값). 바로 예측에 사용. |
| `final_elo2_2026.csv` | 팀별 Elo 레이팅 (학습 입력) |
| `all_matched_country_v2.csv` | 나라 이름 통일용 매핑표 (학습 입력) |
| `example_submission.csv` | 월드컵 조 편성 (조별 순위 산출용) |
| `results.csv` | 전 세계 축구 경기 기록 (없으면 노트북이 자동 다운로드) |
| `requirements.txt` | 필요한 파이썬 패키지 |

## 1. 설치

```bash
pip install -r requirements.txt
```

## 2. 바로 예측하기 (제출된 pkl 사용)

```bash
# 두 팀 이름을 넣으면 결과가 나옵니다 (월드컵 = 중립 경기 기본)
python predict.py "Czech Republic" "Mexico"

# 첫 번째 팀을 홈팀으로 두려면 --home
python predict.py "South Korea" "Japan" --home
```

출력 예시:
```
Czech Republic vs Mexico  (중립)
  예상 골:  0.86 / 1.23
  Czech Republic 승 25.0%  /  무 30.9%  /  Mexico 승 44.1%
  가장 유력한 스코어: 0 - 1
```

코드 안에서 함수로 쓸 수도 있습니다:
```python
from predict import predict_match
print(predict_match("Czech Republic", "Mexico"))
# {'teamA': 'Czech Republic', 'teamB': 'Mexico', 'neutral': True,
#  'expected_goals': (0.86, 1.23), 'prob_A_win': 0.25, 'prob_draw': 0.309,
#  'prob_B_win': 0.441, 'most_likely_score': (0, 1)}
```

## 3. 모델을 처음부터 다시 학습하기 (선택)

`wc_model.pkl` 을 직접 다시 만들고 싶다면 노트북을 실행하세요.
**중요:** pickle 은 scikit-learn 버전에 민감하므로, 노트북을 실행한 환경과
`predict.py` 를 실행하는 환경을 **동일하게** 유지해야 버전 경고 없이 재현됩니다.

```bash
# 노트북 전체를 실행해 wc_model.pkl 을 새로 생성
jupyter nbconvert --to notebook --execute --inplace worldcup_2026_czech_fixed.ipynb
```

## 모델 요약

- **Elo**: 팀별 실력 점수. 이름 불일치로 빠졌던 FIFA 팀(체코 등)을 원천 경기기록으로 복구해 보강.
- **Poisson 회귀**: 두 팀의 실력차 → 각 팀 예상 골 수.
- **Dixon-Coles 보정**: 0:0·1:0 같은 적은 골 경기의 확률을 현실에 맞게 손질.
- 예상 골에서 모든 스코어 확률표를 만들고, 이를 승·무·패 확률로 합산.
