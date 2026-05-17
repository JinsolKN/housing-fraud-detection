# 부동산 허위매물 분류 해커톤 🏠

**DACON 부동산 허위매물 분류 해커톤: 가짜를 색출하라!**  
평가 지표: Macro F1 Score

---

## 문제 정의

부동산 매물 정보(가격, 위치, 게재일, 중개사무소 등)를 기반으로  
허위매물 여부(`허위매물여부`)를 분류하는 이진 분류 문제.

클래스 불균형이 존재하며, 단순 정확도보다 **Macro F1 Score**를 기준으로 평가.

---

## EDA 기반 피처 선택

단순히 모델을 돌리기 전, 각 피처가 실제로 분류에 기여하는지 EDA로 먼저 판단했습니다.

### 제거한 피처와 근거

| 피처 | 제거 근거 |
|---|---|
| `해당층` / `총층` | 층별 허위매물 비율이 전층에 걸쳐 균일 → 분류 신호 없음 |
| `방향` | 서향(W)이 허위 비율 높으나 소수 카테고리 노이즈 크고, 전반적 패턴 불명확 |
| `제공플랫폼` | 플랫폼별 허위매물 비율이 거의 동일 → 매물은 복수 플랫폼에 동시 등록되는 특성 반영 |
| `방수` / `욕실수` | 허위매물과 상관관계 낮음 |
| `매물확인방식` / `주차가능여부` | 분류 기여도 미미 |
| `전용면적` / `총주차대수` | 노이즈 대비 신호 부족 |

> 층별 분석: 1~5층에 매물이 집중되어 있으나 정상/허위 비율이 층마다 유사해 제거 판단  
> 방향 분석: 서향(W) 허위 79건으로 비율이 높아 보이나, NE·NW 등 소수 카테고리가 많아 카테고리 합산 전처리를 검토했으나 최종 제거

---

## 피처 엔지니어링

### 게재일 → 수치 피처 변환

날짜를 그대로 사용하는 대신 두 가지 방식으로 분리

```python
# 절대적 시간 흐름 (게재 시점이 오래될수록 다른 패턴 가능)
train['days_since_first_post'] = (train['게재일'] - train['게재일'].min()).dt.days

# 연간 계절 주기성 (사이클 인코딩)
train['cos_day'] = np.cos(2 * np.pi * train['day_of_year'] / 365)
train['sin_day'] = np.sin(2 * np.pi * train['day_of_year'] / 365)
```

### 관리비 이상치 처리

```python
train['관리비'] *= 10000          # 단위 환산
train = train[train['관리비'] <= 200000]  # 이상치 제거
```

허위매물에서 관리비 허위 기재(비현실적 고액)가 나타날 수 있다고 판단해  
200,000원 초과 데이터를 이상치로 처리

### 중개사무소 빈도 인코딩

```python
agency_counts = train['중개사무소'].value_counts()
train['중개사무소_Count'] = train['중개사무소'].map(agency_counts)
# test 미등장 사무소 → -1 처리
test['중개사무소_Count'] = test['중개사무소'].apply(
    lambda x: agency_counts[x] if x in agency_counts else -1
)
```

중개사무소명 자체보다 해당 사무소의 등록 매물 수가 허위매물 탐지에  
더 유효한 신호가 될 것이라 판단해 빈도 인코딩 적용

---

## 모델링

### 클래스 불균형 처리

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

허위매물 비율이 낮아 Macro F1 기준 소수 클래스 예측이 불리함  
→ SMOTE 오버샘플링으로 학습 데이터 균형 보정

### 하이퍼파라미터 최적화 (Optuna)

XGBoost, LightGBM, CatBoost 각각 **100 trial** Optuna 최적화 수행  
평가 기준: hold-out Macro F1 Score

| 모델 | 주요 탐색 파라미터 |
|---|---|
| XGBoost | n_estimators(820~840), max_depth(11~13), learning_rate, subsample, gamma |
| LightGBM | num_leaves(27~28), feature_fraction, bagging_fraction, learning_rate |
| CatBoost | iterations(765~775), depth(3~5), learning_rate, subsample |

### 앙상블: Soft Voting Classifier

```python
voting_clf = VotingClassifier(
    estimators=[('xgb', xgb_model), ('cat', cat_model), ('lgb', lgb_model)],
    voting='soft'
)
```

단일 모델 대비 세 모델의 예측 확률 평균이 Macro F1 기준으로 더 안정적

### 검증: 10-Fold KFold

```python
kf = KFold(n_splits=10, shuffle=True, random_state=42)
```

| 지표 | 평균 |
|---|---|
| **Macro F1** | **0.9197** |
| Accuracy | 0.9676 |
| Macro Precision | 0.9428 |
| Macro Recall | 0.9031 |

---

## 전체 파이프라인

```
원본 데이터
    ↓
EDA 기반 불필요 컬럼 제거 (층수, 방향, 플랫폼 등)
    ↓
관리비 단위 환산 + 이상치 필터링
    ↓
게재일 → sin/cos 사이클 인코딩 + 경과일 변환
    ↓
중개사무소 → 빈도 인코딩
    ↓
SMOTE 오버샘플링 (학습 데이터만)
    ↓
Optuna 하이퍼파라미터 최적화 (XGB / LGB / CAT 각 100 trial)
    ↓
Soft VotingClassifier 최종 학습
    ↓
10-Fold KFold 검증 → 제출
```

---

## 기술 스택

`Python` `XGBoost` `LightGBM` `CatBoost` `Optuna` `scikit-learn` `imbalanced-learn` `pandas` `matplotlib`

---

## 파일 구조

```
├── house_fraud.ipynb     # 전체 파이프라인 (EDA + 모델링)
├── train.csv          # 학습 데이터 (미포함)
├── test.csv           # 테스트 데이터 (미포함)
└── README.md

최종 순위: 584팀 중 2위로 마무리