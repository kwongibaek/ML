# ML

앙상블학습 (ensemble learning)
: 여러 개의 분류기를 생성하고 각 예측들을 결합함으로써 보다 정확한 예측을 도출하는 기법

1) 랜덤 포레스트 (random forest)
: 결정 트리를 랜덤하게 만들어 각 결정 트리의 예측들을 사용해 최종 예측을 만듬

  --> 부트스트랩 샘플링을 사용 (복원 추출)
  --> 노드 분할 시, 전체 특성 개수의 제곱근만큼 특성 선택 
    => 이 중에서 최선의 분할을 찾음 (좀 더 많은 특성이 훈련에 기여할 기회를 얻음 = 과대적합 감소)
  --> feature_importance_ 를 알 수 있다
  --> OOB (out of bag): 부트스트랩 샘플에 포함되지 않는 샘플 
    => OOB를 가지고 결정 트리 평가 가능 (검증 세트의 역할)
    => cross_validate를 대신할 수 있어 더 많은 샘플 훈련 가능

2) 엑스트라 트리 (extra trees)
: bootstrap 샘플링이 아닌 전체 훈련 세트를 쓰는 랜덤 포레스트

  --> DecisionTreeClassifier()의 splitter가 random인 트리
    => RF는 최적의 분할점인데에 반해 랜덤하게 분할
  --> 랜덤 포레스트보다 더 많은 트리를 훈련해야하나 더 빠른 계산 속도를 지님
    (랜덤 포레스트와 달리 노드의 분할점도 랜덤하게 정함)
  --> bias 상승, variance는 낮아짐 (random forest에 비해)
    (bias가 낮을수록 모델이 패턴을 잘 학습함, variance가 낮을수록 과적합 감소 => trade off 관계)

3) 그레이디언트 부스팅 (gradient boosting)
: 깊이가 얕은 결정트리를 사용하여 이전 트리의 오차 보완하여 과대적합에 강하고 높은 일반화 성능 기대 가능
  (경사하강법을 앙상블에 추가) 

 --> 이때 파라미터가 바뀌는 것은 아님 
    => 새 weak learner 추가함으로서 함수 자체를 점진적으로 개선
  -->subsample을 통해 훈련세트의 비율을 정할 수 있음 
    => overfitting 막음

4)히스토그램 기반 그레디언트 부스팅 (Histogram-based Gradient boosting)
: 그레디언트 부스팅을 개선하기 위해 나온 알고리즘으로 정형 데이터를 다루는 알고리즘 중 가장 인기 많음

  --> feature의 값을 256개 구간으로 나눔 
    => 노드를 분할할 때 최적의 분할을 빠르게 찾을 수 있다. 
  --> NaN (누락된 값)이 있더라도 256개의 구간 중 하나를 이를 위해서 사용되기에 전처리할 필요X
  --> 자체적으로 특성 중요도 제공 X
    => permutation_importance() 함수 사용 
      :특성을 하나씩 랜덤하게 섞어서 모델의 성능 변화 관찰을 통해 특성 중요도 측정
      (n_repeats로 각 feature 별로 몇번 섞어서 결과 낼지 정함)
      (train set, test set, 추정기 모델 [fit을 가진 객체] 모두 사용 가능)

5) XGBoost
: gradient boosting을 구현한 라이브러리 
  (GBDT: Gradient Boosting Decision Tree)

  --> level-wise 방식
  --> LightGBM보다 느리지만 소규모 dataset에서 안정적
  --> sklearn의 cross_validate와 함께 사용 가능

6) LightGBM
: 마이크로소프트에서 개발한 라이브러리 
  (GBDT: Gradient Boosting Decision Tree)

  --> leaf-wise 방식으로 손실을 가장 많이 줄일 수 있는 leaf 먼저 확장
    => 속도가 굉장히 빠름
--> 데이터가 작을 시 과대적합 발생 
=>대용량 데이터에서 유리
