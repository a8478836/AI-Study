# AI-Study
## Data pipeline
데이터를 운반하는 효율적인 입력 파이프라인이 필요 이것은 모델을 학습하는 것과 독립적으로 이루어짐 보통 tf.data.Dataset을 상속해서 클래스를 정의하는 듯
### Data Loading
### Data Preprocessing
1. Standardization: 값의 scale을 평균 0, 분산 1로 표준정규분포를 따르도록 변경, 이상치를 확인하기에 좋음
2. Normalization: 데이터의 scale을 0~1 사이의 값으로 변환하는 것 --> 특정 값이 크거나 너무 작으면 학습시 weight 조정에 영향을 주기 때문에 변환함
3. Regulariztion: 위의 방법들과 다르게 Regularization은 weight가 너무 커지거나 작아지는 것을 방지하기 위한 방법 L1 정규화와 L2 정규화가 있음 / 따로 문서 만들기
### Data Shuffle

