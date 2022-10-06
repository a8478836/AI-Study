# AI-Study
## Data pipeline
데이터를 운반하는 효율적인 입력 파이프라인이 필요. 보통 tf.data.Dataset을 상속해서 클래스를 정의하는 듯
데이터를 적재, 전처리, 셔플 하는 과정을 병렬적으로 수행한다면 전체적인 시간을 줄일 수 있고 컴퓨팅 리소스를 최대한 활용할 수 있음
https://ahnjg.tistory.com/32 해당 블로그에 자세하게 설명되어 있음

### Data Loading
### Data Preprocessing
1. Standardization: 값의 scale을 평균 0, 분산 1로 표준정규분포를 따르도록 변경, 이상치를 확인하기에 좋음
2. Normalization: 데이터의 scale을 0~1 사이의 값으로 변환하는 것 --> 특정 값이 크거나 너무 작으면 학습시 weight 조정에 영향을 주기 때문에 변환함
3. Regulariztion: 위의 방법들과 다르게 Regularization은 weight가 너무 커지거나 작아지는 것을 방지하기 위한 방법 L1 정규화와 L2 정규화가 있음 / 따로 문서 만들기
### Data Shuffle
## Model Create
## Model Training
1. 이미지 학습
2. NLP
3. 추천 모델
4. 시계열 데이터 학습
## Model Testing
## Model Deploy
### Kubernetes
컨테이너 기반의 오케스트레이션 툴
컨테이너 런타임(docker, crio, containerd 등)은 k8s 최신버전에서 docker가 depricated됨
### KubeFlow
### jenkins
