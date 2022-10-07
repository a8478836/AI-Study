# CycleGAN study
## Data pipeline
### Data Loading
### Data Preprocessing
1. Initializer --> 다양한 방법이 있음 확인 필요
  1) RandomNormal
2. tf.data.AUTOTUNE
3. Data augmentation: 이미지를 변형하여 CNN의 성능을 향상, 즉 이미지의 레이블을 변경하지 않고 픽셀을 변형한다. 매우 보편적으로 사용하는 방법
4. Padding: 이미지의 출력 크기를 보정하기 위해 비어있는 공간의 값을 의미 없는 값으로 채우는 것을 의미
  1) zero padding: 0으로 채움
  2) reflection padding; 가장자리 기준으로 input 값을 영역에 반전하여 복사
  3) replication padding: 가장자리 기준으로 input 값을 영역에 그대로 복사



### Data Shuffle
## Model Create
1. Functional API vs Sequential API
Sequential API를 사용하면 상당히 직관적이고 편리하게 layer를 쌓을 수 있지만, 복잡한 model 구현이 어려움
Functional API를 사용하면 좀 더 복잡한 인공신경망을 구현할 수 있음. 단, 사용하기 어려움

2. Bactch Normalization vs Layer Normalization vs Instance Normalization vs Group Normalization

  1) Batch Normalization
  ※ Batch Normalization: Accelerating Deep Network Training b y Reducing Internal Covariate Shift 논문 및 출처: https://eehoeskrap.tistory.com/430 블로그 참조
  신경망에서 학습이 진행될 때 cost 가 0이 되는 global minima를 찾는 쪽으로 학습이 진행된다. 이 때 gradient 기반의 방법들은 미분을 사용하여 극값을 찾게 되는데,
  미분의 특성상 값의 변화에 따라 결과값이 달라진다. 만약 이 변화량이 매우 작거(vanishing)나 매우 커질(exploding) 경우 학습 오차를 줄이지 못하고 특정 값에 수렴해버리는 잘못된 학습이 일어난다.
  이러한 문제를 해결하기 위해 sigmoid나 tanh 등의 비선형 방식의 함수들은 입력된 값을 특정 범위 값으로 매핑해버리는데 이것은 첫 번째 입력 값에 큰 변화량이 있어도 매우 작은 값으로 맵핑하기 때문에
  더더욱 학습이 안된다(이런 비선형 함수를 사용하는 레이어를 쌓을 때마다). 그래서 sigmoid 대신 ReLU를 쓰기도하며 다음과 같은 방식들도 존재한다.
    - 활성화 함수 변경
    - 초기값 설정
    - learning rate 조정
  이 밖에 직접적으로 학습하는 과정을 전체적으로 안정화하자는 전략이 Batch Normalization이다. 정규화(normalization)을 사용하면 local minimum에 빠질 가능성을 낮출 수 있다.
  학습이 불안정하게 진행되는 이유를 internal Covariance Shift 문제라고 한다. 이것은 네트워크의 레이어나 활성화 함수마다 입력 값의 분산이 달라지는 현상이다.

  해당 현상을 막기위해 whitening 이라는 방법이 있는데, 이것은 각 레이어의 분산을 0, 표준편차가 1인 값으로 정규화하는 방법이 있다. 하지만 이 방법은 covariance matrix 계산과 inverse의 계산이
  필요하기 때문에 계산량이 많고, 일부 파라미터의 영향을 무시한다. 단순히 whitening을 사용하면 최적의 파라미터를 찾기위한 역전파와 무관하게 특정 값이 커질 수 있다.

  whitening의 문제를 해결한것이 바로 batch normalization
  
  ![image](https://user-images.githubusercontent.com/28583606/194516665-19ad14aa-9483-4090-ba21-ef22f0afcdf3.png)



## Model Training
## Model Testing
## Model Deploy
