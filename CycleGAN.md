# CycleGAN study
## Data pipeline
### Data Loading
### Data Preprocessing
1. Initializer --> 다양한 방법이 있음 확인 필요
  i) RandomNormal
2. tf.data.AUTOTUNE
3. Data augmentation: 이미지를 변형하여 CNN의 성능을 향상, 즉 이미지의 레이블을 변경하지 않고 픽셀을 변형한다. 매우 보편적으로 사용하는 방법
4. Padding: 이미지의 출력 크기를 보정하기 위해 비어있는 공간의 값을 의미 없는 값으로 채우는 것을 의미
  i) zero padding: 0으로 채움
  ii) reflection padding; 가장자리 기준으로 input 값을 영역에 반전하여 복사
  iii) replication padding: 가장자리 기준으로 input 값을 영역에 그대로 복사



### Data Shuffle
## Model Create
1. Functional API vs Sequential API
Sequential API를 사용하면 상당히 직관적이고 편리하게 layer를 쌓을 수 있지만, 복잡한 model 구현이 어려움
Functional API를 사용하면 좀 더 복잡한 인공신경망을 구현할 수 있음. 단, 사용하기 어려움

2. Bactch Normalization vs Layer Normalization vs Instance Normalization vs Group Normalization

  i) Batch Normalization(BN)
  
  ※ Batch Normalization: Accelerating Deep Network Training b y Reducing Internal Covariate Shift 논문 및 출처: https://eehoeskrap.tistory.com/430 블로그 참조
  신경망에서 학습이 진행될 때 cost 가 0이 되는 global minima를 찾는 쪽으로 학습이 진행된다. 이 때 gradient 기반의 방법들은 미분을 사용하여 극값을 찾게 되는데,
  미분의 특성상 값의 변화에 따라 결과값이 달라진다. 만약 이 변화량이 매우 작거(vanishing)나 매우 커질(exploding) 경우 학습 오차를 줄이지 못하고 특정 값에 수렴해버리는 잘못된 학습이 일어난다.
  이러한 문제를 해결하기 위해 sigmoid나 tanh 등의 비선형 방식의 함수들은 입력된 값을 특정 범위 값으로 매핑해서 해결하려고 시도했다. 하지만 이것은 첫 번째 입력 값에 큰 변화량이 있어도 매우 작은 값으로 맵핑하기 때문에 더더욱 학습이 안된다(이런 비선형 함수를 사용하는 레이어를 쌓을 때마다). 그래서 sigmoid 대신 ReLU를 쓰기도하며 다음과 같은 방식들도 존재한다.
    - 활성화 함수 변경
    - 초기값 설정
    - learning rate 조정
  이 밖에 직접적으로 학습하는 과정을 전체적으로 안정화하자는 전략이 Batch Normalization이다. 정규화(normalization)을 사용하면 local minimum에 빠질 가능성을 낮출 수 있다.
  학습이 불안정하게 진행되는 이유를 internal Covariance Shift 문제라고 한다. 이것은 네트워크의 레이어나 활성화 함수마다 입력 값의 분산이 달라지는 현상이다.
  
  Batch란? 일반적인 gredient 기반의 방법은 모든 데이터를 확인하고 업데이트를 한번 진행하는데, 큰 데이터셋일 경우 메모리부족으로 처리를 못함. 따라서 batch라는 데이터를 나눠서 학습하고 가중치를 업데이트를 하게 됨. (Batch에는 묶은 단위라는 뜻도 있고 전체 데이터를 Batch라고 하는 뜻도 있는 것인가? 표현이 애매하다.)
  
  Batch Gredient Descent이란? 전체 데이터를 학습하고 가중치를 업데이트
  
  Stochastic Gredient Descent 이란? 전체 데이터를 학습하고 가중치를 업데이트하는게 아니라 입력 데이터를 하나씩 학습하고 가중치를 업데이트 하는 방식
  
  Mini-Batch란? 모든 데이터에 대해 가중치 평균을 구하는 것이 아니라 전체 데이터에서 일부 데이터를 묶어서 데이터를 학습하고 가중치를 업데이트 하는 방식

  해당 현상을 막기위해 whitening 이라는 방법이 있는데, 이것은 각 레이어의 분산을 0, 표준편차가 1인 값으로 정규화하는 방법이 있다. 하지만 이 방법은 covariance matrix 계산과 inverse의 계산이 필요하기 때문에 계산량이 많고, 일부 파라미터의 영향을 무시한다. 단순히 whitening을 사용하면 최적의 파라미터를 찾기위한 역전파와 무관하게 특정 값이 커질 수 있다.

  whitening의 문제를 해결한것이 바로 batch normalization(BN)
  
  ![image](https://user-images.githubusercontent.com/28583606/194516665-19ad14aa-9483-4090-ba21-ef22f0afcdf3.png)
  (출처: https://gaussian37.github.io/dl-concept-batchnorm/ 블로그 참조)
  
  
  위의 그림처럼 각 batch 마다 평균과 분산을 조정하여 분포를 조정하고 학습 과정을 안정화하는 것을 batch normalization 방법이다.
  일반 데이터는 뉴런 별로 일어나지만 Convolution에서는 채널별로 정규화가 일어난다.
  이때 batch의 크기는 너무 작아도, 커서도 안된다. 이것이 BN의 단점이다. 그래서 batch 별이 아닌 layer, weight, group, instance 별로 정규화하는 방법들이 있다.
  
  ii) Layer Normalization(LN)
  동일한 레이어의 뉴런들 간의 정규화, CNN으로 분류 문제를 해결할 경우 BN보다 오히려 성능이 떨어진다. RNN에서 오히려 효과가 있음
  
  iii) Instance Normalization(IN)
    BN은 모델의 학습에서만 사용가능하고 추론 및 테스트의 경우에는 사용할 수 없다. IN은 주로 이미지 스타일 변환에 사용되며 BN과 다르게 이미지 한장씩만 계산하여 각 이미지 분포를 사용한다. 또한 입력 이미지의 명암에 영향을 받지 않는 이미지 분류에 사용. (단, IN을 사용한다고해서 더 좋은 결과를 얻을 수 있는 것은 아님)
    
  iv) Group Normalization(GN)
  Batch 의 크기가 매우 작을때 효과적임. IN과 매우 유사한데 이미지의 채널들을 그룹으로 묶어서 평균과 표준편차를 구한다. IN은 이미지 채널이 하나로 묶여 있는 것이고, GN은 각각의 채널이 하나로 묶여 있는 것을 의미
  
  v) Weight Normalization(WN)
  WN은 mini-batch를 정규화하는 것이 아니라 layer의 가중치들을 정규화 한다. 

## Model Training
## Model Testing
## Model Deploy
