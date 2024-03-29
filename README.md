# AI-Study
개인 공부 저장

## Data pipeline
데이터를 운반하는 효율적인 입력 파이프라인이 필요. 보통 tf.data.Dataset을 상속해서 클래스를 정의하는 듯
데이터를 적재, 전처리, 셔플 하는 과정을 병렬적으로 수행한다면 전체적인 시간을 줄일 수 있고 컴퓨팅 리소스를 최대한 활용할 수 있음
https://ahnjg.tistory.com/32 해당 블로그에 자세하게 설명되어 있음

"과제 전형 및 kaggle을 해보면서 데이터의 EDR이 중요하다는 것을 깨달았음.
Feature Engineering도 중요함.."

### Data Loading
  df = pd.read_csv()를 사용하여 .csv 파일들을 읽지만 기타 확장자들(영상이나 의료 분야에서 사용하는 CT 결과)의 경우 다른 방식으로 데이터를 로딩함.
  index를 읽어올 수도 있는데 이때 index=False로 설정하면 index를 읽어오지 않음
  
  1. df.loc()
  2. df.iloc()
### Data Preprocessing
1. 데이터 확인
  보통 pandas를 자주 사용하는데 DASK라는 라이브러리도 눈여겨 볼만한듯
  df.info(): 데이터프레임의 row의 개수, 데이터의 타입 등을 확인 가능
  df.describe(): 문자열 데이터나 수치형 데이터를 따로 보여주는 듯. 수치형 데이터의 경우 분산,평균,최소,최대,중앙 값등을 계산해서 보여준다.
  df.head: 데이터프레임의 위에서 몇개이 데이터를 확인가능
2. 데이터 분포 확인 및 처리
  왜도와 척도
3. 데이터 이상치 확인 및 처리
  seaborn.boxplot()을 사용하거나 matlab으로 IQR를 직접 구해서 확인
4. 데이터 결측치 확인 및 처리
  isnull().sum()으로 확인할 수 있으며 결측치를 채우는 방법에는 다음과 같은 방법들이 있음
  1) 결측치가 있는 레코드를 삭제
  2) 해당 열에 평균값으로 채우기
  3) 해당 열에서 가장 많은 데이터 또는 가장 많은 값의 분포의 데이터로 채우기
  4) 결측치가 없는 데이터를 학습해서 ML로 채우기

  데이터를 채우는 방법은 df.[instance name].fillna()로 하거나 다른 방법 사용
5. 중복된 데이터 확인 및 처리
  duplicated().sum()
6. 데이터 정규화
  - Standardization: 값의 scale을 평균 0, 분산 1로 표준정규분포를 따르도록 변경, 이상치를 확인하기에 좋음
    1) scikit-learn에 StandardScaler
  - Normalization: 데이터의 scale을 0~1 사이의 값으로 변환하는 것 --> 특정 값이 크거나 너무 작으면 학습시 weight 조정에 영향을 주기 때문에 변환함
    1) Min-Max Scaling: 각 feature 마다 0~1사이의 값으로 정규화하는데 이상치에 굉장히 민감하다
    2) MaxAbs Scaling: MaxAbs는 -1 ~ 1의 값으로 데이터를 정규화함 이것도 마찬가지로 이상치에 민감한 편
    3) Robust Scaling: IQR(사분위수)를 이용하여 정규화를 진행하는데 이상치가 많은 데이터에 효과적임. 사분위수는 데이터의 분포에서 신뢰구간을 구한 후 25% ~ 75%의 데이터를 안정적인 데이터로 취급한다. 아래 그림 참고
       ![image](https://user-images.githubusercontent.com/28583606/198941849-a788aacc-e71e-4a08-b37b-9d7236852dd3.png)
        
  - Regulariztion: 위의 방법들과 다르게 Regularization은 weight가 너무 커지거나 작아지는 것을 방지하기 위한 방법 L1 정규화와 L2 정규화가 있음 / 따로 문서 만들기
### Data Shuffle
## Model Create



https://inhovation97.tistory.com/32 --> Batch size와 learning rate의 관련한 논문을 정리 및 실험한 블로그
- Scikit-learn
1. GradientBoosting



- 이미지 분야
1. CycleGAN
2. GoogleNet
3. ResNet


## Model Training

1) Data augmentation: 이미지를 변형하여 CNN의 성능을 향상, 즉 이미지의 레이블을 변경하지 않고 픽셀을 변형한다. 매우 보편적으로 사용하는 방법
## Model Testing
1. GridSearchCV: 파라미터(모델이나 기타 하이퍼파라미터 등)를 dictionary 형태로 만들어서 넣어주면 파라미터의 설정대로 모든 경우를 학습해준다. 결과를 _best_params 이나 _best_estimator 등을 통해 얻을 수 있음
## Model Deploy
### Kubernetes
컨테이너 기반의 오케스트레이션 툴
컨테이너 런타임(docker, crio, containerd 등)은 k8s 최신버전에서 docker가 depricated됨
### KubeFlow
### jenkins
