# GoogleNet
(https://techblog-history-younghunjo1.tistory.com/274?category=1031745 블로그 참조)

1 by 1 convoluion 을 최초로 도입한 GoogleNet(또는 inception network) 해당 모델은 pretrained 모델로 자주 사용된다고 함.
GoogleNet은 inception 모듈을 아래 그림처럼 여러번 이어 붙인 모델

![image](https://user-images.githubusercontent.com/28583606/195258482-90648c8d-6146-4131-8c0e-c4fed31a9f86.png)

GoogleNet의 주요 특징은 모델의 depth를 늘려도 연산장이 증가하지 않고 유지된다. 

## 1 by 1 Convolution

![image](https://user-images.githubusercontent.com/28583606/195276210-debe075c-f237-4111-b9e1-3b3219071577.png)


feature map의 dimension = (input size - (2* padding size) - filter size)/stride + 1

number of parameters in 1 layer = filter size로 계산 = (filter width * filter height * filter depth + 1 (for bias)) * number of channel

1 by 1 convoluntion을 사용하면 이전 레이어의 사이즈를 유지하면서 채널의 수를 줄이거나 늘릴 수 있다. 보통 채널의 수를 줄이는 데에 사용함(압축)
따라서, 채널 개수를 줄이면서 특징들을 압축시켜 어느 정도의 데이터 손실이 있지만 다양한 특징 추출이 가능하며, 파라미터의 갯수를 매우 줄일 수 있다.
보통 입력 데이터에 바로 사용하지 않고 어느 정도 convolution을 거친 후 계산된 feature map에 1 by 1을 사용한다.
그리고 적용하려던 convolution이 3 by 3이었으면 사용하기전에 1 by 1으로 계산해서 나온 feature map에 3 by 3 convolution을 적용하는하는 방식으로 사용
즉, 1 by 1 conv.는 상대적으로 high한 출력층에 가까운 레이어에 사용하는 것이 좋고, 입력 층과 가까운 low한 layer는 단순 convolution를 사용

중간 중간에 classifier가 존재하는데 GoogleNet이 deep model 이기 때문에 vanishing gredient 문제가 발생할 수 있다. 이를 완화해주기 위해서 classifier를 추가하고 특정 깊이마다
역전파를 진행해줌으로써 문제를 해결한다.

※ 그렇다면 2D 이미지가 아니라 1D 데이터에서도 사용할 수 있지 않을까?


## Inception Module

![image](https://user-images.githubusercontent.com/28583606/195282275-18085aac-45af-45d5-b10c-361a1130ece5.png)

![image](https://user-images.githubusercontent.com/28583606/195289408-754c7358-35ff-4277-b8fc-487172603f05.png)

GoogleNet의 inception module은 위 그림과 같다. 마지막에 concatenation 해야 하기때문에 output size는 input size와 동일해야하므로 padding을 same으로 설정해야함

※ Q. 왜 inception module은 4개의 feature map을 생성하고 concatenate 했을까?
A. 이미지에서 어떤 형상은 좀 더 큰 필터를 통과해야한다. 예를 들어 얼굴 이미지에서 1 x 1 필터로 눈을 인식하는 것 보다 더큰 n by n 필터로 눈을 인식하는것이 더 효과적이다. 그렇기 때문에 inception module에서는 1 x 1 convolution 말고 3 x 3 또는 5 x 5 의 convolution filter 연산을 병렬로 계산한다.(즉, 이건 모델 개발자의 마음이라는 뜻) 하지만 잘 생각해보아야할 것은 5 x 5 convolution filter 을 사용한다는 것은 그만큼 파라미터의 개수도 증가한다는 뜻이다. 그렇기 때문에 1 by 1 convolution으로 파라미터의 수를 줄여 효과적인 학습이 일어나도록 했다. 또한 concatenate를 한 이유는 각 feature map은 sparse structure일 수 있기 때문에 concatenate 함으로써 상대적으로 dense한 submatrix를 생성한다.

