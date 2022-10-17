# Going deeper with convolutions

Christian Szegedy(Google Inc.), et al.

## Abstract
본 논문에서는 코드네임 Inception 이라는 'Deep convolutional nerual network'를 소개합니다. 해당 뉴럴 네트워크는 ImageNet Large-Scale Visual Recognition Challenge 2014(ILSVRC14)에서
분류와 탐지를 위한 새로운 아키텍처 또는 기술입니다. 이 아키텍쳐에서 눈여겨 보아야할 점은 네트워크 내부에서 컴퓨팅 자원의 사용 효율을 향상시키는 것입니다. 즉, 연산에 사용되는 자원을 유지하되
네트워크의 깊이과 너비를 크게 늘리자는 취지로 설계 및 구현했습니다. 모델의 성능을 최적화 하기 위해서 Hebbian priciple과 multi-scale processing 기법으로 아키텍처를 설계 했습니다.
ILSVRC14에 제출한 모델의 이름은 GoogleNet이며 22개의 레이어를 쌓은 모델입니다.


## Introduction
지난 3년간 Covoluntional network를 통해 이미지 인식과 객체 탐지 분야가 크게 성장했습니다. 더욱 놀라운 것은 이러한 현상이 우수한 성능의 하드웨어나 큰 데이타셋 또는 거대 모델이 아니라 새로운 아이디어를 기반한 알고리즘 덕분이라는 것입니다. 특히 객체 탐지에서는 R-CNN과 같은 고전적인 컴퓨티 비전 알고리즘과 심층 신경망의 융합 덕분입니다. 그리고 모바일과 임베디드 컴퓨팅에서 전력 및 메모리를 얼마큼 사용하는지 중요한데, 우리의 알고리즘은 자원을 사용함에 있어서 아주 적합한 알고리즘입니다. 실험에서 우리의 모델은 큰 규모의 데이터셋을 사용하더라도 추론 단계에서 15억번의 곱셈 계산을 유지했기 때문에 실응용에도 무리가 없습니다.
이 논문에서는 컴퓨터 비전에서 효율적인 딥 뉴럴 네트워크 아키텍처가 주제이며, 인터넷 밈인 'we need to go deeper'에서 기인했습니다. 'Deep'에는 두 개의 의미가 있습니다. 첫번 째는 Inception Module이라는 형태의 새로운 레벨의 집합을 소개하고, 다른 하나는 실제로 네트워크의 깊이를 깊게 설계 했습니다. 우리 모델은 ILSVRC 2014에서 좋은 성능을 보여주었습니다.

## Related Work
LeNET5-5과 시작된 CNN은 전형적으로 convolutional layer를 여러개 쌓은 네트워크이다. 개별적으로 normalization과 pooling layer를 넣는 경우도 있고 이 다음에 fully-connected layer를 쌓는다. 이런 기본적인 설계에서 다양한 변형들은 MNIST, CIFAR와 같은 유명한 데이터셋을 사용한 이미지 분류 대회에서 최고의 성능을 냈다. ImageNet과 같은 더욱 거대한 네트워크에서 최근 동향은 layer의 수와 크기를 더욱 늘리면서 오버피팅을 예방하기 위해 dropout과 같은 방법을 사용한다.

하지만 max-pooling layer는 공간 정보를 잃어버림에도, 다양한 분야에서 성공적인 효과를 거둔 CNN이 있습니다. 영장류의 대뇌 피질에서 착안하여 개발된 inception 모델이 있는데, 고정된 Gabor 필터를 사용했지만 모든 필터를 학습합니다.(연산량이 많아진다는 것을 의미하는듯). GoogleNet은 22개의 깊은 네트워크 모델입니다.

뉴럴 네트워크의 성능을 끌어올리기 위해 Network-in-network라는 접근 방법이 있습니다. 이 방법은 Convolutional layer에서 ReLU를 적용한 1 by 1 con.를 사용했다. 무거운 GoogleNet에 이 아이디어를 적용했습니다. 1 by 1 conv.를 적용한 이유는 연산 bottleneck을 줄이기 위해 차원 감소 효과와 네트워크 크기에 제한을 주기 위해서입니다. 이 방법으로 인해 모델의 깊이가 깊어지고 넓이 또한 커지더라도 아무런 영향없이 효과적인 성능을 보여줍니다.

현재 object detection에서 가장 우수한 성능을 보여주는 R-CNN은 전체 문제를 2개의 부분 문제로 분해하여 해결합니다. (low-level cue와 superpixel를 이용하고 CNN을 적용한다는 뜻인듯. 즉 2-stage learning) 본 논문에서도 비슷한 파이프라인을 적용했지만, bounding box recall을 향상시키기 위한 multi-box 바법과 bounding box의 더 나은 분류 성능을 위해 앙상블 방법 등을 적용했습니다.

## Motivation and High Level Considerations

대부분의 일직선-전방향 방법의 DNN은 깊이를 깊게하고 유닛의 수를 증가시켜서 성능을 향상시켰으며 이 방법은 가장 간단한 모델의 성능 방법이었다. 하지만 이 방법은 모델이 커지면 파라미터의 수가 증가하고 만약 학습 데이터가 제한되어 있으면 오버 피팅이 일어난다.

## Architectural Details
