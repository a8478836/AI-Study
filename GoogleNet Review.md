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

## Motivation and High Level Considerations

## Architectural Details
