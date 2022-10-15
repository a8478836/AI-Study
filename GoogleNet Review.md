#Going deeper with convolutions
Christian Szegedy(Google Inc.), et al.

## Abstract
본 논문에서는 코드네임 Inception 이라는 'Deep convolutional nerual network'를 소개한다. 해당 뉴럴 네트워크는 ImageNet Large-Scale Visual Recognition Challenge 2014(ILSVRC14)에서
분류와 탐지를 위한 새로운 아키텍처 또는 기술이다. 이 아키텍쳐에서 눈여겨 보아야할 점은 네트워크 내부에서 컴퓨팅 자원의 사용 효율을 향상시키는 것이다. 즉, 연산에 사용되는 자원을 유지하되
네트워크의 깊이과 너비를 크게 늘리자는 취지로 설계 및 구현했습니다. 모델의 성능을 최적화 하기 위해서 Hebbian priciple과 multi-scale processing 기법으로 아키텍처를 설계 했습니다.
ILSVRC14에 제출한 모델의 이름은 GoogleNet이며 22개의 레이어를 쌓은 모델입니다.


## Introduction
