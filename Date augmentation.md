# Data Augmentation
( https://hoya012.github.io/blog/Image-Data-Augmentation-Overview/ 블로그 참조)
데이터 전처리(data preprocessing)은 데이터 전체에 대해 공통적인 알고리즘으로 데이터에 변화를 주는 것이고
데이터 변조(data augmentation)은 데이터마다 매번 다른 알고리즘으로 데이터에 변화를 주는 것

## Pixel-Level Transforms

![image](https://user-images.githubusercontent.com/28583606/196022181-52e19639-7214-4da7-95fe-f241f0ab9dd2.png)


Pixel 단위로 변환하는 pixel-level transform은 대표적으로 Blur(흐리게), Jitter(), Noise(), 등이 있고
대표적인 방법으로는 gaussian blur, montion blur, brightness jitter, constrast jitter, saturation jitter, ISO noise, JPEG compression 등이 있다.

## Spatial-Level Transforms

![image](https://user-images.githubusercontent.com/28583606/196022954-599ab30b-8ad8-4051-bdab-7e80b6196048.png)

이미지 자체를 변환 Flip(뒤집기), Rotation(회전), Crop(일부 자르기), resize(크기 변환), translate(밀기) 등이 있다.
이때, Boundinb box, segementation task의 경우 이미지에 적용한 변환을 ground truth에도 동일하게 적용해줘야하고, classification의 경우 적용시 class가 바뀔 수 있다!
(예를 들어, +가를 45도 회전시 x가 되거나, 6이 9가 되는 등 ㅎ 의 윗 부분을 잘라버리면 ㅇ 이 되는 등...)

![image](https://user-images.githubusercontent.com/28583606/196023457-9ec365c2-5210-4a31-959c-726fbdd6809a.png)

(“Improved Mixed-Example Data Augmentation”, 2018)

이미지 mixing을 통해 데이터 변조

![image](https://user-images.githubusercontent.com/28583606/196023462-42822159-ea89-4de2-9fb2-b3c52481a084.png)

(“MixUp: Beyond Empirical Risk Minimization”, 2018)

두 이미지를 0~1 사이의 값을 통해 weighted linear interploation 하는 기법, beta distribution을 통해 lambda 값을 추출한다. 모델 성능도 좋아지고 잘못된 라벨 학습을 방지해주며
적대적인 예시에 sensitive 하는 등 다양한 효과를 얻을 수 있다.(유명한 방법이라고 함!)


![image](https://user-images.githubusercontent.com/28583606/196023562-37b1880d-147a-4841-8725-aec7bac69974.png)

(“Data augmentation using random image cropping and patches for deep CNNs”, 2018)

이미지를 random 하게 잘라서 붙이고 label을 면적에 따라 soft label을 만들 수 있다. 하지만 생성된 이미지에 객체가 등장하지 않음에도 객체에 대한 label이 부여될 수 있음
--> YOLO V4 모델의 mosic augmentation을 참고


![image](https://user-images.githubusercontent.com/28583606/196026166-4f581f45-3865-4751-a3b2-8f0af9f9aa3c.png)

(“CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features”, 2019)

한 이미지에서 특정 부분을 지우고 다른 이미지의 일부분을 잘라서 합치는 방식. label도 면적에 따라 조절하며 성능이 괜찮은 기법.

![image](https://user-images.githubusercontent.com/28583606/196026261-108ddacf-2a00-4df3-bcac-265d669f5bdb.png)

(“The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization”, 2020)

DeepAugment 기법은 이미 학습된 Image to Image network(Autoencoder 등)의 weight와 activation function을 변화하여 이미지를 변조하는 방법을 제안한다.
기존의 방법들로 생성하기 어려운 다양한 유형의 image를 생성할 수 있다.


## Generative Model based

![image](https://user-images.githubusercontent.com/28583606/196026514-0e78b3d6-ecb6-4677-80a0-d4e1f03131f0.png)

(“GAN-based Synthetic Medical Image Augmentation for increased CNN Performance in Liver Lesion Classification”, 2018)

DCGAN을 통해 생성한 Liver lesion image로 성능을 높인 논문

![image](https://user-images.githubusercontent.com/28583606/196026530-2a0ff2f7-f4ef-4ac6-a47d-309722c7cb12.png)

(“Data Augmentation in Emotion Classification Using Generative Adversarial Networks”, 2017)

CycleGAN으로 얼굴 감정 데이터를 생성하고 클래스 불균형을 완화

![image](https://user-images.githubusercontent.com/28583606/196026571-ff185220-b7a8-46a7-b7cb-0fe6f1115769.png)

(“SinGAN: Learning a Generative Model from a Single Natural Image”, 2019)

한장의 이미지로 GAN을 학습시켜서 다량의 image를 생성한다. 이미지 한장에 하나의 GAN을 학습시켜야하므로 학습 시간이 오래 걸린다는 한계가 존재한다.

## AutoML based

![image](https://user-images.githubusercontent.com/28583606/196026652-bf88aacd-15e1-4b3e-8782-ff2fd0267dd4.png)

(“AutoAugment: Learning Augmentation Policies from Data”, 2018)

RNN을 통해 augmentation policy를 뽑고, 네트워크를 학습한 후 validation accuracy를 뽑아 강화학습의 reward로 사용하여 학습하는 방법.
하지만 상당한 컴퓨팅 비용을 소모한다고 한다.

AutoML은 나중에 다시 확인해보자...

