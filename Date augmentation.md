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
