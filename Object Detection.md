# Object Detection

(https://techblog-history-younghunjo1.tistory.com/178 블로그 참조)
(https://velog.io/@cha-suyeon/%EB%94%A5%EB%9F%AC%EB%8B%9D-Object-Detection-Localization-%EA%B0%9C%EB%85%90-%EC%A0%95%EB%A6%AC 블로그 참조)

이미지 프로세싱의 하위 분야, 보통 CNN은 이미지를 Classification 하는 데에 집중하지만 object detection은 classification과 localization을 합친 개념이다.
즉, 어떤 이미지를 입력 데이터로 넣었을 때 bounding box(localization)으로 객체를 찾고 해당 bounding box안에 있는 객체가 무엇인지(classification) 하는 것을 objection detection이라고 한다.

object recognition: detection과 비슷한 의미지만 dectection의 경우 객체의 존재 여부만 확인하지만 recognition은 객체의 존재여부 뿐만이 아니라 해석까지 하는 것
object segementation: 그림 2) 처럼 탐지된 object에 따라 영역을 표시하는 것
  1) semantic segmentation: 같은 클래스의 object 들은 같은 영역 또는 색으로 분할
  2) instance segmentation: semantic segmenation보다 자세하게 같은 클래스여도 서로 다른 instance를 구별
![image](https://user-images.githubusercontent.com/28583606/195809641-41d1cd9a-4096-4068-b409-88b2c688aa9c.png)
(그림 1)

![image](https://user-images.githubusercontent.com/28583606/195809276-7c8cd6f6-640e-44b0-948c-bf43ec914739.png)
(그림 2)





## 응용 분야
1) 자율 주행

## 성능 지표
1) FPS: Frame per second, 초당 dectection을 수행하는 비율
2) IoU: Intersection over Union,  localization 모델이 인식할 결과에 대한 평가 지표, boundion box와 Ground truth 영역을 나눈 값 이 값이 1일 경우 완전히 일치한다는 뜻
3) NMS: Nom-max suppression: 
4) mAP

## 방법
Region proposal을 수행하는 방법은 다음과 같다.
1) sliding window: window를 움직이면서 객체가 있을 만한 region들을 제안한다. 단점으로 객체가 없는 지역도 찾기 때문에 시간이 오래걸리고 객체를 탐지할 확률도 낮아진다.
2) selective search: 이미지 픽셀의 색상, 무늬, 크기, 형태에 따라 유사한 region을 그룹핑하는 방식이다.

Training Set에 class 뿐만이 아니라 bounding box의 좌표를 추가하면 supervised learning으로 detection이 가능

전체적인 Object dectection의 아키텍쳐는 다음과 같다.
![image](https://user-images.githubusercontent.com/28583606/195813563-a3e7e4d2-c9b8-4923-88fb-aa6595aded0f.png)

