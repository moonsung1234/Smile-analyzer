<h1>Smile Analyzer</h1>

<br/>

- cnn 알고리즘을 사용하여 얼굴이 smile 인지 sad 인지 구분하는 딥러닝 모델.

```python
from smilemodel import SmileModel

sm = SmileModel("학습할 이미지를 넣을 폴더경로")
sm.train(100) # 100 : 학습횟수

array = sm.getImageArray("분류할 이미지 경로")

print(sm.predict(array))
```

<br/>

- 학습 이미지는 smile 이미지, sad 이미지 각각 8700개, 2900개씩 기본적으로 들어있음.
- 학습 이미지의 개수에 따라 모델의 정확성이 달라질 수 있음.

```python
from smilemodel import SmileModel

# getImages : 학습 이미지를 다운받는 함수

sm = SmileModel("학습할 이미지를 넣을 폴더경로")
sm.getImages(스크롤 카운트) # 스크롤 카운트 : 총 스크롤할 횟수를 정의함. (기본 50)
```

<br/>

- 기본적으로 들어있는 학습 이미지로 학습을 진행한 결과, 정확도가 최소 80% 후반에서 최대 90% 후반까지 나오는걸 확인할 수 있음.
- 이미지의 개수에 따라 정확도가 달라질 수 있음.

```python
from smilemodel import SmileModel

# getAccuracy : 정확도를 측정하는 함수

sm = SmileModel("학습할 이미지를 넣을 폴더경로")
sm.getAccuracy(측정할 이미지 개수, 이미지 범위) # 개수 : 정확도를 측정할 이미지 개수, #범위 : 이미지를 가져올 인덱스 범위 (기본 100, 2900)
```
