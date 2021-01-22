
from smilemodel import SmileModel

sm = SmileModel(images_path)
sm.train(100) #100 : 학습횟수

array = sm.getImageArray("분류할 이미지 경로")

print(sm.predict(array))