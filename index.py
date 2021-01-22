
from imaged import ImageDownloader
from imageh import ImageHandler
from imagecnn import ImageCnnModel
import random
import os

images_path = "C:/Users/muns3/OneDrive/Desktop/python-project/learning-program/smile_analyzer/images"

#ImageDownloader section
Idr = ImageDownloader(["smile", "sad face"], images_path, scroll_full_count=50, scroll_count=1000)
#Idr.saveImages()

#ImageHandler section
Ihr = ImageHandler()
smile_images_list = Ihr.getInImageList(os.path.join(images_path, "smile"))
sad_images_list = Ihr.getInImageList(os.path.join(images_path, "sad face"))

#ImageCnnModel secion

images_array_x = []
images_array_t = []

for i in range(2000) :
    random_index = random.randint(0, 1999)
    choose_int = random.randint(0, 1) 

    if choose_int == 1 :
        images_array_x.append(Ihr.getMonochrome(smile_images_list[random_index]).reshape(120**2))
        images_array_t.append([1, 0])

    else :
        images_array_x.append(Ihr.getMonochrome(sad_images_list[random_index]).reshape(120**2))
        images_array_t.append([0, 1])

Inn = ImageCnnModel(0.001, 0, 100, 120**2, 2)
Inn.train(images_array_x, images_array_t)
predicted_value = Inn.predict(Ihr.getMonochrome(smile_images_list[0]).reshape(1, 120**2))

print("predicted value : ", predicted_value)