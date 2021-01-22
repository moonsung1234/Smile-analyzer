
from imaged import ImageDownloader
from imageh import ImageHandler
from imagecnn import ImageCnnModel
import random
import os

#images_path = "C:/Users/muns3/OneDrive/Desktop/python-project/learning-program/smile_analyzer/images"

class SmileModel :
    def __init__(self, image_path) :
        self.image_path = image_path
        self.Ihr = ImageHandler()

    def getImages(self, count) :
        self.Idr = ImageDownloader(["smile", "sad face"], self.image_path, scroll_full_count=count, scroll_count=1000)
        self.Idr.saveImages()

    def __setData(self, loop_count) :
        self.smile_images_list = self.Ihr.getInImageList(os.path.join(self.image_path, "smile"))
        self.sad_images_list = self.Ihr.getInImageList(os.path.join(self.image_path, "sad face"))

        self.images_array_x = []
        self.images_array_t = []

        for i in range(loop_count) :
            random_index = random.randint(0, loop_count - 1)
            choose_int = random.randint(0, 1) 

            if choose_int == 1 :
                self.images_array_x.append(self.Ihr.getMonochrome(self.smile_images_list[random_index]).reshape(120**2))
                self.images_array_t.append([1, 0])

            else :
                self.images_array_x.append(self.Ihr.getMonochrome(self.sad_images_list[random_index]).reshape(120**2))
                self.images_array_t.append([0, 1])

    def getImageArray(self, image_path) :
        return [self.Ihr.getMonochrome(image_path).reshape(120**2)]

    def train(self, train_count) :
        self.__setData(2900)

        self.Inn = ImageCnnModel(0.001, train_count, 100, 120**2, 2)
        self.Inn.train(self.images_array_x, self.images_array_t)

    def predict(self, x_data) :
        try :
            self.Inn

        except :
            self.__setData(10)
            self.Inn = ImageCnnModel(0.001, 0, 100, 120**2, 2)
            self.Inn.train(self.images_array_x, self.images_array_t)

        predicted_value = self.Inn.predict(x_data)
        print(predicted_value)
        predicted_value = [round(predicted_value[0, 0]), round(predicted_value[0, 1])]

        return predicted_value

    def getAccuracy(self, accuracy_count, index_range) :
        self.__setData(index_range)

        try :
            self.Inn

        except :
            self.Inn = ImageCnnModel(0.001, 0, 100, 120**2, 2)
            self.Inn.train(self.images_array_x, self.images_array_t)

        acc_count = 0

        for i in range(accuracy_count) :
            random_index = random.randint(0, index_range - 1)
            choose_int = random.randint(0, 1)
            
            if choose_int == 1 :
                predicted_value = self.Inn.predict(self.Ihr.getMonochrome(self.smile_images_list[random_index]).reshape(1, 120**2))
                predicted_value = [round(predicted_value[0, 0]), round(predicted_value[0, 1])]

                if predicted_value[0] == 1 :
                    acc_count += 1

            else :
                predicted_value = self.Inn.predict(self.Ihr.getMonochrome(self.sad_images_list[random_index]).reshape(1, 120**2))
                predicted_value = [round(predicted_value[0, 0]), round(predicted_value[0, 1])]

                if predicted_value[1] == 1 :
                    acc_count += 1

        return acc_count / accuracy_count