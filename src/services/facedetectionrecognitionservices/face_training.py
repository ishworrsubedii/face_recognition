"""
This script is used to train the face recognizer

"""
import cv2
import numpy as np
from PIL import Image
import os


class FaceTrainer:
    def __init__(self, dataset_path, cascade_path, trainer_path):
        """
        Initialize the face recognizer
        :param dataset_path:  The path to the dataset which contains the images of the faces
        :param cascade_path: The path to the cascade file which is used to detect the faces
        :param trainer_path: The path to the trainer file which will be created after training
        """
        self.dataset_path = dataset_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.trainer_path = trainer_path

    def get_images_and_labels(self):
        """
        Get the images and labels from the dataset
        :return:  The images and labels
        """
        image_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
        face_samples = []
        ids = []
        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert('L')  # grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return face_samples, ids

    def train(self):
        """
        Train the recognizer
        :return:  None
        """
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = self.get_images_and_labels()
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write(self.trainer_path)
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

#
# if __name__ == "__main__":
#     dataset_path = 'resources/dataset'
#     trainer_path = 'resources/trainer/trainer.yml'
#     face_cascade_path = 'resources/FacialRecognition/haarcascade_frontalface_default.xml'
#
#     try:
#         face_trainer = FaceTrainer(dataset_path=dataset_path, cascade_path=face_cascade_path, trainer_path=trainer_path)
#         face_trainer.train()
#     except Exception as e:
#         print(f"An error occurred: {e}")
