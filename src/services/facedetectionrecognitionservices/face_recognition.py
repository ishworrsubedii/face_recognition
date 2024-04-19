import numpy as np
import os
import cv2


class FaceRecognizer:
    def __init__(self, names, source, trainer_path, cascade_path):
        """
        Initialize the face recognizer
        :param names: This includes the names of the people in the dataset
        :param source: source of the video that will be used for recognition
        :param trainer_path: path to the trainer file
        :param cascade_path: path to the cascade file
        """
        self.names = names
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(trainer_path)
        cascadePath = cascade_path
        self.faceCascade = cv2.CascadeClassifier(cascadePath)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cam = cv2.VideoCapture(source)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height
        self.minW = 0.1 * self.cam.get(3)
        self.minH = 0.1 * self.cam.get(4)

    def recognize_faces(self, threshold):
        """
        This method recognizes faces from the video source and displays the recognized faces on the screen
        :param threshold: threshold value for recognition
        :return: None
        """
        while True:
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(self.minW), int(self.minH))
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                confidence = int("  {0}".format(round(100 - confidence)))

                print(f"ID: {id}, Confidence: {confidence}")
                if confidence is not None and confidence > threshold:
                    id = self.names[id]
                else:
                    id = "unknown"
                cv2.putText(img, str(id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)
            cv2.imshow('camera', img)
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

    def cleanup(self):
        """
        This method cleans up the resources used by the program
        :return: None
        """
        print("\n [INFO] Exiting Program and cleanup stuff")
        self.cam.release()
        cv2.destroyAllWindows()

    def read_names_from_file(file_path):
        """
        Read names from file
        :return: list of names
        """
        with open(file_path, 'r') as file:
            names = file.readlines()
        names = [name.strip() for name in names]
        return names

# if __name__ == "__main__":
#     names_path = 'resources/names.txt'
#     source = 0  # for webcam
#     # source = "rtsp://"
#     trainer_path = 'resources/trainer/trainer.yml'
#     cascade_path = 'resources/haarcascade_frontalface_default.xml'
#     threshold_for_recognition = 40
#     names = FaceRecognizer.read_names_from_file(names_path)
#     print(names)
#     try:
#         face_recognizer = FaceRecognizer(names, source=source, trainer_path=trainer_path,
#                                          cascade_path=cascade_path)
#         face_recognizer.recognize_faces(threshold=threshold_for_recognition)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         face_recognizer.cleanup()
