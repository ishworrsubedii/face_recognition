"""
This script captures the faces of the user and saves them to the dataset folder

"""
import cv2
import os


class FaceCapturer:
    def __init__(self, cascade_file_path, source):
        """
        Initialize the camera and the face detector
        :param cascade_file_path: path to the cascade file
        :param source: source of the video that will be used to capture the faces
        """
        self.cam = cv2.VideoCapture(source)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height
        self.face_detector = cv2.CascadeClassifier(cascade_file_path)

    def capture_faces(self, face_id, face_name, output_folder_path):
        """
        Capture faces from the camera and save them to the dataset folder
        :param face_id:  The id of the face
        :param face_name:  The name of the face
        :param output_folder_path: The path to the folder where the faces will be saved
        :return:
        """
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        count = 0
        while True:
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite(f"{output_folder_path}/{face_name}.{str(face_id)}.{str(count)}.jpg", gray[y:y + h, x:x + w])
                cv2.imshow('image', img)
            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break

    def cleanup(self):
        """
        Clean up the camera and close the window
        :return:
        """
        print("\n [INFO] Exiting Program and cleanup stuff")
        self.cam.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     cascade_path = 'resources/haarcascade_frontalface_default.xml'
#     source = 0  # for webcam
#     # source = "rtsp://"
#     try:
#         face_capturer = FaceCapturer(cascade_file_path=cascade_path, source=source)
#         face_id = input('\n enter user id end press <return> ==>  ')
#         face_name = input('\n enter user name end press <return> ==>  ')
#         face_capturer.capture_faces(face_id, face_name)
#
#         with open('resources/names.txt', 'a') as f:
#             f.write(f"{face_name} \n")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         face_capturer.cleanup()
