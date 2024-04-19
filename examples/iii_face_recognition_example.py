"""
Created By: ishwor subedi
Date: 2024-03-21
"""
from numpy import source

from src.services.facedetectionrecognitionservices.face_recognition import FaceRecognizer

if __name__ == "__main__":
    names_path = 'resources/names.txt'
    # source = 0  # for webcam
    source = "rtsp://ishwor:subedi@192.168.1.106:5555/h264_opus.sdp"
    trainer_path = 'resources/trainer/trainer.yml'
    cascade_path = 'resources/haarcascade_frontalface_default.xml'
    threshold_for_recognition = 30
    names = FaceRecognizer.read_names_from_file(names_path)
    print(names)
    try:
        face_recognizer = FaceRecognizer(names, source=source, trainer_path=trainer_path,
                                         cascade_path=cascade_path)
        face_recognizer.recognize_faces(threshold=threshold_for_recognition)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        face_recognizer.cleanup()
