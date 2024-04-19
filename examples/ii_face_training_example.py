"""
Created By: ishwor subedi
Date: 2024-03-21
"""
from src.services.facedetectionrecognitionservices.face_training import FaceTrainer

if __name__ == "__main__":
    dataset_path = 'resources/dataset'
    trainer_path = 'resources/trainer/trainer.yml'
    face_cascade_path = 'resources/haarcascade_frontalface_default.xml'

    try:
        face_trainer = FaceTrainer(dataset_path=dataset_path, cascade_path=face_cascade_path, trainer_path=trainer_path)
        face_trainer.train()
    except Exception as e:
        print(f"An error occurred: {e}")
