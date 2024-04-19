"""
Created By: ishwor subedi
Date: 2024-03-21
"""

from src.services.facedetectionrecognitionservices.face_dataset import FaceCapturer

if __name__ == "__main__":
    cascade_path = 'resources/haarcascade_frontalface_default.xml'
    # source = 0
    source = 0  # for webcam
    output_folder_path = 'resources/dataset'
    try:
        face_capturer = FaceCapturer(cascade_file_path=cascade_path, source=source)
        face_id = input('\n enter user id end press <return> ==>  ')
        face_name = input('\n enter user name end press <return> ==>  ')
        face_capturer.capture_faces(face_id, face_name, output_folder_path=output_folder_path)

        with open('resources/names.txt', 'a') as f:
            f.write(f"{face_name} \n")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        face_capturer.cleanup()
