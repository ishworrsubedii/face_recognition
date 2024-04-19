# OpenCV-Face-Recognition

Real-time face recognition project with OpenCV and Python

## How to install

1. Clone the repository

```angular2html
git clone repo_url
```

2. Create a virtual environment

```angular2html
conda create -n face_recognition python=3.10
```

```angular2html

3. Install the required packages

```angular2html
pip install requirements.txt
```

4. Run the code

This will collect the dataset of the person whose face you want to recognize using webcam.

```angular2html
cd examples
python face_dataset_collection_example.py
```

This will start training the model using the dataset collected using harcascade classifier.

```angular2html
python face_training_example.py
```

This is for inference purpose. It will recognize the face of the person using the webcam.

```angular2html
python face_recognition_example.py
```

## Contributors

Contributions are always welcome. Submit a PR or open an issue.

