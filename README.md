# Face Mask Detection using Computer Vision and Deep Learning

This project implements a face mask detection system using computer vision techniques and a deep learning model trained on images of people wearing face masks and not wearing face masks.

## Project Overview

The project uses a Convolutional Neural Network (CNN) model to classify whether a person in a given image or video stream is wearing a mask or not. The model is trained on images that contain faces, and the face detection is performed using the Haar Cascade Classifier. The final system provides real-time mask detection in a webcam stream.

### Technologies Used:
- **Python**: The primary language used for implementing the project.
- **OpenCV**: For image processing, face detection, and real-time video streaming.
- **TensorFlow/Keras**: For deep learning model building, training, and prediction.

### Key Features:
- Face detection using Haar Cascade Classifier.
- Real-time face mask detection using a pre-trained deep learning model.
- Webcam integration to detect whether people in the video stream are wearing face masks or not.

## Setup and Installation

Follow these steps to set up and run the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/NeelanshaAgarwal/Face-Mask-Detection.git
cd Face-Mask-Detection
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

- *Windows:*

  ```bash
  .\venv\Scripts\activate
  ```

- *Linux/Mac:*

  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

**requirements.txt:**

  - `opencv-python`
  - `tensorflow`
  - `keras`
  - `numpy`
  - `matplotlib` (for visualization)
  - `scikit-learn` (optional for model evaluation)

### 4. Download the Haar Cascade XML file

The project requires the Haar Cascade XML file for face detection. You can download it from:

- [Haar Cascade for Face Detection](https://github.com/opencv/opencv/tree/master/data/haarcascades)

Place the `haarcascade_frontalface_default.xml` file in the project directory.

### 5. Train the model (if not already trained)

If you do not have the pre-trained model (`newsaved_model.h5`), you can train the model by running the training script. Make sure you have your training dataset ready.

### 6. Run the face mask detection script

To run the real-time face mask detection using your webcam:

```bash
python detect_mask.py
```

This will start the webcam and display a window with detected faces and their mask classification. Press `q` to exit.

### File Structure

The project folder contains the following structure:

```bash
Face-Mask-Detection/
│
├── input/                      # Temporary folder for storing input face images
├── models/                     # Folder to store the trained model (newsaved_model.h5)
├── haarcascade_frontalface_default.xml  # Haar cascade XML for face detection
├── detect_mask.py              # Main script to detect masks in real-time using webcam
├── train_model.py              # Script to train the mask detection model
├── README.md                   # Project overview and instructions
├── requirements.txt            # List of required Python packages
└── .gitignore                  # Git ignore file
```

### Troubleshooting

- Make sure the `haarcascade_frontalface_default.xml` file is in the correct directory.

### Contributing

Contributions are welcome! If you'd like to contribute, feel free to fork the project, make changes, and submit a pull request.

### License

This project is open-source and available under the MIT License.

### Acknowledgements

- The face detection algorithm is based on the Haar Cascade Classifier provided by OpenCV.
- The deep learning model is trained using TensorFlow/Keras.


### Notes:

- Ensure that the URLs in the **Clone the repository** section point to your GitHub repository.
- You may need to add or modify sections based on how you want to structure your project.
- Include any other dependencies you may have used during development in the `requirements.txt` file.

Let me know if you need further modifications or details!
