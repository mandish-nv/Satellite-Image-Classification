# Satellite Image Classification (SIC) 

## ABSTRACT

Satellite image classification plays a crucial role in remote sensing applications, including land cover mapping, environmental monitoring, and urban planning. Traditional classification methods often struggle with complex patterns and variations in satellite imagery. This project leverages deep learning, specifically **Convolutional Neural Networks (CNNs)**, to classify satellite images into different land cover categories such as urban, agriculture, forest, and water.

The project employs a CNN-based model, with an emphasis on **Transfer Learning** using advanced architectures like ResNet50V2. The model is trained on a labeled dataset of satellite images, optimizing it using data augmentation, batch normalization, and dropout techniques to enhance generalization and prevent overfitting. The model is evaluated using precision, recall, F1-score, and a confusion matrix to measure classification accuracy.

The results demonstrate that deep learning-based models outperform traditional methods, providing a robust framework for automating land cover classification with improved precision and efficiency.

## Project Structure

The repository contains the following key files:

* **`SIC.ipynb`**: The foundational notebook demonstrating Satellite Image Classification using a standard Convolutional Neural Network (CNN) architecture.
* **`SIC_ResNet.ipynb`**: The advanced notebook implementing Satellite Image Classification using **ResNet Transfer Learning** (specifically ResNet50V2) for improved performance.
* **`app.py`**: A Streamlit application (`streamlit app`) for demonstrating the model's predictions in a user-friendly web interface.
* **`requirements.txt`**: List of Python dependencies required to run the project.
* **`README.md`**: This project overview file.

## Technical Details

### Classification Categories
The models are trained to classify satellite images into the following land cover categories (based on the provided dataset structure):
* urban
* agriculture
* forest
* water

### Model Architecture
The primary model explored in `SIC_ResNet.ipynb` is based on **ResNet50V2** (pre-trained on 'imagenet') with custom classification layers added. The input image size is configured to be **(75, 75, 3)**.

### Key Libraries and Techniques
* Deep Learning Framework: TensorFlow/Keras
* Transfer Learning: VGG16, Xception, InceptionResNetV2, ResNet50V2 are utilized or imported.
* Data Handling: ImageDataGenerator is used for data loading and augmentation.
* Evaluation: `classification_report` and `confusion_matrix` from scikit-learn are used for evaluation.

## Datasets

The project utilizes satellite image datasets available at the following sources:

* [Satellite Image Classification (Zenodo)](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
* [Satellite Image Classification (Kaggle)](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)

Please ensure the data is downloaded and organized in a directory named `data` (as referenced in the notebooks) to run the scripts successfully. The directory structure is expected to contain subfolders for each category (e.g., `data/cloudy`, `data/water`).

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. Install all required packages using the generated `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebooks:**
    Execute the steps in `SIC.ipynb` or `SIC_ResNet.ipynb` to train the models and analyze performance.

4.  **Run the Streamlit App:**
    To launch the interactive classification application:
    ```bash
    streamlit run app.py
    ```
