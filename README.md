# Eye-Disease-Prediction

## Overview

This project aims on predicting eye diseases from retinal images using a Convolutional Neural Network (CNN). It identifies four key conditions:

- **Diabetic Retinopathy**
- **Glaucoma**
- **Cataract**
- **Normal (Healthy Eyes)**

The model was trained on a dataset of 4217 labeled retinal images and evaluated on the test data.

The application is deployed using **Streamlit**, providing a simple, interactive interface where users can upload an image and receive a disease prediction in real-time.

This tool can support early detection of eye-related illnesses and is intended for educational and research purposes.

## Model Summary

The model is a CNN built with TensorFlow, structured as follows:

- 3 × Convolutional layers with ReLU activation
- MaxPooling after each convolution
- Flatten layer followed by Dense (128 units)
- Final output layer with softmax for 4-class classification
- Optimized using Adam and trained with sparse categorical cross-entropy loss

- For accessing the datacard/dataset:
🔗 [Eye Diseases Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data)


## Installation

To set up and run the application locally, follow these steps:

### 1. Install the required dependencies
```bash
pip install -r requirements.txt
````

### 2. Clone the repository

```bash
git clone https://github.com/tanishapd/Eye-Disease-Prediction.git
cd Eye-Disease-Prediction
```

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Troubleshooting

If you encounter issues while running the app, make sure all required dependencies are installed using `pip install -r requirements.txt`. Check that the `.joblib` model file is placed in the correct directory or update its path in the code. If image uploads fail, ensure the file is in `.jpg` or `.jpeg` format and is not corrupted. For deployment errors (e.g., on Render), confirm that all necessary files are pushed and the Python version is compatible. 







