# **🐈Pet Classifier: A Deep Learning Model for Pet Breed Identification**



## **📖 Project Overview**

The Pet Classifier is a deep learning project designed to classify images of pets into their respective breeds. Built using Python, TensorFlow/Keras, and Jupyter Notebook, this project leverages convolutional neural networks (CNNs) to achieve high-accuracy image classification. The model is trained on the Oxford-IIIT Pet Dataset and includes data preprocessing, model training, evaluation, and visualization of performance metrics.

This project is ideal for machine learning enthusiasts, pet lovers, and developers interested in computer vision applications. The codebase is well-documented, easy to understand, and ready for further customization or deployment.



## **🚀 Features**



Multi-Class Classification: Identifies various pet breeds with high accuracy.

Data Preprocessing: Efficiently processes and organizes image data using Python libraries like numpy, pandas, and matplotlib.

CNN Model: Utilizes a convolutional neural network implemented in TensorFlow/Keras for robust image classification.

Performance Visualization: Includes plots for training and validation accuracy/loss to evaluate model performance.

Scalable Design: Easily extendable to include more pet breeds or additional datasets.

Comprehensive Evaluation: Provides detailed metrics such as accuracy and loss on the test set.





## **📂 Dataset**

The project uses the Oxford-IIIT Pet Dataset, which contains images of various pet breeds, primarily cats and dogs. The dataset is downloaded using the kagglehub library and stored in the ./images directory, categorized by breed. Each image is labeled with the breed name and a unique identifier (e.g., breed\_name\_1.jpg). The dataset is processed to extract image names and prepare them for model training.



#### 🛠️ Installation

Follow these steps to set up the project on your local machine:

Prerequisites



Python 3.11.5 or higher

Jupyter Notebook

A working installation of pip for package management

A Kaggle account and API token (for dataset download)



#### Steps



Clone the Repository:

git clone (https://github.com/mdowais-39/Pet-Classification)

cd pet-classifier





Set Up a Virtual Environment (recommended)





Install Dependencies:

Install the required Python packages using the provided requirements.txt file:

pip install -r requirements.txt





Set Up Kaggle API:

To download the dataset, you need a Kaggle API token:



Go to your Kaggle account settings and create a new API token. This will download a kaggle.json file.





Download the Dataset:

Use the kagglehub library to download the Oxford-IIIT Pet Dataset. In a Jupyter Notebook or Python script, run the necessary commands to download the dataset tanlikesmath/the-oxfordiiit-pet-dataset. The dataset will be stored in a cache directory (e.g., ~/.cache/kagglehub/datasets/tanlikesmath/the-oxfordiiit-pet-dataset/versions/1).



Prepare the Dataset:

Move or link the downloaded images to the ./images directory, ensuring they are organized by breed.



Launch Jupyter Notebook:

Start Jupyter Notebook to run the Pet\_classifier.ipynb file:

jupyter notebook





## **📈 Usage**



Open the Notebook:

Open Pet\_classifier.ipynb in Jupyter Notebook.



Run the Cells:

Execute the cells sequentially to:



Import necessary libraries (numpy, pandas, matplotlib, etc.).

Load and preprocess the image dataset from the ./images directory.

Train the CNN model using TensorFlow/Keras.

Evaluate the model on the test set.

Visualize training and validation accuracy/loss.





Model Evaluation:

The notebook includes code to evaluate the model on the test set (X\_test, y\_test), producing metrics such as:



Test Loss: 0.2769

Test Accuracy: 91.54%The evaluation results are displayed after running the model.evaluate(X\_test, y\_test) cell.





Predictions:

Use the model.predict(X\_test) cell to generate predictions (y\_pred) for the test set. The output is an array of probabilities for each class.



Visualizations:

The notebook generates two plots:



Training and Validation Accuracy: Compares the model's accuracy on training and validation sets over 10 epochs.

Training and Validation Loss: Shows the loss trends for training and validation sets.









## **📊 Results**

The model achieves a test accuracy of 91.54% and a test loss of 0.2769, indicating strong performance in classifying pet breeds. The training and validation accuracy/loss plots provide insights into the model's learning process, helping identify potential overfitting or underfitting.





## **🧠 Model Architecture**

The pet classifier uses a convolutional neural network (CNN) implemented in TensorFlow/Keras. While the exact architecture is not fully detailed in the provided notebook snippet, it typically includes:



Convolutional Layers: For feature extraction from images.

Pooling Layers: To reduce spatial dimensions while preserving important features.

Dense Layers: For classification based on extracted features.

Softmax Output: To produce probabilities for each pet breed.



The model is trained over 10 epochs, with performance metrics tracked for both training and validation sets.





