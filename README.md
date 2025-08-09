# CAT-VS-DOG
🐱🐶 A deep learning CNN model to classify images as Cat or Dog. Includes dataset collection, training with data augmentation, and a Streamlit web app for real-time predictions. Built with TensorFlow/Keras, Python, and icrawler
📌 Features
Dataset Collection: Downloads images using Bing Image Crawler (image_down.py)

Model Training: CNN-based classifier with data augmentation (train.py)

Web App Deployment: User-friendly Streamlit app for uploading and classifying images (app.py)

Balanced Training: Uses compute_class_weight to handle class imbalance

Visualization: Accuracy and loss plots after training
🛠️ Tech Stack
Python 3

TensorFlow / Keras

Streamlit

NumPy, Matplotlib

icrawler for image scraping
📂 Project Structure
├── app.py               
├── train.py               
├── image_down.py         
├── cat_vs_dog_model.h5        
└── README.md     
📊 Example Output
Prediction in Streamlit:

🐱 This is a Cat!

🐶 This is a Dog!
