# Mango Leaf Classifier

This project is a machine learning application designed to classify the health status of mango leaves. It utilizes image recognition techniques to identify whether a mango leaf is healthy or affected by a disease.

## Features

-   **Image Classification:** Classifies mango leaves into different health categories.
-   **User-Friendly Interface:** Provides an easy-to-use interface for uploading and classifying images.
-   **Model Training:** Includes the ability to train the model on new datasets.
-   **Model Evaluation:** Provides metrics to evaluate the performance of the trained model.

## Technologies Used

-   **Python:** The primary programming language.
-   **Pytorch:** For building and training the deep learning model.
-   **OpenCV:** For image processing.
-   **Gradio:** For creating the web application interface.
-   **Other Libraries:** NumPy, Pandas, Matplotlib, etc.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mango-leaf-classifier.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd mango-leaf-classifier
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the application**
    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Upload an image:** Use the web interface to upload an image of a mango leaf.
2.  **Classify the image:** The application will process the image and provide a classification result.
3. **Train the model:** You can train the model with new data.

## Project Structure

```
mango-leaf-classificator/
├── app/                     # Application files
│   ├── app.py               # Main application file           
    ├── pyproject.toml       # Project dependencies
└──  README.md           # Project documentation
