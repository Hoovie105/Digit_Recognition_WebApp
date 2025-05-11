The Digit_Recognition_WebApp is a web application that uses a Convolutional Neural Network (CNN) model to recognize handwritten digits. 
The model is trained on the MNIST dataset and implemented using Django. Users can submit an image, and the model will predict the digit, displaying the 
confidence score. The app features a stylish UI built with Bootstrap, image preview before submission, and a loading animation during processing.

![image](https://github.com/user-attachments/assets/d67a5d7c-2b55-4382-bc65-8b7bbaf382ae)


Features:
- Handwritten digit recognition using a CNN model.
- Model confidence score display.
- Stylish UI with Bootstrap for a responsive and clean design.
- Image preview before submission for better user experience.
- Loading animation while processing the prediction.
- Technologies Used
- TensorFlow: For the machine learning model.
- Keras: For building and training the neural network.
- NumPy: For numerical operations.
- Matplotlib: For plotting and visualizations (if needed).
- Django: For the web application backend.
- Python: The primary programming language.
- HTML/CSS: For the frontend structure and styling.
- JavaScript: For interactivity

Installation
To set up the project locally:

1- Clone the repository:
git clone <https://github.com/Hoovie105/Digit_Recognition_WebApp>

2-Navigate to the project directory:
cd digit_recognition

3- Create a virtual environment (optional but recommended):
- python -m venv env
- source env/bin/activate  # For Linux/Mac
- .\env\Scripts\activate   # For Windows

4- Install the required dependencies:
- pip install tensorflow
- pip install matplotlib
- pip install django

5- Run the Django development server:
python manage.py runserver

##Contributing
Feel free to contribute by forking the repository and submitting a pull request. All contributions are welcome!
