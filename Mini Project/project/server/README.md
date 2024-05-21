# AyurBotanica- Backend

### AyurBotanica
- AyurBotanica leverages the usage of Deep Learning to find the Medicinal/Ayurvedic plants of Indian origin.
- Just drag and drop the image of plant/leaf, and hit the predict button to find the species.
  
### Model
- The model uses the EfficientNetV2 architecture for feature extraction. These extracted features are then fed into additional layers, including a Dropout layer for regularization, followed by a Dense layer with softmax activation for classifying the input into one of several predefined classes.
- The model achives an accuracy of 96% on the test data.

### Backend
- The backend is built using Flask, a lightweight web framework for Python. It defines a single endpoint /predict that accepts POST requests containing image data. 
- CORS support is added using the flask_cors extension to allow requests from web applications hosted on different origins. 
- The backend listens for incoming requests and responds with JSON containing the predictions.