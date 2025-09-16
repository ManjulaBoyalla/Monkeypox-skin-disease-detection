import email
from django.shortcuts import render
# Create your views here.
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from unicodedata import name
from django.shortcuts import render
from django.http import HttpResponse
from .models import *
import joblib
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from . models import Monkeydata
# Create your views here.


def index(request):

    return render(request, 'index.html')

def register(request):
    if request.method == "POST":
        name = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        c_password = request.POST['c_password']
        if password == c_password:
            if Users.objects.filter(email=email).exists():
                return render(request, 'register.html', {'message': 'User with this email already exists'})
            new_user = Users(name=name, email=email, password=password)
            new_user.save()

            return render(request, 'login.html', {'message': 'Successfully Registerd!'})
        return render(request, 'register.html', {'message': 'Password and Conform Password does not match!'})
    return render(request, 'register.html')
 

def login(request):
    if request.method == "POST":
        email = request.POST['email']
        password1 = request.POST['password']
        
        try:
            user_obj = Users.objects.get(email=email)
        except Users.DoesNotExist:
            return render(request, 'login.html', {'message': 'Invalid Username or Password!'})
        
        password2 = user_obj.password
        if password1 == password2:
            user_id = user_obj.id
            request.session['user_id'] = user_id
            return render(request, 'home.html')
        else:
            return render(request, 'login.html', {'message': 'Invalid Username or Password!'})
    return render(request, 'login.html')

def home(request):
    return render(request,'home.html')

def about(request):
    return render(request,'about.html')

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
def upload(request):
    # if request.method=='POST':
    #     classes=[]
    #     commment = ''
        
    #     file=request.FILES['file']
    #     m=Monkeydata(img=file)
    #     m.save()

    #     fn = file.name
    #     path = os.path.join('static/img', fn)
    #     with open(path, 'wb+') as destination:
    #         for chunk in file.chunks():
    #             destination.write(chunk)

    #     knn_model = joblib.load(r'Backend\svm_model.pkl')
    #     label_binarizer = joblib.load(r'Backend\label_binarizer_m.pkl')
    #     feature_extractor = load_model(r'Backend\feature_extractor_model_m.h5')

    #     # Function to predict the class of a single image
    #         # Preprocess the image
    #     img = load_img(path, target_size =(150,150))
    #     img_array = img_to_array(img)
    #     img_array = np.expand_dims(img_array, axis=0)
    #     img_array = preprocess_input(img_array)
    #     # img_array = preprocess_image(path)
    #     features = feature_extractor.predict(img_array)
    #     features_flat = features.reshape((features.shape[0], -1))
        
    #     # Predict the class
    #     prediction = knn_model.predict(features_flat)
    #     predicted_class = label_binarizer.inverse_transform(prediction)
        
    #     # return predicted_class[0]
      
    #     # Test the prediction
    #     return render(request,'upload.html',{'result':predicted_class[0],'file_name': file.name})
    if request.method == 'POST':
        classes = []
        comment = ''
        
        file = request.FILES['file']
        m = Monkeydata(img=file)
        m.save()

        fn = file.name
        path = os.path.join('static/img', fn)
        with open(path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        knn_model = joblib.load('models/svm_model.pkl')
        label_binarizer = joblib.load('models/label_binarizer_m.pkl')
        feature_extractor = load_model('models/feature_extractor_model_m.h5')

        # Preprocess the image
        img = load_img(path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features
        features = feature_extractor.predict(img_array)
        features_flat = features.reshape((features.shape[0], -1))
        
        # Check feature shape
        print("Features shape:", features_flat.shape)
        
        # Predict the class
        prediction = knn_model.predict(features_flat)
        print("Prediction shape:", prediction.shape)
        print("Prediction:", prediction)

        # Ensure prediction shape is 2D
        if prediction.ndim == 1:
            prediction1 = np.expand_dims(prediction, axis=1)
            print("Prediction:", prediction)

        # predicted_class = label_binarizer.inverse_transform(prediction1)
        # print("Prediction:", predicted_class)

        # # Ensure predicted_class is not empty
        # if len(predicted_class) > 0:
        #     result = predicted_class[0]
        # else:
        #     result = "Unknown"

        return render(request, 'upload.html', {'result': prediction[0], 'file_name': file.name})
    return render(request,'upload.html')
        
        
