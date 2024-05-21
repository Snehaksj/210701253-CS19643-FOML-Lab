from flask import Flask,request,jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
import numpy as np
import os
from flask_cors import CORS

app=Flask(__name__)
CORS(app)
model=load_model("../Model/AyurBotClassEf.h5",custom_objects={'KerasLayer':hub.KerasLayer})
classes=os.listdir("/mnt/d/Academics/Projects/DL/AyurBotanica-TransferLearningModel/dataset/")
n=len(classes)
print(n)
label_ind={}
for i in range(n):
    label_ind[i]=classes[i]

def preprocess_image(img_path):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array = img_array / 255.0 
    return img_array

@app.route("/predict",methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"No images uploaded"})
    img_file=request.files["image"]
    img_path="temp_path.jpeg"
    img_file.save(img_path)
    img_array=preprocess_image(img_path)
    try:
        predictions=model.predict(img_array)
        print(predictions)
        os.remove(img_path)
        predList=predictions.tolist()
        # print(predList)
        l=[]
        for i in range(n):
            temp=[predList[0][i],label_ind[i]]
            l.append(temp)
        l.sort()
        l.reverse()
        ans=[]
        for i in range(5):
            ans.append(l[i])
        # os.remove(img_path)
        return jsonify({"predictions":ans})
    except Exception as e:
        os.remove(img_path)
        return jsonify({"error":str(e)})

if __name__=='__main__':
    app.run(debug=False)