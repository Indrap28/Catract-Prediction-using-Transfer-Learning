from flask import Flask, render_template,request,jsonify,url_for,redirect
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app=Flask(__name__)
model = tf.keras.models.load_model('Vgg16_97.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/input')
def pred():
    return render_template('details.html')

@app.route('/output', methods=['GET','POST'])
def output():
    if request.method =='POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=load_img(filepath,target_size=(224,224))
           
        x = img_to_array(img)
        x = preprocess_input(x)
        preds = model.predict(np.array([x]))
        if preds< 0.5:
            result='Cataract was found, Kindly consult a doctor'
        else:
            result='Congrats Eye is Normal'
        print("result")
        #predict = prediction
        return render_template("result.html", predict = result)

if __name__=='__main__':
    app.run(debug = False,port = 5000)


