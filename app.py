from flask import Flask, render_template, request, send_from_directory
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from tensorflow import keras
model = keras.models.load_model('static/mask_predict.h5')

#COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    #global COUNT
    img = request.files['image']
    img.save('static/mask.jpg') 
    #img.save('static/{}.jpg'.format(COUNT))    

    sample_mask_img = cv2.imread('static/mask.jpg')

    #sample_mask_img = cv2.imread('static/{}.jpg'.format(COUNT))
    #sample_mask_img = cv2.imread('../input/face-mask-12k-images-dataset/Face Mask Dataset/Train/WithMask/1058.png')
    sample_mask_img = cv2.resize(sample_mask_img,(128,128))
    sample_mask_img = np.reshape(sample_mask_img,[1,128,128,3])
    sample_mask_img = sample_mask_img/255.0
    #model.predict(sample_mask_img)

    prediction = model.predict(sample_mask_img)

    #x = round(prediction[0,0], 2)
    #y = round(prediction[0,1], 2)
    #preds = np.array([x,y])
    #COUNT += 1
    return render_template('prediction.html', data=prediction)


@app.route('/load_img')
def load_img():
    #global COUNT
    return send_from_directory('static', "mask.jpg")
    #return send_from_directory('static', "{}=.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



