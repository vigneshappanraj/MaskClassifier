from flask import Flask, render_template, request, send_from_directory
import cv2
from scipy.spatial import distance

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
model = keras.models.load_model('static/mask_predict.h5')
face_model = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

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
    sample_mask_img = cv2.cvtColor(sample_mask_img, cv2.IMREAD_GRAYSCALE)
    faces = face_model.detectMultiScale(sample_mask_img,scaleFactor=1.1, minNeighbors=4)

    mask_label = {0:'OK!',1:'Mask Podra'}
    dist_label = {0:(0,255,0),1:(255,0,0)}
    MIN_DISTANCE = 0
    a = None

    if len(faces)>=1:
        label = [0 for i in range(len(faces))]
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2],faces[j][:2])
                if dist<MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
        new_img = cv2.cvtColor(sample_mask_img, cv2.COLOR_RGB2BGR) #colored output image
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = new_img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0
            mask_result = model.predict(crop)
            a = mask_result[0][0]
            #cv2.putText(new_img,mask_label[round(mask_result[0][0])],(x, y+90), cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
            #cv2.putText(new_img,str(mask_result[0][0]),(x, y+90), cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
            cv2.rectangle(new_img,(x,y),(x+w,y+h),dist_label[label[i]],1)
            cv2.imwrite('static/mask.jpg', cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        #plt.figure(figsize=(10,10))
        #plt.imshow(new_img)
            
    else:
        print("Retry")


    #sample_mask_img = cv2.imread('static/{}.jpg'.format(COUNT))
    #sample_mask_img = cv2.imread('../input/face-mask-12k-images-dataset/Face Mask Dataset/Train/WithMask/1058.png')
    #sample_mask_img = cv2.resize(sample_mask_img,(128,128))
    #sample_mask_img = np.reshape(sample_mask_img,[1,128,128,3])
    #sample_mask_img = sample_mask_img/255.0
    #model.predict(sample_mask_img)

    #prediction = model.predict(sample_mask_img)

    #x = round(prediction[0,0], 2)
    #y = round(prediction[0,1], 2)
    #preds = np.array([x,y])
    #COUNT += 1
    return render_template('prediction.html', data=a)


@app.route('/load_img')
def load_img():
    #global COUNT
    return send_from_directory('static', "mask.jpg")
    #return send_from_directory('static', "{}=.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



