from flask import Flask
from flask import jsonify
from flask import request
from PIL import Image
import io
import json
import numpy as np

from face_net.face_process import *
from human_detection.human_detect import *

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def image_processing():
   # get data from client
   data = request.get_data()

   img = Image.open(io.BytesIO(data))
   img_array  = np.array(img)

   # image process here
   result = dict()
   # add result to dict
   identity, emotion, age = face_process.face_process(img_array)
   humans = human_detection.human_detect(img_array)
   result['emotion'] = emotion
   result['identity'] = identity
   result['age'] = age
   result['humans'] = humans # [left-up-x, left-up-y, right-bottom-x, right-bottom-y, confidence, class]

   #return json file
   return jsonify(result)

if __name__ == '__main__':
   face_process = face_processing()
   human_detection = human_detection()
   app.run()
