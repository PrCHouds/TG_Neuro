# import libraries

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# load image

def li(p):
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/127.5 -1
    img = np.expand_dims(img,0)
    img = tf.convert_to_tensor(img)
    return img

# preprocess image
def pi(img,td=224):
    shp = tf.cast(tf.shape(img)[1:-1], tf.float32)
    sd  = min(shp)
    scl = td/sd
    nhp = tf.cast(shp*scl, tf.int32)
    img = tf.image.resize(img,nhp)
    img = tf.image.resize_with_crop_or_pad(img, td,td)
    return img

def cartoon(img_p):
    # loading image
    si = li(img_p)

    psi = pi(si,td=512)
    psi.shape

    # model dataflow
    m = 'model/1.tflite'
    i = tf.lite.Interpreter(model_path=m)
    ind = i.get_input_details()
    i.allocate_tensors()
    i.set_tensor(ind[0]['index'],psi)
    i.invoke()

    r = i.tensor(i.get_output_details()[0]['index'])()

    # post process the model output
    o = (np.squeeze(r)+1.0)*127.5
    o = np.clip(o,0,255).astype(np.uint8)
    o = cv2.cvtColor(o,cv2.COLOR_BGR2RGB)
    cv2.imwrite("results/"+img_p,o)

if __name__ == "__main__":
    # checking the results
    # bot.infinity_polling()

    imgs = ["content/"+name for name in os.listdir("content/")]

    for i in imgs:
        cartoon(i)