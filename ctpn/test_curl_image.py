import cv2
import numpy as np
import os
import base64
import json
import requests
import tensorflow as tf

image = r"/home/kspook/text-detection-ctpn/data/demo/006.jpg"
'''
    raw_image =  tf.placeholder(tf.string,  name='tf_example')  #输入原始图像
    #serialized_tf_example =  tf.placeholder(tf.string, shape=[None, None, None, 3], name='tf_example')  #输入原始图像
    #raw_image=tf.placeholder(tf.string, shape=[None, None, None, 3], name='tf_example')  #输入原始图像
    print('raw_image, ')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(
            shape=[], dtype=tf.string),
    }
    tf_example = tf.parse_example(raw_image , feature_configs)
    #tf_example = tf.parse_example(serialized_tf_example, feature_configs)

    jpegs = tf_example['image/encoded']

    image_string = tf.reshape(jpegs, shape=[])
'''
'''
image = cv2.imread("/home/kspook/text-detection-ctpn/data/demo/006.jpg", cv2.IMREAD_COLOR)
image = image.astype(np.float32) / 255
image = image.tolist()
'''
URL="http://localhost:9001/v1/models/ctpn:predict" 
#URL = "http://{HOST:port}/v1/models/<modelname>/versions/1:classify" 
headers = {"content-type": "application/json"}
image_content = base64.b64encode(open(image,'rb').read()).decode("utf-8")
body = {
    #"signature_name": "ctpn_recs_predict",
    "signature_name": "predict_images_post",
    "inputs": [
       	      image
       	      #{"image": { "b64": image }}
       	      #{"image": { "b64": image_content }}
       	      #{"image": { "b64": "$(base64 /home/kspook/text-detection-ctpn/data/demo/006.jpg)" }}
    ]
}
'''
body={
    # "model_spec": {"name": "ctpn", "signature_name": "ctpn_recs_predict"}, 
     "signature_name": "ctpn_recs_predict",
     "inputs": {"images": 
                        {"dtype": 7, 
                         "tensor_shape": {"dim":[{"size": 1}]}, 
                         "string_val": [image_content]
                        }
      }
}
'''
data=json.dumps(body)
print(data)
r= requests.post(URL, data=json.dumps(body), headers = headers) 
print(r.text)


'''
    #"signature_name": "serving_default",
    #"examples": [
    #            {"image/encoded": {"b64":image_content}}
    #            ]
    #}
curl -X POST \
  http://localhost:9001/v1/models/ctpn \
  -d '{
  "signature_name": "ctpn_recs_predict",
  "inputs": {
     	"image": { "b64": "$(base64 /home/kspook/text-detection-ctpn/data/demo/006.jpg)" }
  }
}'
'''
