import cv2
import numpy as np
import os
import base64
import json
import requests

#image = r"/home/kspook/text-detection-ctpn/data/demo/006.jpg"

image = cv2.imread("/home/kspook/text-detection-ctpn/data/demo/006.jpg", cv2.IMREAD_COLOR)
image = image.astype(np.float32) / 255
image = image.tolist()


URL="http://localhost:9001/v1/models/ctpn:predict" 
#URL = "http://{HOST:port}/v1/models/<modelname>/versions/1:classify" 
headers = {"content-type": "application/json"}
#image_content = base64.b64encode(open(image,'rb').read()).decode("utf-8")
body = {
    "signature_name": "ctpn_recs_predict",
    "inputs": [
       	      {"image": { "b64": image }}
       	      #{"image": { "b64": image_content }}
       	      #{"image": { "b64": "$(base64 /home/kspook/text-detection-ctpn/data/demo/006.jpg)" }}
    ]
}
r = requests.post(URL, data=json.dumps(body), headers = headers) 
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
