import onnxruntime
import cv2
import numpy as np
import base64
import os
import requests
import json
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def do_cnn(msg):
    img64 = base64.b64decode(msg); 
    npimg = np.fromstring(img64, dtype=np.uint8); 
    img = cv2.imdecode(npimg, 0)  #以灰度图的方式读取要预测的图片
    img = cv2.resize(img, (28, 28))
    #假设输入是白底黑字，进行反色，因为学习是黑底白字
    height,width=img.shape
    dst=np.zeros((height,width),np.uint8)
    for i in range(height):
        for j in range(width):
            dst[i,j]=255-img[i,j]

    img = dst
    img=np.array(img).astype(np.float32)
    img=np.expand_dims(img,0)
    img=np.expand_dims(img,0)#扩展后，为[1，1，28，28]
    ort_session = onnxruntime.InferenceSession("mnist_cnn_onnx.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out = ort_outs[0]
    print(img_out)
    return img_out


if __name__ =='__main__':
    scfurl= "http://"+os.environ["SCF_RUNTIME_API"]+":"+os.environ["SCF_RUNTIME_API_PORT"]
    #ready 
    requests.post(scfurl+"/runtime/init/ready")
    while True:
        req=requests.get(scfurl+'/runtime/invocation/next').json()
        res=do_cnn(req["image"])
        requests.post(scfurl+'/runtime/invocation/response', data=json.dumps({
            "shape":res.shape,
            "size":res.size,
            "result":res.argmax()
        }, cls=NpEncoder), headers={'Content-Type': 'application/json'})