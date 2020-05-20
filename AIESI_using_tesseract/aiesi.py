

import cv2
import argparse
import numpy as np
!pip install opencv-wrapper
import opencv_wrapper as cvw
from skimage.filters import threshold_local
import json

!pip install -U git+https://github.com/madmaze/pytesseract.git

import pytesseract
!sudo apt install tesseract-ocr
!pip install colorama



def get_acc(directory, path):
    font     = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor  = (255,0,0)
    lineType = 1
    path = directory+path
    # path = args['image']
    # op_path = args['output']

    op_path = directory
    if op_path[-1]!='/':
    	op_path.append('/')


    #Threshold
    image = cv2.imread(path)

    height,width,channel = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    T = threshold_local(gray, 15, offset = 6, method = "gaussian") # generic, mean, median, gaussian
    thresh = (gray > T).astype("uint8") * 255
    thresh = ~thresh


    #Dilation
    kernel =np.ones((1,1), np.uint8)
    ero = cv2.erode(thresh, kernel, iterations= 1)
    img_dilation = cv2.dilate(ero, kernel, iterations=1)

    # Remove noise
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilation, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    final = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if sizes[i] >= 10:   #filter small dotted regions
            final[labels == i + 1] = 255

    kern = np.ones((5,15), np.uint8)
    img_dilation = cv2.dilate(final, kern, iterations = 1)
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rects = map(lambda c: cv2.boundingRect(c), contours)
    sorted_rects = sorted(rects, key =lambda r: r[0])
    sorted_rects = sorted(sorted_rects, key =lambda r: r[1])

    tt = image.copy()
    dictionary = {}
    etfo=''
    for i,rect in enumerate(sorted_rects):
        temp_dic = {}
        x,y,w,h = rect
        if(w<20 or h<20):
            continue
        temp_dic['coords'] = [x,y,w,h]
        words = []
        temp = tt[y:y+h, x:x+w]
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        hi = pytesseract.image_to_data(temp, config=r'--psm 6')
        hi = hi.split()
        ind = 22
        while(True):
            if (ind>len(hi)):
                break
            if(int(hi[ind])==-1):
                ind+=11
            else:

                tem = {}
                tem['confidence'] = hi[ind]
                tem['text'] = hi[ind+1]
                etfo=etfo+hi[ind+1]
                etfo=etfo+" "
                x+=len(hi[ind+1])*20
                ind+=12
                words.append(tem)
        temp_dic['words'] = words
        etfo=etfo+'\n'
        #cvw.rectangle(image, rect, cvw.Color.GREEN, thickness=1)
        dictionary[i] = temp_dic


    cv2.imwrite(op_path+"result.png", image)
    return json.dumps(dictionary),etfo

gson_data, etfo = get_acc('sample/','hello.png')
print(gson_data)



import argparse
import torch
import json
from torch import nn

class MyModel0(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 5)

    def forward(self, inpt):
        embedded = self.embed(inpt)
        feature, _ = self.lstm(embedded)
        oupt = self.linear(feature)
        return oupt

"""### Hyper- parameters"""

device= 'cpu'
hidden_size = 256

device= torch.device('cpu')

!pip install colorama

import json
import os
import random
from os import path
from string import ascii_uppercase, digits, punctuation

import colorama
import numpy
import regex

VOCAB= ascii_uppercase+digits+punctuation+" \t\n"

print(etfo)

temp_text=''
for i in range(664):
    temp_text=temp_text+etfo[i]
etfo=temp_text

# conver lower text to uppper text in ETFO
etfo= etfo.upper()
print(etfo)

def get_test_data():
    
    text = etfo
    text_tensor = torch.zeros(len(text), 1, dtype=torch.long)
    text_tensor[:, 0] = torch.LongTensor([VOCAB.find(c) for c in text])

    return text_tensor.to(device)

text_tensor = get_test_data()

print(text_tensor.shape)

def pred_to_dict(text, pred, prob):
    res = {"company": ("", 0), "date": ("", 0), "address": ("", 0), "total": ("", 0)}
    keys = list(res.keys())

    seps = [0] + (numpy.nonzero(numpy.diff(pred))[0] + 1).tolist() + [len(pred)]
    for i in range(len(seps) - 1):
        pred_class = pred[seps[i]] - 1
        if pred_class == -1:
            continue

        new_key = keys[pred_class]
        new_prob = prob[seps[i] : seps[i + 1]].max()
        if new_prob > res[new_key][1]:
            res[new_key] = (text[seps[i] : seps[i + 1]], new_prob)

    return {k: regex.sub(r"[\t\n]", " ", v[0].strip()) for k, v in res.items()}

print(text_tensor.shape)
for i in range(len(text_tensor)-1):
  if text_tensor[i]<0 or text_tensor[i]>70:
    text_tensor = torch.cat([text_tensor[0:i], text_tensor[i+1:]])

def test():
    
    model = MyModel0(len(VOCAB), 16, hidden_size).to(device)

    model.load_state_dict(torch.load("model.pth"))

    model.eval()
  
    with torch.no_grad():
            oupt = model(text_tensor)
            prob = torch.nn.functional.softmax(oupt, dim=2)
            prob, pred = torch.max(prob, dim=2)
            prob = prob.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()
            real_text = etfo
            result = pred_to_dict(real_text, pred, prob)

            with open("result/" + 'result' + ".json", "w", encoding="utf-8") as json_opened:
                json.dump(result, json_opened, indent=4)

            print(result)
            #print(key)


if __name__ == "__main__":
    test()

