import cv2
import argparse
import configparser

import numpy as np

import torch
import torchvision

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json 

import time


def load_model(model_name):
# Create ResNet model (example)
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    inference_model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    saved_model='./'+model_name+'.pt'
    torch.save(inference_model, saved_model)
    return inference_model




def rn50_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    
    return preprocess

# decode the results into ([predicted class, description], probability)
def predict(img, model):
#    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available(): #comment this line to run on CPU
        input_batch = input_batch.to('cuda') #comment this line to run on CPU
        model.to('cuda') #comment this line to run on CPU

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return d[str(ind.item())], sm_output[ind] #([predicted class, description], probability)

def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    #torch.cuda.synchronize() #comment this line to run on CPU
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            #torch.cuda.synchronize() #comment this line to run on CPU
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Processing config input.")
    parser.add_argument("config_file", type=str, help="specify config file")  
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    rtsp_link = config.get("camera_info", "rtsp_link")
    inference_model=config.get("model","inference_model")
    try:
        model=load_model(inference_model)
    except ValueError:
        print("Cannot load model")
    model.eval()
    try:
        cap=cv2.VideoCapture(rtsp_link)
    except ValueError:
        print("Cannot open rtsp link!")
    
    with open("./imagenet_class_index.json") as json_file: 
        d = json.load(json_file)   
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while True:
        ret,frame=cap.read()
        if not ret:
            break
        img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image= Image.fromarray(img)
        pred, prob = predict(image, model)
        plt.imshow(img)
        plt.axis('off')
        plt.title(pred)
        print('{} - Predicted: {}, Probablility: {}'.format(time.time(),pred, prob))

    cap.release()
    cv2.destroyAllWindows()
