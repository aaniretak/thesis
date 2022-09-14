#Imports
import tensorflow as tf # Tensorflow 2.3
import tensorflow_hub as hub # Tensorflow-hub 0.12
import PIL.Image as Image # Pillow 
import numpy as np # Numpy
from tqdm import tqdm # progress bar package
import tensorflow_datasets as tfds
import os
from time import process_time
from memory_profiler import memory_usage
import tempfile
import tracemalloc
import matplotlib.image as mpimg
import pandas as pd
import PIL.Image as Image # Pillow
import math

################################################################################################################################################################

# Opening the text file and saving to the corresponding lists.
def load_values(file): 
    val_groundtruth = []
    with open(file, 'r') as f:
        val_set = f.read().splitlines()
    for line in val_set:
        # Image ground truth.
        ground_truth = line.split(' ')[1]
        val_groundtruth.append(int(ground_truth))
    return val_groundtruth

################################################################################################################################################################

# Image Preprocessing
def prepare(path, size):
    im = Image.open(path)
    im = im.convert('RGB')
    re_size = int(round(1.14286*size))
    
    width, height = im.size
    new_height = height * re_size // min(width,height)
    new_width = width * re_size // min(width,height)
    im = im.resize((new_width,new_height))

    left = math.floor((new_width - size)/2)
    top = math.ceil((new_height - size)/2)
    right = math.floor((new_width + size)/2)
    bottom = math.ceil((new_height + size)/2)


    im = im.crop((left, top, right, bottom))
    input = (np.array(im))/255
    return input

################################################################################################################################################################

# Ellipse defined by the labels of DAGM
def calc_ellipse(x, y,label):
    # https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
    ...
    [semi_major, semi_minor, rotation, x_centre, y_centre] = label
    term1 = (((x - x_centre) * np.cos(rotation)) + (
        (y - y_centre) * np.sin(rotation)))**2
    term2 = (((x - x_centre) * np.sin(rotation)) - (
        (y - y_centre) * np.cos(rotation)))**2
    ellipse = ((term1 / semi_major**2) + (term2 / semi_minor**2)) <= 1
    return ellipse

##################################################################################################################################################################

# IoU calculation
def iou_calc(labimg, pred):

    pred = np.round(np.squeeze(pred),decimals = 6).astype(dtype=bool)
    intersection = np.logical_and(labimg, pred)
    union = np.logical_or(labimg, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

###################################################################################################################################################################

# Model Inference
def modelInference(model_list):

    [link,batchshape,dataset,background] = model_list

    #Groundtruth
    
    val_groundtruth = load_values(dataset+'/val.txt')

    # Load Model

    model = tf.keras.Sequential([
    hub.KerasLayer(link)])
    model.build(batchshape)
    batch = batchshape[0]

    # Inference and metrics

    t1 = 0 # Variable for correct predictions
    t5 = 0 # Variable for top-5 correct predictions

    imagenum = 0
    latency = []
    avg_lat = []
    data = np.zeros(tuple(batchshape))
    tracemalloc.start()

    for file in tqdm(os.listdir(dataset)[:-1]):
        image = Image.open(os.path.join(dataset, file), 'r')
        data[imagenum%batch,:] = prepare(os.path.join(dataset, file),batchshape[1]) # Preprocessing
        imagenum += 1

        if (imagenum%batch == 0):
            start = process_time()    
            prediction = model.predict(data,batch_size = batch)
            latency.append(process_time()-start)
            avg_lat.append(latency[-1]/batch)
            data = np.zeros(tuple(batchshape))
        
            for i in range(batch):
                if((val_groundtruth[imagenum - 1 - i]+background) in np.argpartition(prediction[batch - 1 -i], -5)[-5:]): # argpartition σε Ο(n) για τα top5
                        t5+=1
                        if(val_groundtruth[imagenum - 1 - i]+background) == np.argmax(prediction[batch - 1 -i]):
                            t1+=1

    if (imagenum%batch != 0):
        start = process_time()    
        prediction = model.predict(data[:(imagenum%batch)],batch_size = batch)
        latency.append(process_time()-start)
        avg_lat.append(latency[-1]/(imagenum%batch))

        for i in range(imagenum - (imagenum % batch),imagenum):
            if((val_groundtruth[i]+background) in np.argpartition(prediction[i - (imagenum - (imagenum % batch))], -5)[-5:]): # argpartition σε Ο(n) για τα top5
                t5+=1
                if(val_groundtruth[i]+background) == np.argmax(prediction[i - (imagenum - (imagenum % batch))]):
                    t1+=1

    memory = round(((tracemalloc.get_traced_memory()[1])/1024)) #peak memory usage   
    t1_acc = round((t1/(imagenum))*100,1) #top-1 accuracy
    t5_acc = round((t5/(imagenum))*100,1) #top-5 accuracy
    throughput = round((imagenum)/sum(latency),2) 
    tracemalloc.stop()

    return {'memory':memory,'top1': t1_acc,'top5':t5_acc,'avg(latency)':avg_lat,'latency':latency,'throughput':throughput}