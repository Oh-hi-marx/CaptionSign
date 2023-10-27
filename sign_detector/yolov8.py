import os
from ultralytics import YOLO

import cv2
from tqdm import tqdm 
import numpy as np

def expand(detections, w,h, ratio = 0.5):
    boxes = []
    for d in detections:
        width = d[2]-d[0]
        height = d[3] -d[1]
        longest = max(width, height)
        shortest = min(width, height)
        center = [d[0] + (width/2), d[1]+(height/2)]
        
        box = [0,0,0,0]

        halflongest = (longest/2) * (1+ratio)   
        for i in range(30):
            if(center[0] - halflongest > 0 ):
                box = [center[0]-halflongest, box[1],box[2],box[3]]
                break
            halflongest *=0.95

        halflongest = (longest/2) * (1+ratio)   
        for i in range(30):
            if(center[1] - halflongest > 0 ):
                box = [box[0], center[1]-halflongest,box[2],box[3]]
                break
            halflongest *=0.95

        
        halflongest = (longest/2) * (1+ratio)   
        for i in range(30):
            if(center[0] + halflongest < w ):
                box = [box[0], box[1],center[0]+halflongest,box[3]]
                break
            halflongest *=0.95

        
        halflongest = (longest/2) * (1+ratio)   
        for i in range(30):
            if(center[1]+ halflongest <h):
                box = [box[0], box[1],box[2],center[1]+ halflongest]
                break
            halflongest *=0.95

        

        box= [int(f) for f in box]
        boxes.append(box)
    return boxes

def writeVideo(frames, path, w,h , fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path , fourcc, fps, (w, h))
    for frame in frames:
        fh,fw,_ = frame.shape
        #paste onto square background
        longest = max(fw,fh)
        blank = np.zeros((longest,longest,3), np.uint8)
        wgap =longest-fw
        hgap = longest-fh
        if(wgap):
            wgap = int(wgap/2)
        if(hgap):
            hgap = int(hgap/2)
        blank[hgap:fh+hgap,wgap:fw+wgap]= frame
        fh,fw,_ = blank.shape
        #resize to output resolution
        if(fh!=h or fw!=w):
            blank = cv2.resize(blank, (w,h))
        video.write(blank)
    video.release()

videopath = 'videos'
outputpath = 'outputs'
stride = 5 #in seconds
videos = [videopath + os.sep + f for f in os.listdir(videopath)]

model = YOLO("yolov8l.pt")
outw = outh = 336
for video in tqdm(videos):
    try:
        savepath = outputpath+ os.sep + video.split(os.sep)[-1].rsplit('.',1)[0]
        os.makedirs(savepath, exist_ok=True)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride_frames = int(fps * stride)

        counter = 0
        persons = []
        crops = []
        #read frames
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if ret == True:
                h,w,c = frame.shape
                if(counter%stride_frames==0):
                    for i, crop in enumerate(crops):
                        path = savepath+os.sep+ str(counter) +"_"+str(i)+".mp4"
                        writeVideo(crop, path, outw,outh, fps)
                    persons = [] #delete previous results
                    crops = []
                    #detect person
                    results = model.predict(frame, classes =[0], conf =0.7, verbose = False)
                    detections = results[0].boxes.xyxy.cpu().numpy()
                    persons = expand(detections, w,h)
                    crops = [[] for i in persons]
                #crop out persons
                for i, person in enumerate(persons):
                    crops[i].append(frame[person[1]:person[3], person[0]:person[2]])
                counter+=1
            else:
                for i, crop in enumerate(crops):
                    path = savepath+os.sep+ str(counter) +"_"+str(i)+".mp4"
                    writeVideo(crop, path, outw,outh, fps)
                break
    except Exception as E:
        print(E)