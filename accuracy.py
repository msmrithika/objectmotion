import time

import cv2
import numpy as np

def classifier(video_path):
    TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
         'kcf' : cv2.TrackerKCF_create,
         'mil': cv2.TrackerMIL_create,
         }
    trackers = cv2.legacy.MultiTracker_create()
    #video_path = 'C:\\Users\\Smrithika\\ds_'+'1'+'.mp4'
    #cap = cv2.VideoCapture(0,cv2.CAP_DSHOWS)
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorKNN(100, 400, False)	
    ret, frame = cap.read()
    k=int(input('no of objects'))
    r=list(range(0,k))
    for i in range(k):
        cv2.imshow('Frame',frame)
        bbi = cv2.selectROI('Frame',frame)
        tracker_i = TrDict['csrt']()
        trackers.add(tracker_i,frame,bbi)         

    number_of_white_pix_beg=[None]*k
    number_of_white_pix=[None]*k
    count=0
    while(1):
        ret, frame = cap.read()
        #time.sleep(0.2)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #kernel = np.ones((5,5),np.float32)/25
        #gray = cv2.filter2D(gray,-1,kernel)
        if ret==0:
            break
        fgmask = fgbg.apply(frame)
        thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.Canny(thresh, 500, 500)

    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    
        cntrs = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        hull = []
        for i in range(len(cntrs)):
            hull.append(cv2.convexHull(cntrs[i], False))
        for i in range(len(cntrs)):
            color = 255
            cv2.drawContours(morph, hull, i, color, -1, 8)
        contours,hierarchy = cv2.findContours(morph, 1, 2)    
        (success,boxes) = trackers.update(morph)
        for box,j in zip(boxes,r):
            (x,y,w,h) = [int(a) for a in box]
            cv2.rectangle(morph,(x,y),(x+w,y+h),(0,0,255),2)
            if count==5:
                number_of_white_pix_beg[j] = np.sum(morph[x:x+w,y:y+h] == 255) 
            if count>5 :   
                number_of_white_pix[j] = np.sum(morph[x:x+w,y:y+h] == 255)        
        count+=1
        cv2.imshow('frame',morph)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    cap.release()
    cv2.destroyAllWindows()    
    for j in range(k):   
        if number_of_white_pix_beg[j]>number_of_white_pix[j]:
            return 0
        else: return 1

if __name__== "__main__":
    y_true=[1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,1,1,0,1,0,1,0,1,0,1,0]
    y_pred=[None]*30
    for d in range(30):
        e=str(d+1)
        video_path = 'C:\\Users\\Smrithika\\ds_'+e+'.mp4'
        print(video_path)
        y_pred[d]=classifier(video_path)
    accuracy=np.sum(np.equal(y_true, y_pred)) / len(y_true)
    print(accuracy)
        
        

