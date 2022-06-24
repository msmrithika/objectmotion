import numpy as np
import cv2
import time
video_path = 'C:\\Users\\Smrithika\\dataset7.mp4'
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOWS)
cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorKNN(500, 400, False)	
number_of_white_pix_beg=0
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
        color = (255, 0, 0)
        cv2.drawContours(morph, hull, i, color, -1, 8)
    
    if count==5:
        number_of_white_pix_beg = np.sum(morph == 255) 
    count+=1     
    number_of_white_pix = np.sum(morph == 255)
   
    cv2.imshow('frame',morph)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
if number_of_white_pix_beg>number_of_white_pix:
    print('receding')
else: print('approaching')
cap.release()
cv2.destroyAllWindows()
© 2022 GitHub, Inc.
Terms
Privacy
