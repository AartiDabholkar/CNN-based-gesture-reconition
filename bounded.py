import cv2
import numpy as np


#cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(0)
i=1
j=1
name=""



while(cap.isOpened()):
	_,img=cap.read()
	cv2.rectangle(img,(590,10),(350,225),(255,0,0),2)
	img1=img[10:225,350:590]
	#cv2.line(img,(350,128),(600,400),(255,0,0),5)
	cv2.imshow('Frame',img)
	cv2.imshow('Region Of Interest',img1)
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	if k == 13:
		name=str(i)+"_"+str(j)+".jpg"
		cv2.imwrite('C:\\python\\sign_letters_train_new\\'+name,img1)
		if(j<20):
			j+=1
		else:
			while(0xFF & cv2.waitKey(0)!=ord('n')):
				j=1
			j=1
			i+=1
		

cap.release()        
cv2.destroyAllWindows()
