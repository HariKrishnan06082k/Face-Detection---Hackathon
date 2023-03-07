from __future__ import division
import warnings
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
warnings.filterwarnings("ignore")
import os


#FUNCTION FOR DETECTING BLUR or PIXELATED
#Returns True if either blurred or pixelated
class FaceDetection:
    def __init__(self,img):
        self.image = img

    def blur_pixelated(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pix = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        size=60
        thresh=10
        (h, w) = gray_image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        fft = np.fft.fft2(gray_image)
        fftShift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fftShift))
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        #The image will be considered "blurry" if the mean value of the magnitudes is less than the threshold value
        if mean <= thresh:
            print("\nPicture Rejected")
            print("Mean = {}".format(mean))
            print("Pix = {}".format(pix))
            return True
        else:
            print("\nPictured Passed the blurriness test")
            print("Mean = {}".format(mean))
            print("Pix = {}".format(pix))
            return False
        
    def realvscartoon(self):
        s=0
        ym=0
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([self.image],[i],None,[256],[0,256])
            auc=s+sum(histr)
            if ym<max(histr):ym=max(histr)
        if auc/ym>20:
            print("\nReal - PICTURE ACCEPTED")
        else :
            print("\nCartoon - PICTURE REJECTED")
            
            
    def calc_hist(self,img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)
    
    def spoof(self):
    
        modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        clf = joblib.load('face_spoofing.pkl')
    
    
        sample_number = 1
        count = 0
        measures = np.zeros(sample_number, dtype=np.float)
        
    
        
        blob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        faces3 = net.forward()
    
        measures[count%sample_number]=0
        height, width = self.image.shape[:2]
        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                roi = self.image[y:y1, x:x1]
                
                img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
                img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
        
                ycrcb_hist = self.calc_hist(img_ycrcb)
                luv_hist = self.calc_hist(img_luv)
        
                feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
                feature_vector = feature_vector.reshape(1, len(feature_vector))
        
                prediction = clf.predict_proba(feature_vector)
                prob = prediction[0][1]
        
                measures[count % sample_number] = prob
        
                #cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
        
                #print (measures, np.mean(measures))
                if 0 not in measures:
                    if np.mean(measures) >= 0.85:
                        print('\nNot a spoof image')
                    else:
                        print('\nSpoof image')
        count+=1
    
    def emotion_detector(self):
        json_file = open('fer.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("fer.h5")
        x=None
        y=None
        labels = ['Rejected', 'Rejected', 'Rejected', 'Accepted', 'Accepted', 'Accepted', 'Rejected']
        gray=cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face.detectMultiScale(gray, 1.3  , 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            #predicting the emotion
            yhat= loaded_model.predict(cropped_img)
            print("\nEmotion: "+labels[int(np.argmax(yhat))])
    
    def mask_image(self):
    	# load our serialized face detector model from disk
    	print("\n[INFO] loading face detector model...")
    	prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    	weightsPath = os.path.sep.join(["face_detector",
    		"res10_300x300_ssd_iter_140000.caffemodel"])
    	net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
    
    	# load the face mask detector model from disk
    	print("[INFO] loading face mask detector model...")
    	model = load_model("mask_detector.model")
    
    	# load the input image from disk, clone it, and grab the image spatial
    	# dimensions
    	orig = self.image.copy()
    	(h, w) = self.image.shape[:2]
    
    	# construct a blob from the image
    	blob = cv2.dnn.blobFromImage(orig, 1.0, (300, 300),
    		(104.0, 177.0, 123.0))
    
    	# pass the blob through the network and obtain the face detections
    	net.setInput(blob)
    	detections = net.forward()
    
    	# loop over the detections
    	for i in range(0, detections.shape[2]):
    		# extract the confidence (i.e., probability) associated with
    		# the detection
    		confidence = detections[0, 0, i, 2]
    
    		# filter out weak detections by ensuring the confidence is
    		# greater than the minimum confidence
    		if confidence > 0.5:
    			# compute the (x, y)-coordinates of the bounding box for
    			# the object
    			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    			(startX, startY, endX, endY) = box.astype("int")
    
    			# ensure the bounding boxes fall within the dimensions of
    			# the frame
    			(startX, startY) = (max(0, startX), max(0, startY))
    			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
    			# extract the face ROI, convert it from BGR to RGB channel
    			# ordering, resize it to 224x224, and preprocess it
    			face = orig[startY:endY, startX:endX]
    			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    			face = cv2.resize(face, (224, 224))
    			face = img_to_array(face)
    			face = preprocess_input(face)
    			face = np.expand_dims(face, axis=0)
    
    			# pass the face through the model to determine if the face
    			# has a mask or not
    			(mask, withoutMask) = model.predict(face)[0]
    
    			# determine the class label and color we'll use to draw
    			# the bounding box and text
    			label = "Mask" if mask > withoutMask else "No Mask"
    			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    			# include the probability in the label
    			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    
    			# display the label and bounding box rectangle on the output
    			# frame
    			cv2.putText(orig, label, (startX, startY - 10),
    				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    			cv2.rectangle(orig, (startX, startY), (endX, endY), color, 2)
    
    	# show the output image
    	cv2.imshow("Output", orig)
    	cv2.waitKey(0)
    
    def main(self):
        self.blur_pixelated()
        self.realvscartoon()
        self.spoof()
        self.emotion_detector()
        self.mask_image()
        
if __name__ == '__main__':
    img = cv2.imread("static/uploads/mask_man.jpg")  # read image from django directory 
    face_det = FaceDetection(img)
    face_det.main()
    