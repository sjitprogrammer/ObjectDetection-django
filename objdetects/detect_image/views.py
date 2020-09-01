from django.shortcuts import render
from django.http import HttpResponse, JsonResponse,StreamingHttpResponse
from django.views.decorators import gzip
import urllib.request
from PIL import Image
import cv2
import numpy as np
import time

def homPage(request):
    return render(request, 'test.html')

def yolo_detect_api(request):
    data = {'success':False}
    
    # capture = cv2.VideoCapture(0)
    # ret, frame = capture.read()
    
    # while(True):
    #     ret, frame = capture.read()
    #     gray = cv2.cvtColor(frame,cv2.COLOR_RGB2BGRA)
    #     cv2.imshow('Test', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # capture.release()
    # cv2.destroyAllWindows()
    if request.method == "GET":
        url = request.GET['image_url']
        result, count = yolo_detect(url)

        if result:
            data['success'] = True


    data['objects'] = result
    data['count'] = count
    return JsonResponse(data)

def yolo_detect(url):
    # Load Yolo
    net = cv2.dnn.readNet("./yolo-obj_last.weights", "./cfg/yolo-obj.cfg")
    classes = []
    with open("./data/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image

    with urllib.request.urlopen(url) as url:
        with open('temp.jpg', 'wb') as f:
            f.write(url.read())

    img = cv2.imread('temp.jpg')
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    

    count = 0
    objects = []

    if(len(boxes)==0):
        return objects,count

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            count+=1
            objects.append(label)
            print ("Object is : ",label)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, label, (x, y + 30), font, 2, (0,255,0), 2)


    print ("Count : ",count)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return objects, count


def yolo_detect_camera_api():
    #Load YOLO
    net = cv2.dnn.readNet("./yolov3-tiny.weights","./cfg/yolov3-tiny.cfg")
    classes = []
    with open("./data/coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors= np.random.uniform(0,255,size=(len(classes),3))
    
    #loading image
    cap=cv2.VideoCapture(0) #0 for 1st webcam
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time= time.time()
    frame_id = 0

    while True:
        _,frame= cap.read() # 
        frame_id+=1
        
        height,width,channels = frame.shape
        #detecting objects
        blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),swapRB=True, crop=False) #reduce 416 to 320    

            
        net.setInput(blob)
        outs = net.forward(outputlayers)
        #print(outs[1])


        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    #onject detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

        count = 0
        objects = []
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                count+=1
                objects.append(label)
                confidence= confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
                

        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        name_obj = ""
        for i in range(len(objects)):
            if name_obj=="":
                name_obj = objects[i]
            else:
                name_obj += ","+objects[i]
           

        # cv2.putText(frame,"FPS:"+str(round(fps,2))+"\n count:"+str(count)+"\n Obj name:"+name_obj,(10,50),font,2,(0,0,0),1)
        text = "\n FPS:"+str(round(fps,2))+"\n Count:"+str(count)+"\n ObjName:"+name_obj
        y0, dy = 10, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(frame, line, (10, y ), font, 2, (0,255,0), 2)
        cv2.imwrite('demo.jpg', frame)
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' +open('demo.jpg', 'rb').read() + b'\r\n\r\n')


def video_feed_1(request):
	return StreamingHttpResponse(yolo_detect_camera_api(), content_type='multipart/x-mixed-replace; boundary=frame')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,image = self.video.read()
        ret,jpeg = cv2.imencode('.jpg',image)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def index(request): 
    try:
        return StreamingHttpResponse(gen(VideoCamera()),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted")




