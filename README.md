# ObjectDetection-django

YOLO object detector with images url<br>

http://127.0.0.1:8000/api/detect/image?image_url=https://www.driving.co.uk/s3/st-driving-prod/uploads/2020/02/2020-Vauxhall-Corsa-SRi-UK-01.jpg

Output :<br>

{"success": true, "objects": ["car"], "count": 1}

<br>
<img src="https://github.com/sjitprogrammer/ObjectDetection-django/blob/master/outpuy.PNG">
<br>


Or
<br>

Detect with camera stream

http://localhost:8000/api/detect/

<br>
<img src="https://github.com/sjitprogrammer/ObjectDetection-django/blob/master/car.png">
<br>
