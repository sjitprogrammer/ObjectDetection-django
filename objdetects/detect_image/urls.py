from django.urls import path,include
from .views import yolo_detect_api,yolo_detect_camera_api,video_feed_1,homPage
urlpatterns = [
    path('image',yolo_detect_api),
    path('camera',video_feed_1, name="video-feed-1"),
    path('',homPage)
]
