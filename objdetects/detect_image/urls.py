from django.urls import path,include
from .views import yolo_detect_api
urlpatterns = [
    path('image',yolo_detect_api)
]
