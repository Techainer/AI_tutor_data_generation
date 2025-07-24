from locust import HttpUser, task, between
import numpy as np
import random
import cv2
import glob
import os

all_image_path = glob.glob("tests/sample_dataset/images/*.jpg")
class APIUser(HttpUser):
    # Setting the host name and wait_time
    host = 'http://0.0.0.0:8002'
    wait_time = between(3, 5)

    @task()
    def predict(self):
        image_path = random.choice(all_image_path)
        self.client.post("/call/predict", files=dict(image=open(image_path, 'rb')))