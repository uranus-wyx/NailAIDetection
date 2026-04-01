from locust import HttpUser, task, constant
import os

class UploadUser(HttpUser):
    wait_time = constant(0)

    @task
    def submit(self):
        with open("test.jpg", "rb") as f:
            self.client.post(
                "/submit",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )

    @task
    def predict(self):
        with open("test.jpg", "rb") as f:
            self.client.post(
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")},
                timeout=180,
            )