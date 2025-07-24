from mlchain.workflows import Task, Parallel
from mlchain.client import Client
import glob
import cv2

model = Client(api_address='0.0.0.0:8044', serializer='msgpack_blosc').model(check_status=False)

def task(path):
    img = cv2.imread(path)
    if img is not None:
        return model.predict(img)
    else:
        return None

all_task = []
all_sample = glob.glob('./test_failcase_card/*')
for each in all_sample:
    all_task.append(
        Task(
            task, each
        )
    )

res = Parallel(all_task, max_threads=8).run(progress_bar=True)