import numpy as np
import onnxruntime

weight_path = "/mnt/ssd/santapo/revamp_idcard/vietnamese-identity-card/resnet50-v2-7.onnx"
device = ["CUDAExecutionProvider", "CPUExecutionProvider"]
model = onnxruntime.InferenceSession(weight_path, providers=device)

def predict(images):
    ort_inputs = {model.get_inputs()[0].name: images.astype(np.float32)}
    ort_outputs = model.run(None, ort_inputs)
    # print(ort_outputs)

# for _ in range(1000):
#     rand_images = np.random.rand(1, 3, 800, 1280)
#     predict(rand_images)

# for _ in range(1000):
#     rand_images = np.random.rand(4, 3, 800, 1280)
#     predict(rand_images)

# for _ in range(1000):
#     rand_images = np.random.rand(8, 3, 800, 1280)
#     predict(rand_images)

is_continue = True
batch_size = 1

while is_continue:
    rand_images = np.random.rand(batch_size, 3, 224, 224)
    predict(rand_images)
    

print("DONE!")