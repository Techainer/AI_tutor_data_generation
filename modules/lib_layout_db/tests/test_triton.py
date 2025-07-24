import numpy as np
import torch


import tritonclient.gr as input_client


model_name = "card_textline_detection"
model_version = ""
triton_url = "localhost:11000"

triton_client = input_client.InferenceServerClient(url=triton_url, verbose=False, concurrency=8)

batch_size = 1
rand_images = torch.randn(batch_size, 3, 512, 512).numpy()

inputs = [input_client.InferInput('x', rand_images.shape, 'FP32')]
inputs[0].set_data_from_numpy(rand_images)
outputs = [input_client.InferRequestedOutput('sigmoid_90.tmp_0')]
responses = triton_client.infer(model_name,
                                inputs,
                                model_version=model_version,
                                outputs=outputs)
triton_res = responses.as_numpy('sigmoid_90.tmp_0')

import numpy as np
import onnxruntime

weight_path = "/mnt/ssd/santapo/revamp_idcard/lib_layout_db/models/card_textline_detection/1/layout_db_efficientnetv2s_20220511.onnx"
device = ["CUDAExecutionProvider", "CPUExecutionProvider"]
model = onnxruntime.InferenceSession(weight_path, providers=device)

def predict(images):
    ort_inputs = {model.get_inputs()[0].name: images.astype(np.float32)}
    ort_outputs = model.run(None, ort_inputs)
    return ort_outputs
    # print(ort_outputs)

res = predict(rand_images)