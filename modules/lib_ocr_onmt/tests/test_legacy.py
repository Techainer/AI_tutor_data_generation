import torch

device = "cuda"
weight_path = "models/card_ocr/1/all_data_step_2572000_scripted.pt"
model = torch.jit.load(weight_path)
model.to(device)

def predict(images):
   output = model(images.to(device))

is_continue = True
batch_size = 1

while is_continue:
    rand_images = torch.rand(batch_size, 3, 64, 224)
    predict(rand_images)
    

print("DONE!")