import argparse
import os
import string
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from train import Net, ResNet18, Ensemble

parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "Path to directory with image samples.")
args = parser.parse_args()

if args.input:
    path = args.input
    dir_list = os.listdir(path)
    for x in dir_list:
        if x.endswith((".png", ".jpg", ".jpeg")):
            
            initial_labels = list(range(36))
            updated_labels = list(filter(lambda x: x not in (18, 24, 26), initial_labels))
            reverse_mapping = {i: l for (i, l) in enumerate(updated_labels)}
            ascii_mapping = {x: ord(y) for (x, y) in enumerate(string.digits + string.ascii_uppercase) if y not in ('O', 'I', 'Q')}
            
            net = Net()
            resnet = ResNet18()
            model = Ensemble(resnet, net)
            model.load_state_dict(torch.load("model.pth",  map_location='cpu'))

            image = Image.open(os.path.join(path, x))
            image = image.convert("L")
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.rotate(90)
            image = image.resize((28, 28))
            image = ToTensor()(image)
            image = image.unsqueeze(0)

            model.eval()
            with torch.no_grad():
                output = model(image)

            pred = int(torch.max(output, 1)[1])
            reverse_map = reverse_mapping[pred]
            ascii_map = ascii_mapping[reverse_map]

            print(f"{ascii_map},{os.path.join(path, x)}")
