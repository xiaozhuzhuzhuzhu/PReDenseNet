import torchvision
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from PReDenseNet import PRDenseNet_gru


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
# 获取要测试的雨图
image_path = "./real1.png"
image = Image.open(image_path)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((175, 233)), torchvision.transforms.ToTensor()])
image = transform(image)

# 获取保存的模型
checkpoint = torch.load("./model_epoch100.pth")

model = PRDenseNet_gru(recurrent_iter=5).to(device)
model.load_state_dict(checkpoint["net"])
print(model)
model.eval()

image = torch.reshape(image, (1, 3, 175, 233))
image = image.to(device)
with torch.no_grad():
    output = model(image)


output = torch.reshape(output, (3, 175, 233))
print(output)
show = ToPILImage()
show(output).show()
