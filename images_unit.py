import os
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms

# mean and std of ImageNet to use pre-trained VGG
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                 std=IMAGENET_STD)

denormalize = transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/std for std in IMAGENET_STD])

unloader = transforms.ToPILImage()

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_path, transform):
        super(ImageFolder, self).__init__()
        
        self.file_names = sorted(os.listdir(root_path ))
        self.root_path = root_path        
        self.transform = transform
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
        return self.transform(image)
    
def get_transformer(imsize=None, cropsize=None):
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        transformer.append(transforms.RandomCrop(cropsize)),
    transformer.append(transforms.ToTensor())
    transformer.append(normalize)
    return transforms.Compose(transformer)

def imsave(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)    
    torchvision.utils.save_image(tensor, path)
    return None
    
def imload(path, imsize=None, cropsize=None):
    transformer = get_transformer(imsize, cropsize)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)

def postprocess_image(tensor, imsize=None, cropsize=None):
    
    # ImageNet mean và std
    mean = torch.tensor(IMAGENET_MEAN)
    std = torch.tensor(IMAGENET_STD)

    # Giải chuẩn hóa (Unnormalize)
    tensor = tensor.squeeze(0).cpu()  # Loại bỏ batch dimension và chuyển về CPU
    unnormalized = tensor * std[:, None, None] + mean[:, None, None]  # Giải chuẩn hóa

    # Đảm bảo giá trị nằm trong khoảng [0, 1]
    unnormalized = torch.clamp(unnormalized, 0, 1)

    # Chuyển từ Tensor sang NumPy
    output_image = unnormalized.numpy().transpose(1, 2, 0)  # Chuyển kênh (C, H, W) -> (H, W, C)
    output_image = (output_image * 255).astype('uint8')  # Chuyển về [0, 255]

    # Chuyển đổi từ RGB sang BGR để OpenCV hiển thị
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    return output_image

def imload_realtime(frame, imsize=None, cropsize=None):
    if isinstance(frame, np.ndarray):
        # Chuyển đổi frame từ BGR (OpenCV) sang RGB (PIL yêu cầu)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Chuyển đổi numpy.ndarray thành PIL.Image
        frame = Image.fromarray(frame)
    else:
        # Nếu là đường dẫn file, mở file như bình thường
        frame = Image.open(frame).convert("RGB")
    # Chuyển đổi kích thước ảnh và trả về tensor
    transformer = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transformer(frame).unsqueeze(0)