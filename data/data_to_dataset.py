import torch
import numpy as np
import torch.utils.data as data
import SimpleITK as sitk
from torchvision import transforms
from transformers import AutoTokenizer

class dataset3D(torch.utils.data.Dataset):
    def __init__(self, patients_info):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") # Bio_ClinicalBERT
        self.data_list = []
        for patient_name, info in patients_info.items():
            img_path = info['path']  
            text_info = info['text']  
            label = info['label']  
            self.data_list.append([img_path, text_info, label])
            
    def min_max_normalization(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor
    
    def __getitem__(self, index):
        img_path, text_tensor, label = self.data_list[index]
        img_name = img_path.split('/')[2]

        get_img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(get_img).astype(np.float32)

        def adjust_window(image, window_width=100, window_level=50):
            lower_bound = window_level - (window_width / 2)
            upper_bound = window_level + (window_width / 2)
            image = np.clip(image, lower_bound, upper_bound)
            image = (image - lower_bound) / (upper_bound - lower_bound)
            image = (image * 255).astype(np.float32)
            return image

        img = adjust_window(img, window_width=100, window_level=50)

        target_size = (128, 512, 512)
        depth, height, width = img.shape
        target_depth, target_height, target_width = target_size

        if depth < target_depth:
            padding = (target_depth - depth) // 2
            img = np.pad(img, ((padding, target_depth - depth - padding), (0, 0), (0, 0)), mode='constant')
        else:
            start = (depth - target_depth) // 2
            img = img[start:start + target_depth, :, :]
        
        if height < target_height:
            padding = (target_height - height) // 2
            img = np.pad(img, ((0, 0), (padding, target_height - height - padding), (0, 0)), mode='constant')
        else:
            start = (height - target_height) // 2
            img = img[:, start:start + target_height, :]
        
        if width < target_width:
            padding = (target_width - width) // 2
            img = np.pad(img, ((0, 0), (0, 0), (padding, target_width - width - padding)), mode='constant')
        else:
            start = (width - target_width) // 2
            img = img[:, :, start:start + target_width]

        img_tensor = img[np.newaxis, :, :, :]
        img_tensor = torch.from_numpy(img_tensor).float()
        
        text_tensor = self.tokenizer(text_tensor, padding='max_length', max_length=64, truncation=True, return_tensors="pt")        
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, text_tensor, label_tensor, img_name
        
    def __len__(self):
        return len(self.data_list)

    def build_index(self):
        self.name_to_index = {self.data_list[i][0].split('/')[1]: i for i in range(len(self.data_list))}
