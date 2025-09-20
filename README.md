# Multi-modal Deep Learning for Predicting Functional Outcomes in ICH
Note--The code is not yet fully organized and is currently mainly provided for researchers to reproduce the model, along with example data and sample tables. Once the paper “Multi-modal Deep Learning for Predicting Functional Outcomes in Intracerebral Hemorrhage Using 3D CT and Clinical Data” is officially accepted, we will further refine the reproduction workflow instructions and upload the model weights to support more comprehensive reproduction and application by the community.

# ICHPrognosis: 多模态深度学习模型预测脑出血患者预后

本项目提供了我们研究中使用的 **模型权重文件**，包括：

1. **多模态预后预测模型权重**（基于 3D CT 影像 + 临床文本）  
2. **在我们数据集上训练好的 nnU-Net v2 分割模型权重**  

---

## 🔗 权重下载

所有权重文件均托管于 Google Drive：  

👉 [点击下载模型权重](https://drive.google.com/drive/folders/1pW6QGRM6AF2CuE5ohcE-WfbhgxOBGISZ?usp=drive_link)

---

## 📂 文件说明

- `fold-x_best_model.pth`  
  - 使用 `torch.save(model.state_dict())` 保存的 **PyTorch 模型权重**（`state_dict`）。  
  - 仅包含训练好的参数，不包含模型结构。  

- `nnunetv2_*`  
  - 基于 **nnU-Net v2** 框架，在我们本地数据集上训练得到的分割模型权重。  

---

## 🚀 使用方法

### 1. 加载 PyTorch 模型权重

```python
import torch
from model import MyModel   # 请根据实际模型结构修改

# 定义模型
model = MyModel()

# 加载权重
state_dict = torch.load("fold-1_best_model.pth", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

