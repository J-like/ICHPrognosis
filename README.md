## Multi-modal Deep Learning for Predicting Functional Outcomes in ICH

本项目提供了我们研究中使用的 **模型代码与权重文件**，包括：

1. **模型代码**
2. **在我们数据集上训练好的 nnU-Net v2 分割模型权重**  
3. **多模态预后预测模型权重**  

---

### 🔗 权重下载

所有权重文件均托管于 Google Drive：  

👉 [点击下载模型权重](https://drive.google.com/drive/folders/1pW6QGRM6AF2CuE5ohcE-WfbhgxOBGISZ?usp=drive_link)

注意，我们代码中的BioClinicalBERT 预训练模型来自 HuggingFace：  
👉 [BioClinicalBERT on HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

---

### 🚀 使用方法

1. 将数据替换为自己的数据，并依据示例文件的目录格式进行存放。
2. （可选）若需使用 nnU-Net v2进行3D CT 颅骨剥离，可以下载我们已经训练好的nnU-Net v2 分割模型权重，之后的操作请参考其官方文档：[nnU-Net v2 GitHub](https://github.com/MIC-DKFZ/nnUNet)。  
3. 运行以下命令即可开始训练：  
    ```bash
    python train.py
    ```  
---

### ⚠️ 注意事项

- 由于不同中心的数据具有异质性，建议研究者利用个人的数据进行重新训练模型。
- `.pth` 文件仅包含 **模型权重**，请确保在加载时定义与训练阶段一致的模型结构。  
- nnU-Net v2 模型的使用需遵循官方框架的路径规范。  
- 建议 BioClinicalBERT 从 HuggingFace 单独下载，并在运行时加载。  

---

### 📖 引用

如果您使用了我们的多模态模型，请引用我们的论文（待补充）。  


