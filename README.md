## Multi-modal Deep Learning for Predicting Functional Outcomes in ICH

This repository provides the **model code and pretrained weights** used in our study, including:

1. **Model code**  
2. **nnU-Net v2 segmentation model weights** (trained on our dataset)  
3. **Multi-modal prognostic prediction model weights**  

---

### 🔗 Download Weights

All weights are hosted on Google Drive:  

👉 [Download model weights](https://drive.google.com/drive/folders/1pW6QGRM6AF2CuE5ohcE-WfbhgxOBGISZ?usp=drive_link)

Note: The BioClinicalBERT pretrained model used in our code is from HuggingFace:  
👉 [BioClinicalBERT on HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

---

### 🚀 Usage

1. Replace the dataset with your own data, and organize it following the directory format in the example files.  
2. *(Optional)* If you need to perform 3D CT skull stripping using nnU-Net v2, you can download our pretrained nnU-Net v2 segmentation weights. For further steps, please refer to the official documentation: [nnU-Net v2 GitHub](https://github.com/MIC-DKFZ/nnUNet).  
3. Start training with the following command:  
    ```bash
    python train.py
    ```  

---

### ⚠️ Notes

- Due to the heterogeneity of data from different medical centers, we recommend retraining the model on your own dataset.  
- `.pth` files only contain the **model weights**. Please ensure that the model definition matches the training phase when loading them.  
- nnU-Net v2 models must follow the official framework’s path conventions.  
- BioClinicalBERT should be downloaded separately from HuggingFace and loaded during runtime.  

---

### 📖 Citation

If you use our multi-modal model, please cite our paper (to be added).  
