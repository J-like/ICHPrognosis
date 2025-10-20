## Multi-modal Deep Learning for Predicting Functional Outcomes in ICH

This repository provides the **model code and pretrained weights** used in our study, including:

1. **Model code**  
2. **nnU-Net v2 segmentation model weights** (trained on our dataset)  
3. **Multi-modal prognostic prediction model weights**  

---

### 🔗 Download Weights

All weights are hosted on Google Drive (primary). If the links fail, you can use the **backup link on Zenodo**.

- **nnU-Net v2 segmentation model weights:**  
👉 [Download nnU-Net v2 weights](https://drive.google.com/drive/folders/1QxuVMiCTDg65k_e30gdFU2hLs5t1VlBP?usp=drive_link)  
👉 **Alternative link:** [Zenodo Link](https://zenodo.org/records/17397901?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0YmVjZTE0LTA0YjctNDFhZS05OTJlLWVkOGE0OGIwZDQ3MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MDA1YmU4MTA1YjcyZjVkZDFhZTNiZWJmNTIyMWM5MiJ9.-BZFLTwHiOOrfT-kw_kSH78FePfrqLpaZSVR7EWLt8SlOiClDTvqRzQ6riKc_AW21KHRKk7OAWeyWDdihYPH_w)  

- **Multi-modal prognostic prediction model weights:**  
👉 [Download multi-modal model weights](https://drive.google.com/drive/folders/15RC24J6VoNf8OmvVwXzMR7nL2SyerJpe?usp=drive_link)  
👉 **Alternative link:** [Zenodo Link](https://zenodo.org/records/17397901?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0YmVjZTE0LTA0YjctNDFhZS05OTJlLWVkOGE0OGIwZDQ3MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MDA1YmU4MTA1YjcyZjVkZDFhZTNiZWJmNTIyMWM5MiJ9.-BZFLTwHiOOrfT-kw_kSH78FePfrqLpaZSVR7EWLt8SlOiClDTvqRzQ6riKc_AW21KHRKk7OAWeyWDdihYPH_w)  

> Note: The BioClinicalBERT pretrained model used in our code is from HuggingFace:  
👉 [BioClinicalBERT on HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

---

### 🚀 Usage

1. Replace the dataset with your own data, and organize it following the directory format in the example files.  

2. **3D CT Skull Stripping with nnU-Net v2 (Optional)**  
   If you want to perform 3D CT skull stripping using nnU-Net v2:  
   - Download our pretrained nnU-Net v2 weights:  
     - [nnU-Net v2 weights](https://drive.google.com/drive/folders/1QxuVMiCTDg65k_e30gdFU2hLs5t1VlBP?usp=drive_link)  
     - **Alternative link:** [Zenodo Link](https://zenodo.org/records/17397901?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0YmVjZTE0LTA0YjctNDFhZS05OTJlLWVkOGE0OGIwZDQ3MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MDA1YmU4MTA1YjcyZjVkZDFhZTNiZWJmNTIyMWM5MiJ9.-BZFLTwHiOOrfT-kw_kSH78FePfrqLpaZSVR7EWLt8SlOiClDTvqRzQ6riKc_AW21KHRKk7OAWeyWDdihYPH_w)  
   - Follow the official nnU-Net v2 documentation for inference/training: [nnU-Net v2 GitHub](https://github.com/MIC-DKFZ/nnUNet)  

3. **Multi-modal Prognostic Prediction Model**  
   - Download the pretrained multi-modal model weights:  
     - [multi-modal weights](https://drive.google.com/drive/folders/15RC24J6VoNf8OmvVwXzMR7nL2SyerJpe?usp=drive_link)  
     - **Alternative link:** [Zenodo Link](https://zenodo.org/records/17397901?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0YmVjZTE0LTA0YjctNDFhZS05OTJlLWVkOGE0OGIwZDQ3MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MDA1YmU4MTA1YjcyZjVkZDFhZTNiZWJmNTIyMWM5MiJ9.-BZFLTwHiOOrfT-kw_kSH78FePfrqLpaZSVR7EWLt8SlOiClDTvqRzQ6riKc_AW21KHRKk7OAWeyWDdihYPH_w)  
   - Start training or inference with:  
     ```bash
     python train.py
     ```  

> ⚠️ **Important:** Although pretrained weights are provided, it is **highly recommended to retrain the models using your own dataset** for optimal performance and generalization.

---

### ⚠️ Notes

- If the weight download links fail, you can download them from the alternative link.
- Due to the heterogeneity of data from different medical centers, we recommend retraining the model on your own dataset.  
- `.pth` files only contain the **model weights**. Ensure that the model definition matches the training phase when loading them.  
- nnU-Net v2 models must follow the official framework’s path conventions.  
- It is recommended to download BioClinicalBERT separately from HuggingFace and load it during runtime.  

---

### 📖 Citation

If you use our multi-modal model, please cite our paper (to be added).  
