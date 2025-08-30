import openpyxl
import pandas as pd
import os
import re
import torch
import torch.nn as nn
from collections import Counter
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def extract_number(patient_id):
    match = re.search(r'-(\d+)$', patient_id)
    if match:
        return int(match.group(1))
    else:
        return 0
import glob

def screch_excel(file_path):
    df = pd.read_excel(file_path)
    df['new_name'] = df['new_name'].astype(str).str.strip()
    df_dict = df.set_index('new_name').T.to_dict('list')

    patient_files = [f for f in os.listdir('./dataset')]

    patient_remove_list = []
    data_dict = {}  
    
    table_names = ['Gender', 'Age', 'High blood pressure', 'Diabetes', 'Smoke', 'Alcoholic',
               'Time of onset', 'GCS','NIHSS']

    for patient_file in patient_files:
        patient_name = patient_file.split('-')[0]
        if patient_name in patient_remove_list:
            continue
        lever = extract_number(patient_file)
        if patient_name in df_dict:
            row = df_dict[patient_name]
            text = row[1:]
            
            gender = "male" if text[0] == 1.0 else "female"
            age = int(text[1])
            hbp = "with" if text[2] == 1.0 else "without"
            diabetes = "with" if text[3] == 1.0 else "without"
            smoke = "a smoker" if text[4] == 1.0 else "a non-smoker"
            alcohol = "an alcohol consumer" if text[5] == 1.0 else "not an alcohol consumer"
            onset_time = int(text[6])
            gcs = int(text[7])
            nihss = int(text[8])
            
            
            # The user can modify the wording as needed.
            words = (
                f"This {age}-year-old {gender} presents {hbp} high blood pressure and {diabetes} diabetes. "
                f"The patient is {smoke} and {alcohol}. "
                f"Symptoms began {onset_time} hours prior, "
                f"with a Glasgow Coma Scale (GCS) score of {gcs} and an NIHSS score of {nihss}."
            )
            
            path = f'./dataset/{patient_file}/{patient_file}.nii.gz'

            data_dict[patient_file] = {
                'label': row[0],
                'text': words,
                'path': path,
            }
    # print(data_dict)
    return data_dict

