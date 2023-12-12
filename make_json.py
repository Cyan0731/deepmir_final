import librosa
from glob import glob
import numpy as np
import os
from tqdm import tqdm
import soundfile as sf
import json

# change the params here===============
SR = 32000
accom_files = glob("/home/cyan/workstation/DL_music/final/vocal_dataset_syn/*/ACCOM_MixDown.wav")
accom_files.sort()

output_dir = "/home/cyan/workstation/DL_music/final/audiocraft/dataset/acca_32k/accom/"
# ======================================


for accom_file in tqdm(accom_files):
    file_name = accom_file.split('/')[-2]
    accom_output = os.path.join(output_dir, file_name+'.wav')
    vocal_output = accom_output.replace("accom", "vocal")
    
    vocal_file = accom_file.replace("ACCOM_MixDown", "lead_vocal_MixDown")
    if os.path.exists(accom_file.replace("ACCOM_MixDown", "accom")):
        accom_file = accom_file.replace("ACCOM_MixDown", "accom")

    accom, sr = librosa.load(accom_file, sr=SR)
    vocal, sr = librosa.load(vocal_file, sr=SR)
    len_diff = accom.shape[0] - vocal.shape[0]

    if len_diff >= 0:
        vocal = np.pad(vocal, (0,len_diff), 'constant', constant_values=0)
    else:
        vocal = vocal[:len_diff]

    # check silence
    out_utter = np.mean(np.abs(accom) > 1e-4)
    cond_utter = np.mean(np.abs(vocal) > 1e-4)
    print(file_name, out_utter, cond_utter)

    if out_utter < 0.5 or cond_utter < 0.5:
        continue
    else:
        if not os.path.exists(accom_output):
            sf.write(accom_output, accom, SR)
        if not os.path.exists(vocal_output):
            sf.write(vocal_output, vocal, SR)

    accom_json = {"key": "", 
            "artist": "", 
            "sample_rate": SR, 
            "file_extension": "wav", 
            "description": "", 
            "keywords": "", 
            "duration": librosa.get_duration(y=accom, sr=SR), 
            "bpm": "", 
            "genre": "", 
            "title": "", 
            "name": "", 
            "instrument": "", 
            "moods": ""}
    
    with open(accom_output.replace('wav', 'json'),'w') as f:
        json.dump(accom_json, f, indent=4)

    

