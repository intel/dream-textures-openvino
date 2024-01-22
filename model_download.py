from huggingface_hub import snapshot_download
import os
import sys
import shutil
from pathlib import Path

install_location = os.path.join(os.path.expanduser("~"), "Blender_SD_models")

access_token  = None


def download_hf_model(repo_id, model_fp16,model_int8):

    while True:
        try:
            download_folder = snapshot_download(repo_id=repo_id, token=access_token)
            break
        except Exception as e:
             print("Error retry:" + str(e))
        
    SD_path_FP16 = os.path.join(install_location, model_fp16, "model_size_512")
    SD_path_INT8 = os.path.join(install_location, model_int8, "model_size_512")




    if os.path.isdir(SD_path_FP16):
            shutil.rmtree(SD_path_FP16)

    if os.path.isdir(SD_path_INT8):
        shutil.rmtree(SD_path_INT8)


    FP16_model = os.path.join(download_folder, "FP16")
    INT8_model = os.path.join(download_folder, "INT8")
    print("download_folder", download_folder)
    shutil.copytree(download_folder, SD_path_FP16, ignore=shutil.ignore_patterns('FP16', 'INT8'))
    shutil.copytree(download_folder, SD_path_INT8, ignore=shutil.ignore_patterns('FP16', 'INT8'))
    shutil.copytree(FP16_model, SD_path_FP16, dirs_exist_ok=True)
    shutil.copytree(INT8_model, SD_path_INT8, dirs_exist_ok=True)
    delete_folder=os.path.join(download_folder, "../../..")
    shutil.rmtree(delete_folder, ignore_errors=True)



while True:

    print("=========Chose SD-1.5 models to download =========")
    print("1 - Stable-diffusion-1.5-quantized")
    print("2 - Stable_Diffusion_1_5_controlnet_depth")
    print("3 - All the above models")
    print("4 - Skip All SD-1.5 Model setup")
    choice = input("Enter the Number for the model you want to download: ")

    if choice=="1":
        print("Downloading Intel/sd-1.5-square-quantized Models")

        repo_id="Intel/sd-1.5-square-quantized"
        model_fp16 = "Stable_Diffusion_1_5"
        model_int8 = "Stable_Diffusion_1_5_int8"
        download_hf_model(repo_id, model_fp16,model_int8)

    elif choice=="2":
        print("Downloading Intel/sd-1.5-controlnet-depth-quantized Models")

        repo_id="Intel/sd-1.5-controlnet-depth-quantized"
        model_fp16 = "Stable_Diffusion_1_5_controlnet_depth"
        model_int8 = "Stable_Diffusion_1_5_controlnet_depth_int8"
        download_hf_model(repo_id, model_fp16,model_int8)

    elif choice=="3":
         print("Downloading all the models")
         download_hf_model("Intel/sd-1.5-square-quantized", "Stable_Diffusion_1_5","Stable_Diffusion_1_5_int8")
         download_hf_model("Intel/sd-1.5-controlnet-depth-quantized", "Stable_Diffusion_1_5_controlnet_depth","Stable_Diffusion_1_5_controlnet_depth_int8")
         break
    
    elif choice=="4":
        print("Exiting SD-1.5 Model setup.........")
        break


         
         








