# OpenVINO™ support for Dream Textures addon in Blender

* Create textures, concept art, background assets, and more with a simple text prompt
* Texture entire scenes with 'Project Dream Texture' and depth to image
* Run the models locally on your Intel system and ability to choose CPU,iGPU,dGPU and/or NPU to offload the models. 


# Installation

### A. Install OpenVINO™
- Download and install [OpenVINO™](https://github.com/openvinotoolkit/openvino/releases) for your operating system.
- Note that this addon has been tested with 2023.1.0 
- For Intel&reg; Core™ Ultra support, you need to download and install OpenVINO™ from the archive.

### B. Dream-Texture-Openvino Install
Skip steps 1 and 2 if you already have Python3 and Git on Windows

#### 1. Install Python
- Download a Python installer from python.org. Choose Python 3.10 and make sure to pick a 64 bit version. For example, this 3.10.11 installer: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe <br>
- Double click on the installer to run it, and follow the steps in the installer. Check the box to add Python to your PATH, and to install py. At the end of the installer, there is an option to disable the PATH length limit. It is recommended to click this. <br>

#### 2. Install Git
- Download and install [GIT](https://git-scm.com/)

#### 3. Install the MSFT Visual C++ Runtime
- Download and install [the latest supported redist](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

#### 4. Install Blender
- Install [Blender-3.4](https://mirrors.dotsrc.org/blender/release/Blender3.4/blender-3.4.0-windows-x64.msi)

#### 5. Install Dream-texture-openvino addon

1. Clone this repository: <br>
    ```
    git clone https://github.com/intel/dream-textures-openvino.git dream_textures
    ``` 
2. Copy this folder - "dream_textures" into your Blender addons folder -  ```C:\Program Files\Blender Foundation\Blender 3.4\3.4\scripts\addons``` <br>
3. If you can't find the add-on folder, you can look at another third-party add-on you already have in Blender preferences and see where it is located.<br>
4. Navigate to the dream-texture folder - <br>
    ```
    cd C:\Program Files\Blender Foundation\Blender 3.4\3.4\scripts\addons\dream_textures
    ``` 
5. Install all the required python packages and download the models from Hugging face. <br>
   All of the packages are installed to ```dream_textures\.python_dependencies```. <br>
   The following commands assume they are being run from inside the dream_textures folder. <br>
   ```
   install.bat
   ```
6. Replace "openvino" folder in ```C:\Program Files\Blender Foundation\Blender 3.4\3.4\scripts\addons\dream_textures\.python_dependencies``` with the "openvino" folder present in your openvino_2023.1\python\openvino (from section A) <br>

7. If you just want to download models - <br>
    ```
    "C:\Program Files\Blender Foundation\Blender 3.4\3.4\python\bin\python.exe" model_download.py
    ```
8. Setup OpenVINO™ Environment <br>
   <b>Note that you will need to do these steps everytime you start Blender</b>
   ```
   C:\Path\to\where\you\installed\OpenVINO\setupvars.bat
   ```
7. Start Blender application, and go to Edit->Preferences->Add-on and search for dream texture and enable it. 
   ```
   "C:\Program Files\Blender Foundation\Blender 3.4\blender.exe"
   ```



# Usage

Here's a few quick guides:


## [Image Generation](docs/IMAGE_GENERATION.md)
Create textures, concept art, and more with text prompts. Learn how to use the various configuration options to get exactly what you're looking for.



## [Texture Projection](docs/TEXTURE_PROJECTION.md)
Texture entire models and scenes with depth to image.



## [History](docs/HISTORY.md)
Recall, export, and import history entries for later use.



# Acknowledgements
* Plugin architecture heavily inspired from dream-textures project by carson-katri - https://github.com/carson-katri/dream-textures


# Disclaimer
Stable Diffusion’s data model is governed by the Creative ML Open Rail M license, which is not an open source license.
https://github.com/CompVis/stable-diffusion. Users are responsible for their own assessment whether their proposed use of the project code and model would be governed by and permissible under this license.
