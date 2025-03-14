# ComfyUI and Flux on Jetson Orin

## Overview
This guide will walk you through the installation and setup of ComfyUI and Flux on a Jetson Orin device. It includes solutions to common issues that may arise during installation. 
 [The original tutorial can be found here](https://www.jetson-ai-lab.com/tutorial_comfyui_flux.html )


## 1. Install Miniconda and Create a Python 3.10 Environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh
conda update conda
conda create -n comfyui python=3.10
conda activate comfyui
```

## 2. Install CUDA, cuDNN, and TensorRT

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 cuda-compat-12-4
sudo apt-get install cudnn python3-libnvinfer python3-libnvinfer-dev tensorrt
```

## 3. Verify and Configure CUDA

```bash
ls -l /usr/local | grep cuda
sudo ln -s /usr/local/cuda-12.4 /usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
```

## 4. Compile and Install bitsandbytes with CUDA Support

```bash
export BNB_CUDA_VERSION=124
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
mkdir -p build
cd build
cmake .. -DCOMPUTE_BACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4
make -j$(nproc)
cd ..
python setup.py install
```

Verify the installation:
```bash
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```
If you see errors related to missing shared libraries, try rebuilding:
```bash
python setup.py clean --all
python setup.py install
```

## 5. Install PyTorch, TorchVision, and TorchAudio

```bash
pip install http://jetson.webredirect.org/jp6/cu124/+f/5fe/ee5f5d1a75229/torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install http://jetson.webredirect.org/jp6/cu124/+f/988/cb71323efff87/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
pip install http://jetson.webredirect.org/jp6/cu124/+f/0aa/a066463c02b4a/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
```

## 6. Clone and Setup ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
pip install "numpy<2"
```

If you encounter missing module errors, install them as needed:
```bash
pip install pyyaml
pip install requests
pip install Pillow
pip install scipy
pip install attrs
```

If any package installs globally instead of in the Conda environment, force reinstall:
```bash
pip install --force-reinstall <package_name>
```

## 7. Run ComfyUI

```bash
python main.py
```

If you see warnings about an old PyTorch version, update it:
```bash
pip install --upgrade torch
```

## 8. Load the Flux Model

Download the workflow file and model files from Hugging Face and place them in the appropriate folders:
```bash
models/unet/flux1-schnell.safetensors
models/vae/ae.safetensors
models/clip/clip_l.safetensors
models/clip/t5xxl_fp8_e4m3fn.safetensors
```

## 9. Access the Interface
Once running, access the UI via:
```bash
127.0.0.1:8188
```

## 10. Run as a Server
Keep ComfyUI running so you can call it whenever you need. If you just want to run it in the background quickly, you could do:
```bash
nohup env HEADLESS=1 python main.py > comfyui.log 2>&1 &
```

Explanation:
env HEADLESS=1 ensures that the environment variable is correctly set for the python process.
> comfyui.log 2>&1 & redirects both stdout and stderr to comfyui.log so you can check for errors if needed.
To check if it’s running:
```bash
ps aux | grep main.py
```

To stop it if needed:

```bash
pkill -f main.py
```

Enjoy generating images with ComfyUI and Flux on Jetson Orin!
