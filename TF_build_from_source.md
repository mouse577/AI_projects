📜 Full Detailed Markdown (what worked)
File: tensorflow_build_guide.md

markdown
Copy
Edit
# 🚀 Custom TensorFlow 2.13.1 Build Guide (GPU Acceleration, AVX, Ubuntu 24.04)

---

## 1. 📦 Environment Setup

```bash
conda create -n tf_build_env_py39 python=3.9 -y
conda activate tf_build_env_py39
pip install numpy==1.21.6 wheel setuptools keras_preprocessing opt_einsum
2. 🛠 System Dependencies
bash
Copy
Edit
sudo apt update
sudo apt install -y build-essential gcc-11 g++-11 openjdk-11-jdk zip unzip git curl patchelf
3. 🛠 Install Bazel 5.3.0
bash
Copy
Edit
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" > /etc/apt/sources.list.d/bazel.list'
sudo apt update
sudo apt install bazel-5.3.0
sudo apt-mark hold bazel
4. 🚀 Clone TensorFlow Source Code
bash
Copy
Edit
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r2.13
5. ⚙️ Configure TensorFlow
bash
Copy
Edit
./configure
ROCm support: n

CUDA support: y

TensorRT support: n

Compute capability: 8.6

Android build: n

6. ✏️ Modify Compiler Settings
bash
Copy
Edit
nano ~/tensorflow/.tf_configure.bazelrc
Change:

diff
Copy
Edit
--action_env GCC_HOST_COMPILER_PATH=/usr/bin/x86_64-linux-gnu-gcc-13
To:

diff
Copy
Edit
--action_env GCC_HOST_COMPILER_PATH=/usr/bin/gcc-11
7. 🌟 Export Build Environment Variables
bash
Copy
Edit
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc-11
export CC_OPT_FLAGS="-march=native -w"
export TF_CUDA_FLAGS="--allow-unsupported-compiler"
export TF_NVCC_FLAGS="-allow-unsupported-compiler"
8. 🛠 Build TensorFlow
bash
Copy
Edit
bazel clean
bazel build --config=opt --config=cuda --jobs=8 //tensorflow/tools/pip_package:build_pip_package
9. 📦 Create TensorFlow Wheel Package
bash
Copy
Edit
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
10. 📥 Install TensorFlow Wheel
bash
Copy
Edit
pip install /tmp/tensorflow_pkg/tensorflow-2.13.1-*.whl
11. ✅ Verify Installation
python
Copy
Edit
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
🎯 Final Summary

Component	Version
TensorFlow	2.13.1
CUDA	12.1
cuDNN	8.x
Python	3.9
Bazel	5.3.0
GCC	11
GPU Acceleration	✅ Enabled
AVX, AVX2	✅ Enabled
XLA	✅ Enabled
bash
Copy
Edit

---

### ⚡ Short Ultra-Fast Reinstallation Guide (lightning-fast commands)

**File:** `ultra_fast_tf_reinstall.md`

```markdown
# ⚡ Ultra-Fast TensorFlow Rebuild Guide

```bash
conda activate tf_build_env_py39
cd ~/tensorflow

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc-11
export CC_OPT_FLAGS="-march=native -w"
export TF_CUDA_FLAGS="--allow-unsupported-compiler"
export TF_NVCC_FLAGS="-allow-unsupported-compiler"

bazel clean
bazel build --config=opt --config=cuda --jobs=8 //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-2.13.1-*.whl
✅ Done!

yaml
Copy
Edit



🚀 DeepLabCut 2.3.8 GPU + GUI Installation & Launch Cheat Sheet
🛠️ Step 1: Install System Libraries
Install OS-level libraries needed for wxPython and DLC GUI:

bash
Copy
Edit
sudo apt update
sudo apt install -y \
    libgtk-3-dev libjpeg-dev libtiff-dev libsdl2-dev libnotify-dev \
    freeglut3-dev libsm-dev libxtst-dev libwebkit2gtk-4.1-dev
(note: libwebkit2gtk-4.1-dev is for Ubuntu 24.04)

🛠️ Step 2: Create Conda Environment
bash
Copy
Edit
conda create -n tf_build_env_py39 python=3.9 anaconda
conda activate tf_build_env_py39
🛠️ Step 3: Install DeepLabCut with GUI support
bash
Copy
Edit
pip install "deeplabcut[gui]==2.3.8"
If problems occur, force reinstall:

bash
Copy
Edit
pip install --force-reinstall "deeplabcut[gui]==2.3.8"
🛠️ Step 4: Handle Typing Extensions Warning (optional)
If TensorFlow warns:

bash
Copy
Edit
pip install "typing-extensions<4.6.0"
(Usually not needed unless problems arise.)

🛠️ Step 5: Always Set Environment Variables Before Running
Every new terminal session:

bash
Copy
Edit
conda activate tf_build_env_py39
export DLCLIGHT=0
python
Then inside Python:

python
Copy
Edit
import deeplabcut
✅ DLC will load in full GUI mode (not light mode).

🧹 Quick Project Workflow Cheat Sheet
Create a New Project
python
Copy
Edit
import deeplabcut

deeplabcut.create_new_project(
    'DummyProject', 'John',
    ['/path/to/your/video1.mp4', '/path/to/your/video2.mp4'],
    working_directory='/home/jg/DLC_dummy',
    copy_videos=True
)
✅ After creation, edit your config.yaml:

Set bodyparts:

yaml
Copy
Edit
bodyparts:
  - center
  - left_side
  - right_side
Delete the line for objectA.

Skeleton section can be updated later.

Add More Videos
python
Copy
Edit
import glob

video_list = glob.glob('/home/jg/DLC_dummy/DummyProject-John-2025-04-27/videos/*.mp4')

deeplabcut.add_new_videos(
    '/home/jg/DLC_dummy/DummyProject-John-2025-04-27/config.yaml',
    video_list,
    copy_videos=False
)
Extract Frames
python
Copy
Edit
deeplabcut.extract_frames(
    '/home/jg/DLC_dummy/DummyProject-John-2025-04-27/config.yaml',
    mode='kmeans',
    algo='kmeans'
)
Label Frames with GUI
python
Copy
Edit
deeplabcut.label_frames('/home/jg/DLC_dummy/DummyProject-John-2025-04-27/config.yaml')
🎯 After Reboot / Terminal Restart
Each time:

bash
Copy
Edit
conda activate tf_build_env_py39
export DLCLIGHT=0
python
Then inside Python:

python
Copy
Edit
import deeplabcut
🧠 Notes
Always export DLCLIGHT=0 before running Python

TensorFlow 2.13.1 (custom built) GPU accelerated (RTX 3090, CUDA 12.1 confirmed)

wxPython 4.2.1 correctly installed for GUI

If anything weird happens:

bash
Copy
Edit
pip uninstall deeplabcut -y
pip install "deeplabcut[gui]==2.3.8"
✅ You are now ready for labeling, training, and analyzing on full GPU with GUI support!


