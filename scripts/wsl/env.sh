# GPU TensorFlow environment for the traffic project (WSL)
NV=/root/tfgpu/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')/usr/lib/wsl/lib
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONIOENCODING=utf-8
export TF_FORCE_GPU_ALLOW_GROWTH=true
