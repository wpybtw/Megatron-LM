apt update
apt install -y zsh git tmux curl

sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions


sed -i 's/.*plugins=(git).*/plugins=(git  zsh-autosuggestions)/'  ~/.zshrc

curl -LsSf https://astral.sh/uv/install.sh | sh

export MAX_JOBS=64
export TORCH_CUDA_ARCH_LIST="9.0"
export CMAKE_CUDA_ARCHITECTURES=90

uv pip install nvidia-mathdx pip pybind11

# apex
export NVCC_APPEND_FLAGS="--threads 16"
git clone https://github.com/NVIDIA/apex
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

uv pip install --no-build-isolation "transformer_engine[pytorch]" -v 
uv pip install --no-build-isolation "megatron-core[mlm,dev]" -v 
uv pip install --no-build-isolation -e ".[mlm,dev]" -v 
