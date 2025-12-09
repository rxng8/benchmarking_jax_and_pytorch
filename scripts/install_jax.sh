
# Before running this script, you can create a miniconda environment with:
# conda create -n bench python=3.12 -y
# conda activate bench

# Install pytorch for cuda 12.6
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# If using jax to benchmark against pytorch, install jax with:
pip install -U "jax[cuda12]"

# If importing tsxai
pip install ruamel.yaml==0.17.32 optax

# Hydra config. NOTE: hydra-core only supports until python 3.11
# pip install hydra-core --upgrade

# Intall box2d dependencies (requires special handling)
conda install -c conda-forge swig boost-cpp -y

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Then install the requirement file
pip install -r "$SCRIPT_DIR/requirements_jax.txt"

# Install ffmpeg for rendering gifs
conda install -c conda-forge ffmpeg=6.1.1 -y # version 7 does not work


