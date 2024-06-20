conda create --name surfel-splatting python=3.8.18
conda activate surfel-splatting
conda install ffmpeg=4.2.2
conda install typing_extensions=4.9.0
module load cuda/12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
