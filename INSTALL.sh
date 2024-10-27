git submodule update --init --recursive
ln -s /home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data_cvpr data
ln -s /home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data_cvpr data_cvpr
conda create --name surfel_splatting_compared python=3.8.18
conda activate surfel_splatting_compared
conda install ffmpeg=4.2.2
conda install typing_extensions=4.9.0
module load cuda/12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-surfel-rasterization # pip install submodules_gpu3/simple-knn
pip install submodules/simple-knn # pip install submodules_gpu3/simple-knn
