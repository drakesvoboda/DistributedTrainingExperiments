wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $HOME/anaconda3
./anaconda3/bin/conda init bash
. ~/.bashrc
conda create -n pytorch_env -y pytorch torchvision torchaudio numpy matplotlib tqdm -c pytorch
conda activate pytorch_env
yes | pip install sklearn psrecord future transformers
conda install -y -c conda-forge tensorboard
git clone https://github.com/drakesvoboda/eecs598/ 
cd eecs598/Assignment/
export GLOO_SOCKET_IFNAME=eno1d1
export TP_SOCKET_IFNAME=eno1d1
export NCCL_SOCKET_IFNAME=eno1d1