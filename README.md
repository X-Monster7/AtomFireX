# AtomFire
Knowledge should be recorded！

## setup

### 自动安装
1. 安装anaconda or conda
2. 在仓库根目录下，运行命令：conda env create -f environment.yml
3. 等待较长时间，可能timeout，需要多次运行命令。存在失败可能！

### 手动安装
1. conda create env -n your_name
2. conda activate your_name
3. conda install mamba / pip install mamba。mamba在包管理方面，兼容conda（conda **可直接替换为mamba**），但比conda更快（3-5倍）。
4. mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia。
5. 哪里错了安哪里。
