# AtomFire
Knowledge should be recorded！

## 仓库说明
本仓库的目的是记录大家学习过程中的一些笔记，复现的模型以及一些有用的工具。
仓库结构说明：
1. aspect。切面编程，类似于java springboot中的@注解，非代码侵入式，实现日志、打印网络请求等功能。python装饰器可以实现这一功能
2. doc。文档，API、资料等。可考虑jupyter进行编写
3. example。核心内容，包括各种子学习模块，其中net主要是复现的模型。
4. reference。参考内容，例如复现模型的参考实现。
5. test。测试代码
6. tool。工具。软件、脚本、爬虫等
7. util。与tool不同之处在于，这是代码的util（工具），而不是功能意义上的。

## 配置
### 自动安装
1. 安装anaconda or conda
2. 在仓库根目录下，运行命令：conda env create -f environment.yml
3. 等待较长时间，可能timeout，需要多次运行命令。存在失败可能！

### 手动安装
1. conda create env -n your_name
2. conda activate your_name
3. conda install mamba / pip install mamba。mamba在包管理方面，兼容conda（conda **可直接替换为mamba**），但比conda更快（3-5倍）。
4. mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia。（或许要提高Nvidia驱动程序版本以支持cuda 11.8，使用该命令不需要额外安装cuda 11.8）
5. 哪里错了安哪里。
