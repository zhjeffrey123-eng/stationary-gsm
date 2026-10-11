# 周期金属像素超表面的驻定 GSM 提取

Python代码提供 PEC 像素片的电流网络训练、固定规模训练比较、物理标签重建及 GSM 提取示例。数据为5000个训练图案和100个固定验证图案，网格为 N=24、Q=192、M=1，频率5/7/9 GHz、法向入射、18个入射列。代码和数据均不需要 MATLAB。

版本：[v3.2.0](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/tag/v3.2.0)。代码和训练H5分别下载；H5保留内部gzip压缩，不需要外层ZIP。

训练数据：[training_dataset_5000.h5](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/download/v3.2.0/training_dataset_5000.h5)，1,147,395,483字节，SHA-256：

```text
d31d313533f8c5c3e7b9947a8b2005f9f9afca821a3bd3469cafc5b1914a4d9a
```

下载后放入代码根目录下的 `data/`。发布附件中的 `SHA256SUMS-v3.2.0.txt`同时给出代码ZIP与H5的校验值。

安装 Python 3.12 或以上版本，在代码目录运行：

```bash
python -m pip install -r requirements.txt
python -m stationary_gsm --help
```

将训练数据放到 `data/training_dataset_5000.h5`，或通过 `--data` 指定位置。数据结构与几何规则见 [data/README.md](data/README.md)。软件包不分发论文、图稿、预训练权重、测试集或实验结果；训练时新生成的权重和记录保存在用户指定的新目录。

## 固定训练规模与磁盘缓存

```bash
python -m stationary_gsm train-profile --training-count 600 --hidden 64 --out runs/N600
python -m stationary_gsm train-series --out runs/six_sizes
```

默认预算为64单元、80轮、每轮128批、每批4096个有效电流系数；AdamW、全局梯度范数裁剪5、余弦学习率0.001至0.0001。`train-series`按100/300/600/1000/2000/5000串行启动六个独立 Python 进程，每个进程固定四个线性代数线程。单个训练可用 `--hidden 128`；其他预算参数见命令帮助。

标准128单元网络的实际重训入口为：

```bash
python -m stationary_gsm train-profile --training-count 600 --hidden 128 --out runs/standard128
```

它只使用原600训练图案（H5索引0:600）和固定100验证图案（5000:5100），保存验证最佳模型 `runs/standard128/validation_best.npz`。历史128单元模型由MATLAB `rng(120026)`训练，验证最佳轮数为80；Python重训使用不同的随机流与归约实现，不承诺得到相同历史权重或原图数值。示例提取可使用本地新训练的 `validation_best.npz`，无需历史模型附件。

100和300图案取原600训练图案的固定置换前缀，子集seed为20261011；600及以上取训练池的顺序前缀。所有规模固定使用同一100个验证图案。验证图案只用于验证和模型选择，不参与梯度或归一化尺度计算。参数初始化与批次seed为120026。

电流标签逐拓扑读取到磁盘映射缓存 `active_current_cache.bin`。缓存按有效金属行、频率、入射列存储，再提供原来的频率、有效行、入射列视图；随机批次仍先抽拓扑、再抽有效行、频率和入射列。缓存保留源complex64系数，逐拓扑 RMS 仍按原顺序以双精度归约，避免将全部标签放入私有内存。操作系统可能将文件映射页面计入实际工作集，磁盘缓存不等同于零内存开销。

检查点固定在10/20/40/60/80轮保存，验证最佳模型另存为 `validation_best.npz`。训练损失是按场归一化的系数平方误差；验证电流误差为每个图案、每个频率的全有效行与18列的相对 Frobenius 误差。它们是不同统计量，不能把验证loss当作电流误差中位数。

六规模训练全部完成后，`train-series`自动生成固定验证选择；也可独立运行：

```bash
python -m stationary_gsm select-validation --runs runs/six_sizes
```

选择规则是10轮至80轮验证电流误差中位数下降最多的规模；差距在1e-12内时取较小规模。各检查点的验证统计均为100图案乘3频率，共300条件。该入口只读取运行中新产生的验证摘要，没有测试文件输入。

## 时间和实际内存的边界

资源计量入口要求 Windows，使用 `GetProcessMemoryInfo` 的真实 `PeakWorkingSetSize`，单位为 MiB（2²⁰字节）；每个规模均使用新进程。该值为进程累计峰值，包括解释器、库导入、协议准备、数据读取、文件映射、训练、验证及权重保存。基线工作集在训练H5读取前记录；峰值不由数组大小估算，也不是间歇采样的最大值。

墙钟从训练H5与缓存读取前开始，到检查点权重保存后截取；包括几何编码、所需训练和验证标签读取、磁盘缓存创建与flush、RMS、初始化、训练、计划验证和保存。后续累计时间包含先前常规日志、哈希及CSV写入开销。Python导入、初始split与协议冻结、离线物理标签生成、父进程等待和测试评价均不计入此墙钟。不同硬件、软件和文件系统会产生不同实际成本。

## 用自己的模型提取 GSM

```bash
python -m stationary_gsm demo --model runs/N600/checkpoint_80.npz --out runs/demo --frequency 7 --tolerance 0.1
```

示例现场生成几何并计算 EFIE 参考电流，不读取未分发的测试标签或结果。网络学习先剥离解析入射相位的复电流，输出以所选训练图案的全局 RMS 缩放；预测后恢复相位与尺度。线性映射、驻定二次泛函和功率平衡算子使用同一份预测/校正电流，从而比较提取算子本身。FFT-COCG校正使用真实残差停止条件。多层级联代码保留为标准二端口 GSM 级联，不包含现成多层设计或实验响应。

## 重建扩展物理标签

```bash
python -m stationary_gsm generate-labels --template data/training_dataset_5000.h5 --out regenerated_labels
```

该命令把模板中的原600训练图案和100验证图案的数组原样复制，读取新增4400图案的已知mask，在5/7/9 GHz现场解全部18入射列。每个频率只构建一次完整阻抗矩阵，再对有效金属行切片求解。所有新增条件独立用FFT算子复核逐列相对残差和传播通道功率缺陷，二者均要求≤1e-9；验证在电流转为complex64存储前执行。任一非有限值或数值超限都会停止，不删除或替换几何。

新增几何的生成规则为固定PCG64 seed2026101101、Gaussian平滑、64像素中选前32个金属像素、排除周期角接触，使用D4变换乘64周期平移的规范轨道去重。构造数据时还排除了原900图案和独立测试1000图案的轨道。公开H5中的实际mask是标签重建输入；seed单独不能在缺少那些排除轨道时复刻最终几何列表。重建入口直接使用已给定的mask，不需要测试文件，也不重新随机采样。

物理标签生成的离线成本与网络训练成本分开；该命令会写入自己的标签记录和验证文件，软件包没有预置这些结果。磁盘空间应能容纳源H5、重建H5、三个阻抗缓存和训练所需的有效电流缓存。
