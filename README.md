# 周期金属像素超表面的驻定 GSM 提取

Python代码提供 PEC 像素片的电流网络训练、固定规模训练比较、物理标签重建及 GSM 提取示例。网络包括原 NumPy 多层感知机和周期卷积网络。数据为5000个训练图案和100个固定验证图案，网格为 N=24、Q=192、M=1，频率5/7/9 GHz、法向入射、18个入射列。代码和数据均不需要 MATLAB。

代码版本为3.3.0，增加完整电流场监督的周期卷积训练入口。训练数据沿用已发布的 [v3.2.0](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/tag/v3.2.0) 附件，不重复分发。训练H5保留内部gzip压缩，不需要外层ZIP。代码和文档见[仓库](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl)。

训练数据：[training_dataset_5000.h5](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/download/v3.2.0/training_dataset_5000.h5)，1,147,395,483字节，SHA-256：

```text
d31d313533f8c5c3e7b9947a8b2005f9f9afca821a3bd3469cafc5b1914a4d9a
```

下载后放入代码根目录下的 `data/`。复用H5按上述SHA-256核验，3.3.0代码ZIP按同版本附件 `SHA256SUMS-v3.3.0.txt` 核验。

采用 Python 3.12，在代码目录运行：

```bash
python -m pip install -r requirements.txt
python -m stationary_gsm --help
```

将训练数据放到 `data/training_dataset_5000.h5`，或通过 `--data` 指定位置。数据结构与几何规则见 [data/README.md](data/README.md)。软件包不分发论文、图稿、预训练权重、测试集或实验结果；训练时新生成的权重和记录保存在用户指定的新目录。

## 周期卷积网络与完整电流场监督

`stationary_cnn/` 是训练及预测共用的数值核心。`stationary_gsm/cnn_commands.py` 提供命令行封装，不重复实现损失或相位转换。只有调用卷积命令时才导入 Torch，原 NumPy 提取和训练入口不依赖 Torch。

首先安装上面的基础依赖，然后按自己的运行平台选择一个 Torch 配置。CPU 配置不需要 GPU：

```bash
python -m pip install -r requirements-torch-cpu.txt
```

CUDA 配置采用 Torch 2.7.1、CUDA 11.8，需要兼容的 NVIDIA 驱动：

```bash
python -m pip install -r requirements-torch-cu118.txt
```

无需同时安装两个配置。CPU 与 CUDA 的浮点归约及实际成本可能不同，重训不承诺得到相同历史权重。CUDA 不可用时会明确报错，可改为 `--device cpu`。该入口禁止 TF32，使用 float32 参数、精确 GELU、固定初始化和采样种子，开启确定性算法。它不修改系统 Python、驱动或其他程序。

训练一组600图案：

```bash
python -m stationary_gsm train-cnn --training-count 600 --epochs 48 --checkpoint-epochs 12 24 36 48 --device cuda --out runs/cnn600
```

CPU 入口只需将 `--device cuda` 改为 `--device cpu`。数据可用 `--data` 指定，源文件必须与上面的公开 SHA-256 相符。输出目录必须尚不存在。所有规模取训练分区的顺序前缀，验证固定为 H5 索引5000:5100，验证电流不参与梯度或归一化计算。

完整规模研究使用独立进程串行运行：

```bash
python -m stationary_gsm train-cnn-series --counts 100 300 600 1000 2000 5000 --epochs 48 --checkpoint-epochs 12 24 36 48 --device cuda --out runs/cnn_sizes
```

每个规模建立自己的缓存。程序不会用上一个规模的累计进程峰值，也不会仅让某一个规模承担冷缓存创建成本。训练规模数值必须是正的四倍数，不能超过5000。自定义总轮数时，检查点应落在总轮数范围内，终点总是保存。

### 输入、输出与相位

几何输入为 `mask[y,x]`，形状8×8，金属为1、空气为0。模型将其转成±1，经五层32通道、3×3周期卷积。频率与入射端口输入为 `[(f−7)/2, m, n, pol]`，其中 f 的单位为 GHz，m、n 是 M=1 的 Floquet 阶次，pol 为±1的切向电场分量标识。两层查询网络产生每个卷积层的幅度和偏置调制。模型共49,220个参数。

网络输出36个实数通道，依次包含两个电流分量、实部与虚部、每个8×8像素内部的3×3细网格位置。`indexing.py` 统一通道、空间及端口编号，将输出重排成 N=24 的电流网格。非有效金属电流行硬置零，不把空气像素的输出当作电流未知量。

标签是 η₀J 的复电流。先剥离解析入射相位，再用所选训练图案的全局 RMS 缩放。预测时按相反顺序恢复尺度和相位。两个电流分量采用各自交错采样坐标，相位是 `exp(i2π(mx+ny))`，x、y为相对于周期的坐标。公开电磁模块的有效行编号与 H5 必须一致，训练开始时会逐图案检查。

### 训练目标与误差统计

每次更新包含四幅图案，每幅图案包含三频率、十八端口和全部有效电流行，合计216个完整场。一个 epoch 是全部训练图案的一次随机置换遍历，更新次数为 Ntrain/4。没有从完整矩阵中另外挑选有利的系数。不同规模的一个 epoch 对应的实际更新次数不同，应结合墙钟比较，不能只比较轮数。

训练目标为各图案、各频率的相对电流矩阵误差平方的平均：

```text
L = mean_case( sum_active,port |Jpred − Jref|² / sum_active,port |Jref|² )
```

case 指一幅图案在一个频率下的完整矩阵。分母先由原标签独立计算。每个 case 权重相同，不因某图案具有更多有效行而增加其权重。训练使用 AdamW，β=(0.9,0.999)、ε=1e−8、weight decay=1e−5，梯度范数大于5时裁剪，学习率按固定总轮数从0.001余弦衰减到0.0001。梯度裁剪范数按逐参数 float64 平方和计算。

验证报告的是每个 case 的相对 Frobenius 范数本身，包括均值、中位数、95百分位和最大值。均值为全部300个条件的算术平均，不是训练 loss 的平方根，也不是把所有图案拼成一个大矩阵后的误差。每幅验证图案的三个频率保留在一起，不能把300条件当作300幅独立几何。大误差和非单调点不会被过滤。增加规模或轮数不构成严格单调下降保证。

### 输出文件及成本边界

| 输出 | 含义 |
|---|---|
| checkpoint_12.npz 等 | 计划检查点的网络、缩放、网格、训练规模及协议标识，NumPy安全读取 |
| complete_state_12.pt 等 | 网络、完整Adam矩与步数、初始化/采样/Torch随机状态、轮数及历史 |
| validation_conditions.csv | 每检查点的100图案乘3频率误差，保留全部条件 |
| checkpoint_summary.csv | 验证统计、模型文件、资源及对应成本时间戳文件 |
| training_history.csv | 每轮真实矩阵平方损失、学习率、更新次数与梯度范数 |
| resource_history.csv | 实际操作系统与GPU资源，不能用数组尺寸替代 |
| cost_epoch_12.json 等 | 检查点统一成本边界，学习时间图与终点资源图使用同一来源 |
| training_summary.json | 架构、预算、数据、验证与实际完成状态 |
| demod_current_cache.bin | 逐图案建立的去相位有效电流磁盘映射，运行输出，不随代码发布 |
| validation_reference.bin | 独立保存的原验证电流，避免用训练目标量化值反算验证真值 |

墙钟从本次缓存构建或加载前开始，包含 train-only RMS、初始化、训练、计划验证、权重和完整状态保存，以及常规CSV与摘要同步写入。全部常规输出同步后截取唯一成本时间戳，再将该时间戳写入 `cost_epoch_*.json`。最后这个时间戳自身的写入不计入所记录的墙钟。解释器/库导入、源文件哈希和运行前检查、离线物理标签生成及独立测试评价均排除。终点成本使用与最后检查点相同的时间戳，不能另换较早或较晚的计时口径。

Windows 记录操作系统真实累计 `PeakWorkingSetSize`、当前工作集和私有提交内存。GPU峰值 allocated 与 reserved 单独记录，不能加起来或当作系统RAM。其他平台只记录 `getrusage` 最大RSS，不能解释为 Windows 私有提交内存。内存映射可能被操作系统计入工作集，不等于零内存成本。所有规模应在相同平台运行并报告实际设备、线程及软件版本。

恢复自己已保存的未完成轨迹：

```bash
python -m stationary_gsm train-cnn --training-count 600 --epochs 48 --checkpoint-epochs 12 24 36 48 --device cuda --out runs/cnn600 --resume runs/cnn600/complete_state_24.pt
```

恢复时架构、完整预算、元协议、数据及随机种子必须一致，不是只载入权重重新开始。仅重放置换编号以复核采样摘要，不重算已完成训练。若修改总轮数或学习率，那是新的实验轨迹。正常资源暂停会在完整状态中保留累计成本，不能冒充一次无中断运行。异常中断只能恢复最近已保存的完整检查点，尚未落盘的更新和耗时不能据此重建。

### 用自己的卷积模型提取GSM

```bash
python -m stationary_gsm demo-cnn --model runs/cnn600/checkpoint_48.npz --frequency 7 --tolerance 0.1 --maximum-iterations 512 --device cuda --out runs/cnn_demo
```

例子现场生成环形像素面，计算参考解、预测电流及同一轮物理残差修正，随后用 projection、带权 reciprocal 和 stationary 三种算子提取GSM。三种算子收到相同电流。迭代上限或端口失败都在输出中保留，不能把尚未达到残差容限的条件称为收敛。参考解用于本次示例的误差检验，程序不读取未公开测试集。

也可加 `--data data/training_dataset_5000.h5 --training-index 0` 选择一幅公开训练几何。该选项只读取mask，不读取现成电流或响应，现场重新计算此单频率的参考解。索引只能取0:5000的训练分区。

电流训练误差与GSM提取误差是不同量。驻定修正抑制固定离散方程的首阶输出误差，但不自动保证任何原始网络电流都足够准确，不保证连续物理误差，也不直接给出筛选保优或总体加速结论。训练时间、在线预测、物理修正和GSM提取成本应分别记录。

## 原 NumPy 网络与磁盘缓存

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

示例现场生成几何并计算 EFIE 参考电流，不读取未分发的测试标签或结果。网络学习先剥离解析入射相位的复电流，输出以所选训练图案的全局 RMS 缩放；预测后恢复相位与尺度。线性映射、带权互易对称化和驻定二次泛函使用同一份预测/校正电流，从而比较提取算子本身。FFT-COCG校正使用真实残差停止条件。多层级联代码保留为标准二端口 GSM 级联，不包含现成多层设计或实验响应。

## 重建扩展物理标签

```bash
python -m stationary_gsm generate-labels --template data/training_dataset_5000.h5 --out regenerated_labels
```

该命令把模板中的原600训练图案和100验证图案的数组原样复制，读取新增4400图案的已知mask，在5/7/9 GHz现场解全部18入射列。每个频率只构建一次完整阻抗矩阵，再对有效金属行切片求解。所有新增条件独立用FFT算子复核逐列相对残差和传播通道功率缺陷，二者均要求≤1e-9；验证在电流转为complex64存储前执行。任一非有限值或数值超限都会停止，不删除或替换几何。

新增几何的生成规则为固定PCG64 seed2026101101、Gaussian平滑、64像素中选前32个金属像素、排除周期角接触，使用D4变换乘64周期平移的规范轨道去重。构造数据时还排除了原900图案和独立测试1000图案的轨道。公开H5中的实际mask是标签重建输入；seed单独不能在缺少那些排除轨道时复刻最终几何列表。重建入口直接使用已给定的mask，不需要测试文件，也不重新随机采样。

物理标签生成的离线成本与网络训练成本分开；该命令会写入自己的标签记录和验证文件，软件包没有预置这些结果。磁盘空间应能容纳源H5、重建H5、三个阻抗缓存和训练所需的有效电流缓存。
