# Stationary GSM extraction — v3.1.0

本仓库仅提供 Python 数值代码和固定训练/验证标签。代码为周期20 mm的8×8零厚度PEC像素面实现谱域EFIE、电流网络、COCG修正、同电流三种GSM提取及多层GSM连接。中文注释说明入射相位、RMS缩放和互易普通转置。

下载本版代码附件及 [v3.0.0 固定训练附件](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/download/v3.0.0/stationary-gsm-training-v3.0.0.zip)，解压到同一目录。训练附件产生 `data/training.h5`，含600个训练图案（split=1）和100个验证图案（split=2），不含200个保留测试图案。N24/Q192/M1、5/7/9 GHz、法向入射、18端口。所有保留标签数组与原数据逐元素一致；验证集不参与梯度。本版复用这份固定附件，不重新上传或增加图案。

Python 3.12，依赖采用本次实际检查的版本：

```bash
python -m pip install -r requirements.txt
python -m stationary_gsm train --hidden 128 --out results/train128
python -m stationary_gsm train --hidden 64 --out results/train64
python -m stationary_gsm demo --model results/train128/model.npz --out results/example
```

训练默认80 epochs、每轮128 batches、每batch 4096个有效电流样本、seed120026。监督只用相对电流MSE；训练集全局RMS缩放标签，移除解析入射相位，再以验证电流MSE选择 `model.npz`。默认另保存第10轮 `checkpoint_10.npz`，可用 `--epochs` / `--checkpoint-epochs` 更改预算。64和128单元共享同一训练实现，参数只改变两隐藏层宽度。输出目录必须尚不存在；预训练权重不分发，新训练模型不承诺逐位重现原权重。

`demo`用刚训练的模型和代码现场生成的环形像素面计算参考EFIE解、预测电流、COCG修正，再从同一组电流提取projection/reciprocal/stationary GSM。它将本次生成的数组及误差写入新目录，记录失败或未收敛端口；不读取论文测试集或已归档结果。网络编码仅适用M=1和法向入射，数值提取代码也支持成对斜入射算子。多层接口见 `stationary_gsm/stack.py` 的 `cascade` 与 `coupled`。

Windows 可运行带实际资源计量的固定子集训练。以下每条命令都启动一个新Python进程，按顺序执行：

```bash
python -m stationary_gsm train-profile --training-count 100 --out results/profile100
python -m stationary_gsm train-profile --training-count 300 --out results/profile300
python -m stationary_gsm train-profile --training-count 600 --out results/profile600
```

`train-profile`默认64隐藏单元、80 epochs、每轮128 batches、每batch 4096样本、固定4线程。subset seed20261011确定600训练图案的置换，100/300为其嵌套前缀；600使用原顺序。同100验证图案始终保留，RMS只由选中训练子集计算。train seed120026、GELU、AdamW和余弦学习率与数值训练实现相同。固定保存第10/20/40/60/80轮 `checkpoint_*.npz`；另存验证电流MSE最佳 `validation_best.npz`。固定检查点不由验证或测试表现替换。

每epoch现场生成 `resource_history.csv`、`training_history.csv`，固定检查点另记录 `checkpoints.csv`。`training_wall_seconds`是秒，从打开训练H5之前到该检查点权重保存之后累计，包含标签/有效电流缓存、几何/RMS、初始化、训练、定期验证、检查点写盘及之前常规日志/哈希/CSV开销；不含Python imports或测试评价。内存是Windows `GetProcessMemoryInfo` 的真实 `PeakWorkingSetSize`：该独立进程生命周期累计峰值，包含解释器/库、缓存、训练/验证和写盘；同时记录打开H5前的基线 `WorkingSetSize`。MiB=bytes/2²⁰。这些值取决于硬件、软件和系统负载，不是数组大小估计，也没有以当前工作集替代峰值。

profile入口当前只支持Windows的这项OS计量；其他平台可使用前面的 `train` 数值入口。短预算检查可用 `--epochs 1 --batches 1 --batch 64 --checkpoint-epochs 1`，不能用作收敛或性能结论。`train-profile`只缓存选中训练图案和验证图案；常规 `train`缓存全部公开训练/验证图案。新模型均需由用户运行产生，不在发布包内。数据键及量纲见 [data/README.md](data/README.md)，训练附件SHA也列于其中。固定版本引用见 [CITATION.cff](CITATION.cff)。
