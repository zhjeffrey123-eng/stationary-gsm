# Stationary GSM extraction — v3.0.0

本仓库仅提供 Python 数值代码和固定训练/验证标签。代码为周期20 mm的8×8零厚度PEC像素面实现谱域EFIE、电流网络、COCG修正、同电流三种GSM提取及多层GSM连接。中文注释说明入射相位、RMS缩放和互易普通转置。

下载代码附件和训练附件，解压到同一目录。训练附件产生 `data/training.h5`，含600个训练图案（split=1）和100个验证图案（split=2），不含200个保留测试图案。N24/Q192/M1、5/7/9 GHz、法向入射、18端口。所有保留标签数组与原数据逐元素一致；验证集不参与梯度。

Python 3.12，依赖采用本次实际检查的版本：

```bash
python -m pip install -r requirements.txt
python -m stationary_gsm train --hidden 128 --out results/train128
python -m stationary_gsm train --hidden 64 --out results/train64
python -m stationary_gsm demo --model results/train128/model.npz --out results/example
```

训练默认80 epochs、每轮128 batches、每batch 4096个有效电流样本、seed120026。监督只用相对电流MSE；训练集全局RMS缩放标签，移除解析入射相位，再以验证电流MSE选择 `model.npz`。默认另保存第10轮 `checkpoint_10.npz`，可用 `--epochs` / `--checkpoint-epochs` 更改预算。64和128单元共享同一训练实现，参数只改变两隐藏层宽度。输出目录必须尚不存在；预训练权重不分发，新训练模型不承诺逐位重现原权重。

`demo`用刚训练的模型和代码现场生成的环形像素面计算参考EFIE解、预测电流、COCG修正，再从同一组电流提取projection/reciprocal/stationary GSM。它将本次生成的数组及误差写入新目录，记录失败或未收敛端口；不读取论文测试集或已归档结果。网络编码仅适用M=1和法向入射，数值提取代码也支持成对斜入射算子。多层接口见 `stationary_gsm/stack.py` 的 `cascade` 与 `coupled`。

较短的入口检查可使用 `--epochs 1 --batches 1 --batch 64`；这仍读取全部公开训练/验证图案，不能用作收敛或性能结论。完整训练会缓存各图案的有效电流标签，需要较大的内存和运行时间。数据键及量纲见 [data/README.md](data/README.md)。固定版本引用见 [CITATION.cff](CITATION.cff)。
