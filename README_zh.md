# 金属像素超表面 GSM 提取：Python 代码与数据

本包对应 AWPL 稿件 *A Stationary GSM Extraction Method for 2D Periodic Metal Pixel Metasurfaces*，只涉及自由空间中的双周期 PEC 金属像素模型。周期为 20 mm，图案为 8×8 二值矩阵，1 表示金属，0 表示空气。普通投影、互易平均和驻值提取使用完全相同的近似表面电流。

## 安装与快速检查

使用 Python 3.11 或更新版本。计算只依赖免费 Python 库，不需要 MATLAB 授权。

```bash
python -m venv .venv
```

Windows 激活环境：

```powershell
.venv\Scripts\activate
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

随后安装依赖并运行：

```bash
python -m pip install -r requirements.txt
python -m stationary_gsm verify
python -m stationary_gsm demo
```

`verify` 从原始 90000 行 CSV 重新统计表 I，并检查三种方法是否复用同一电流。1000 个拓扑分别在三个频率评价，不能把 3000 个频率病例当作独立几何。

`demo` 实际求解一个 7 GHz、40° 斜入射 PEC 方片，检查投影和互易平均的一阶误差、驻值提取的二阶误差、互易恒等式及功率守恒。这两个命令无需大型数据附件。

## 数据下载与格式

固定版本下载入口：[v2.0.0 Release](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/tag/v2.0.0)。将 `stationary-gsm-data-v2.0.0.zip` 解压到仓库根目录，合并 `output` 目录，不要多增加一层目录。

公开包没有 `.m` 和 `.mat` 文件。大型电流标签使用 `.h5`，权重和参考结果使用 `.npz`。原始 CSV 完整保留。两种格式都可由 Python 直接读取。

```python
import numpy as np
from stationary_gsm.data import Labels, load_bundle

model = dict(np.load('model.npz', allow_pickle=False))
print(model['W1'].shape)  # (128, 71)

with Labels('output/pec8x8_trial/dataset.h5') as labels:
    mask, ids, currents, reflection = labels.case(700)  # 原数据集拓扑 701
    print(mask.shape, currents.shape)

reference = load_bundle('output/awpl/revision20_metal_paper/stack_validation/stack_L2_N16_Q128_f7_th40.npz')
print(reference['Sref'].shape)
```

HDF5 数组布局如下：

| 字段 | 布局 | 含义 |
|---|---|---|
| masks | [拓扑,y,x] | 二值金属图案 |
| active | [拓扑,全周期自由度] | 有效金属支撑 |
| currents | [拓扑,频率,全周期自由度,端口] | 复数 η₀J 标签 |
| responses | [拓扑,频率,出射端口,入射端口] | 复数反射矩阵 |
| freq | [频率] | GHz |
| split | [拓扑] | 1 训练、2 验证、3 原测试，仅原 900 拓扑集有此字段 |

Python 数组从零开始编号。`Labels.case(700)` 对应原拓扑编号 701。参考缓存中保留的旧 `info.ids` 是从一开始的编号，使用时必须减 1。Python 新参考解显式写入 `index_base=0`。

323 个原数据文件都经过逐值核对，数组形状、类型、复数实虚部、标量和嵌套字段一致。HDF5 标签为了便于 Python 逐拓扑读取，改变了轴顺序，但数值不变。转换映射与文件摘要见 `docs/data_conversion.json`，当前包摘要见 `MANIFEST_SHA256.csv`。

## 完整验证与复算

```bash
python -m stationary_gsm audit
python -m stationary_gsm evaluate --count 1000 --out results/evaluation
python -m stationary_gsm stack --full --out results/stack
python -m stationary_gsm fine --N 48 64 80 96 --frequencies 5 7 9 --angles 0 20 40 --ratios 6 8 --out results/fine
python -m stationary_gsm fields --out results/fields
python -m stationary_gsm figures --out results/figures
```

`audit` 包含原统计、斜入射示例、FFT 与有效阻抗块一致性、正反横向入射的互易关系、冻结网络与旧 CSV 对比、二层三层原参考 GSM 对比以及 N96 原场图的逐点对比。

`evaluate` 用冻结网络计算 1000 个新测试结构，并保存纯网络电流、指定迭代次数和真实残差阈值快照的同电流三法比较。失败病例保留，不能将未达标快照当成有效输出。

`stack` 默认运行少量二层三层算例，`--full` 才运行完整 108 个参考条件。层间距为 8 mm 和 12 mm，接口保留 M=0/1/2/3。全耦合 EFIE 和 GSM 连接采用不同组装机制，但共享屋顶基和 Green 物理模型。

`fine` 使用 FFT 矩阵自由预条件参考求解，最后单独检查真实 EFIE 残差。N80/96 耗时较长，建议先用 N48 检查环境。命令中选择的完整网格组合比原发布记录的选定 116 条件更广，Python 复算结果另存。

`fields` 从保存的真实 N80/96 电流重建表面 J 和透射侧 z=2 mm 的近场。`figures` 按四幅主图的原始 JSON 几何导出 PDF、SVG、PNG，字体为 Times New Roman 8 pt，色块和曲线不插值。系统需要已安装该字体。

所有复算命令要求输出目录尚不存在，防止覆盖原论文数据。可用 `--threads 1` 限制数值库线程，避免细网格任务并行占用过多资源。

## 训练和标签重算

```bash
python -m stationary_gsm labels --count 900 --out results/labels
python -m stationary_gsm novel-labels --count 1000 --out results/novel-labels
python -m stationary_gsm train --out results/training
python -m stationary_gsm evaluate --model results/training/model.npz --count 1000 --out results/retrained-evaluation
```

训练网络为 71–128–128–2，GELU 激活，80 epochs，AdamW，按验证集选择权重。源码包含解析反向传播和逐场相对电流 MSE，不需要专有深度学习工具箱。输入、输出归一化和物理修正见 `docs/METHODS_zh.md`。

快速环境检查可用 `train --epochs 1 --batches 1 --batch 64`。这只是程序检查，不是论文的训练结果。Python 和原计算环境的随机数、BLAS 和单精度特殊函数不同，因此重新训练不承诺得到相同权重。复核论文的冻结网络结果应使用已发布的 `model.npz`。

原论文 CSV 中的计时对应原始计算环境。Python 复算的计时单独记录，不得把原时间标成 Python 实测。条件精度统计使用共同达标的 2997 个病例，完成率以全部 3000 个病例为分母，拓扑 954 的三个频率失败记录始终保留。
