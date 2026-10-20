# 双周期 PEC 结构的驻值 GSM 提取

本仓库提供谱域电场积分方程、互易端口、驻值 GSM 提取和层间连接的 Python 实现。输入是单层像素几何和已有近似表面电流，输出是包含传播与倏逝 Floquet 模式的复数散射矩阵。

## 安装与示例

建议使用 Python 3.11 或以上版本，在虚拟环境中安装依赖。

```bash
python -m pip install -r requirements.txt
python -m stationary_gsm demo --out local_demo
```

`demo` 在运行时构造一个 PEC 环形像素图案，现场求解该单条件的参考电流，再加入相对幅值为 0.02 的电流扰动。投影、互易平均和驻值提取读取完全相同的扰动电流。该单条件示例演示三种提取接口。运行目录需为尚不存在的目录。

## 提取自己的电流

```bash
python -m stationary_gsm extract --input my_currents.npz --out my_GSM
```

NPZ 文件采用以下字段。

| 字段 | 类型与含义 |
|---|---|
| `mask` | 8×8 布尔矩阵，`True` 为 PEC，`False` 为空气 |
| `N` | 每个周期方向的电流网格细分数，需为 8 的整数倍 |
| `Q` | 周期 Green 谱截断，保留 −Q 至 Q 的二维阶次 |
| `M` | 模态端口截断，此命令固定为 1 |
| `frequency_GHz` | 频率，单位 GHz |
| `theta_degrees` | 从法向测量的入射角，默认 0 |
| `phi_degrees` | 从 x 轴测量的方位角，默认 30 |
| `Yplus` | 复数近似电流系数矩阵，形状为 active_rows×18 |
| `Yminus` | 相反横向波矢的近似电流，形状同上，斜入射时必须提供 |

电流系数采用 η₀J 的屋顶基函数展开，η₀ 为自由空间波阻抗。矩阵行次序必须与 `physics.active(mask, N)` 返回的零基索引一致。18 列依次对应 m,n=−1,0,1 各阶的切向 Ex、Ey 入射端口。法向入射时两组横向波矢相同，可省略 `Yminus`。程序不更换用户提供的电流。

`extracted_GSM.npz` 保存三种提取的正、反横向波矢 GSM。每个矩阵均为 36×36，两侧各含 18 个端口。`extraction.json` 保存真实 EFIE 残差和接口设置。GSM 中的量是复数幅值，计算反射和透射功率时须采用端口导纳权重。

## 模型和主要文件

实现采用自由空间中零厚度、周期为 20 mm 的 PEC 像素面，以及 e^(−iωt) 时间约定。基函数仅定义在金属区域。模态截止点不包含在本接口的适用条件内。

| 文件 | 内容 |
|---|---|
| `physics.py` | 周期 Green 谱、屋顶基函数、Floquet 端口与 EFIE |
| `extraction.py` | 同电流投影、互易平均和驻值输出 |
| `correction.py` | 对已有电流的 COCG 残差修正 |
| `stack.py` | 多层 GSM 的端口连接 |
| `example.py` | 用户电流输入与单条件使用示例 |

驻值提取采用普通转置构成互易双线性型，而不是共轭转置。其二阶输出误差恒等式针对固定离散模型，不控制网格截断或连续物理误差。层间连接须保留足够的倏逝端口。

## 论文案例的参考复算

| 目录 | 对应内容 |
|---|---|
| `data/resolution` | 图 2(c,d) 中结构 701、40° 入射的完整 N = 96 GSM 点；包含 N = 80、96 的 36×36 复矩阵和几何 |
| `data/fss` | 图 4 的五层 FSS 几何、耦合 EFIE 参考曲线、峰值和网格检查 |
| `data/polarizer` | 图 5 的 10–11 GHz 五层极化器几何、独立 lab x/y 入射的物理响应与 Jones 复矩阵、网格/谱/接口检查 |

从仓库根目录执行以下命令。输出目录需为尚不存在的目录。

```bash
python -m examples.stack_reference --geometry data/fss/geometry.json --frequencies 7 --polarization x --out local_fss
python -m examples.polarizer.reproduce_polarizer --mode reference --frequencies 10,10.5,11 --N 48 --output local_polarizer
python -m examples.polarizer.reproduce_polarizer --mode matched --N 48 --output local_readout
python -m examples.recompute_resolution --geometry data/resolution/geometry.json --pattern 701 --N 80 96 --Q-ratios 6 8 --theta 40 --out local_resolution
```

`stack_reference` 分别执行直接耦合 EFIE 与参考层 GSM 级联，并保存 4×4 基模复矩阵、功率、轴比及真实残差。其默认接口截断为 M = 1，可用 `--M` 调整。两种算法共享离散基函数和 Green 模型；网格与接口差异在数据中分别记录。`recompute_resolution` 对每个设置实际求解全部 18 个入射列，并用较细设置的完整 GSM 范数归一化相邻设置之差。N = 80、96 的计算耗时取决于硬件和收敛情况。

FSS 的 `peak_and_mesh.csv` 明确区分耦合 EFIE 与 M = 3 参考层级联。输入自己的近似电流时，使用上面的 `extract` 接口与 `stack.py` 进行同电流比较。学习模型和训练、测试及 CPU 基准原始材料保持私有。

极化器的真实周期为 13.3333 mm，方形晶格及公共 u/v 主轴相对 lab x/y 同时旋转 45°。几何文件保存五层实际掩模、空气间隙和使各矩形中心重合的精确平移。`reproduce_polarizer` 用 20 mm 核心的精确 PEC 尺度变换计算每个物理频率，并在层间 Green 谱和端口中保留这些平移；lab x/y 两个输入均为显式求解的独立列。N = 48、Q = 384 的 41 个频率点中，总透射功率 T 为 0.973866–0.987805，AR 为 0.132099–0.200999 dB；N = 64 检查的 AR 不超过 0.274714 dB。实际双输入复矩阵保留在数据中，圆度定义为 2Im(Ex·conj(Ey))/T。

`matched` 模式以全零电流初始化非学习 GMRES，逐端口保留第一次真实 EFIE 残差不超过 0.1 的电流。相同形状复用相同电流，三种提取共用同一快照，并与相同 N = 48、Q = 384、M = 1 的参考电流级联比较。`polarizer_readouts.npz` 同时保存完整 4×4 lab 基模 GSM 与 36×36 u/v 外部 GSM。无需模型权重或训练资料即可重生成电流和响应；省略 `--frequencies` 时重算全部 41 点。

## 引用与校验

引用信息见 `CITATION.cff`。当前发布为 [v3.7.0](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/tag/v3.7.0)。包内源码、文档和数据的 SHA-256 值见 `CODE_SHA256SUMS.txt`；发布 ZIP 另附校验文件。
