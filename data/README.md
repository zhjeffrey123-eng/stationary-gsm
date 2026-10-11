训练附件解压到本目录后得到 training.h5。

固定600训练图案（split=1）和100验证图案（split=2），原有200测试图案不包含。
原图案顺序及所有保留的原始数组逐元素不变。数据为8×8 PEC像素面，周期20 mm，
N=24、Q=192、M=1；频率5/7/9 GHz；18个入射Floquet端口；法向入射。
HDF5键：masks、active、currents、responses、freq、split。
currents使用η₀J，布局[图案,频率,电流自由度,入射端口]；responses为相应反射端口标签。
训练监督仅使用电流，不以GSM误差选权重。验证集只选择电流MSE最小模型，不参与梯度。

训练数据复用固定v3.0.0附件：

https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/download/v3.0.0/stationary-gsm-training-v3.0.0.zip

SHA-256: `b748f84d90f60c39ebde7e02c7c779ed30978510a4090afc54fa74270b131630`；156762261 bytes。
解压该附件到本代码根目录，得到 `data/training.h5`。v3.1.0不扩展或重新分发训练图案。
