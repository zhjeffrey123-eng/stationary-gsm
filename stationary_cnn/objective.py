"""全场监督与逐case相对平方Frobenius损失；不依赖任何私有采样源码。

一个case是一幅图案在一个频率下的全部active电流行×18个入射端口。
每case独立用真解整体平方范数作分母，再对图案及频率均匀平均。
不能把所有图案的系数先一起求mean，否则active行多的图案会得到额外权重。
"""
from __future__ import annotations
import numpy as np
from . import indexing


def phase_table():return indexing.phases(np.arange(1152)[:,None],np.arange(18)[None,:])


def target_host(currents,ids,scale,phase=None):
    """把[nactive,3,18]原复电流转换到CNN54×36×8×8的归一化实虚标签。"""
    currents=np.asarray(currents);ids=np.asarray(ids,dtype=np.int64)
    if currents.shape!=(len(ids),3,18) or len(ids)==0:raise ValueError("positive active rows x3freq x18ports required")
    if not np.isfinite(scale) or scale<=0:raise ValueError("Positive train-only RMS required")
    phase=phase_table() if phase is None else phase
    truth128=currents.astype(np.complex128)
    # 先用原真解作float64分母，不能用float32标签反算分母而掩盖量化误差。
    denominator=np.sum(np.abs(truth128)**2,axis=(0,2))/float(scale)**2
    if not np.all(np.isfinite(denominator)) or not np.all(denominator>0):raise ArithmeticError("Zero/nonfinite reference matrix norm")
    label=truth128/phase[ids,None,:]
    values=label.transpose(1,2,0).reshape(54,len(ids));loc=indexing.row_locations(ids);field=np.arange(54)[:,None]
    target=np.zeros((54,36,8,8),np.float32)
    target[field,loc["real_channel"][None,:],loc["anchor_y"][None,:],loc["anchor_x"][None,:]]=values.real.astype(np.float32)/np.float32(scale)
    target[field,loc["imag_channel"][None,:],loc["anchor_y"][None,:],loc["anchor_x"][None,:]]=values.imag.astype(np.float32)/np.float32(scale)
    return target,denominator


def loss_numpy(prediction,target,denominator,float32=False):
    dtype=np.float32 if float32 else np.float64
    prediction=np.asarray(prediction,dtype=dtype);target=np.asarray(target,dtype=dtype);denominator=np.asarray(denominator,dtype=dtype)
    if denominator.ndim!=2 or denominator.shape[1]!=3 or len(prediction)!=54*len(denominator):raise ValueError("Denominator must be [topologies,3]")
    numerator=np.sum((prediction-target)**2,axis=(1,2,3)).reshape(-1,3,18).sum(axis=2)
    return float(np.mean(numerator/denominator))


def loss_torch(torch,prediction,target,denominator):
    # 实虚部通道直接相加；不除2，不除nactive或18，因为分母已经是整体范数。
    numerator=(prediction-target).square().sum(dim=(1,2,3)).reshape(-1,3,18).sum(dim=2)
    return (numerator/denominator).mean()


def target_from_demod(demod,ids,scale,raw_norm_squared):
    """读取file-backed复float32去相位缓存，在当前小批次内归一化及回排。

    全缓存不复制成anonymous ndarray。去相位缓存保留完整active×3×18数据；
    scale仍在取batch时相除，确保与先导的float32先cast后scale运算顺序一致。
    """
    demod=np.asanyarray(demod);ids=np.asarray(ids,dtype=np.int64)
    if demod.shape!=(len(ids),3,18):raise ValueError("Demod cache shape mismatch")
    values=demod.transpose(1,2,0).reshape(54,len(ids));loc=indexing.row_locations(ids);field=np.arange(54)[:,None]
    target=np.zeros((54,36,8,8),np.float32)
    target[field,loc["real_channel"][None,:],loc["anchor_y"][None,:],loc["anchor_x"][None,:]]=values.real/np.float32(scale)
    target[field,loc["imag_channel"][None,:],loc["anchor_y"][None,:],loc["anchor_x"][None,:]]=values.imag/np.float32(scale)
    denominator=np.asarray(raw_norm_squared,dtype=np.float64)/float(scale)**2
    if not np.all(np.isfinite(denominator)) or not np.all(denominator>0):raise ArithmeticError("Invalid full-case denominator")
    return target,denominator
