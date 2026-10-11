"""纯 NumPy 电流索引与相位还原，导入此文件不加载 Torch。

卷积输出布局为[field,36通道,8个y锚点,8个x锚点]。
通道=((component*2+realimag)*3+suby)*3+subx；component是两个切向电流，
realimag分别是实部/虚部，suby/subx是每个8×8宿主像素内的3×3子网格。
原细网格行号为component*24*24+iy*24+ix；field=frequency_index*18+port。
一幅图案的有效行与原电磁求解器严格相同，先缩放再恢复复相位。
"""
from __future__ import annotations
import numpy as np

N = 24
PORTS = 18
CHANNELS = 36


def row_locations(row):
    row = np.asarray(row, dtype=np.int64)
    if np.any((row < 0) | (row >= 2*N*N)):
        raise ValueError("Current row outside the fixed N24, two-component grid")
    component, remainder = row//(N*N), row%(N*N)
    ix, iy = remainder%N, remainder//N
    subx, suby = ix%3, iy%3
    real_channel = component*18 + suby*3 + subx
    return dict(component=component, ix=ix, iy=iy, anchor_x=ix//3, anchor_y=iy//3,
                subx=subx, suby=suby, real_channel=real_channel, imag_channel=real_channel+9)


def host_to_grid(host):
    host = np.asarray(host)
    if host.ndim != 4 or host.shape[1:] != (36,8,8):
        raise ValueError("Expected host [fields,36,8,8]")
    # c,ri,suby,subx,ay,ax -> c,ri,ay,suby,ax,subx；合并锚点和子格为24×24。
    return host.reshape(-1,2,2,3,3,8,8).transpose(0,1,2,5,3,6,4).reshape(-1,2,2,24,24)


def gather_numpy(host, field, row):
    """按原电磁行号采集[实部/虚部,系数]，此步尚未缩放或恢复相位。"""
    host = np.asarray(host)
    field, row = np.broadcast_arrays(np.asarray(field,dtype=np.int64),np.asarray(row,dtype=np.int64))
    if host.ndim != 4 or host.shape[1:] != (36,8,8) or np.any((field<0)|(field>=len(host))):
        raise ValueError("Invalid host or field indices")
    loc = row_locations(row)
    return np.stack([host[field,loc["real_channel"],loc["anchor_y"],loc["anchor_x"]],
                     host[field,loc["imag_channel"],loc["anchor_y"],loc["anchor_x"]]])


def phases(row, port, N=24):
    if N != 24:
        raise ValueError("The frozen CNN requires N24")
    port = np.asarray(port,dtype=np.int64)
    if np.any((port<0)|(port>=18)):
        raise ValueError("Expected 18 M1 ports")
    loc = row_locations(row)
    m, n = (port//2)%3-1, (port//2)//3-1
    # 18端口含9个(m,n)模态和2种偏振；两个电流分量的半格错位与原求解器一致。
    x = (loc["ix"]+.5*(loc["component"]==1))/24
    y = (loc["iy"]+.5*(loc["component"]==0))/24
    return np.exp(1j*2*np.pi*(m*x+n*y))


def queries(frequencies):
    """频率优先、18端口次之；列为归一化频率、m、n、±1偏振。"""
    frequencies = np.asarray(frequencies,dtype=np.float64).reshape(-1)
    port = np.tile(np.arange(18),len(frequencies))
    return np.column_stack([np.repeat((frequencies-7)/2,18), (port//2)%3-1,
                            (port//2)//3-1, 2*(port%2)-1]).astype(np.float32)


def active_host_mask(ids):
    """有效电流行对应实/虚两通道均为1；其他位置硬置0，不参与监督。"""
    result = np.zeros((36,8,8),dtype=np.float32)
    loc = row_locations(ids)
    result[loc["real_channel"],loc["anchor_y"],loc["anchor_x"]] = 1
    result[loc["imag_channel"],loc["anchor_y"],loc["anchor_x"]] = 1
    return result


def predict_from_host(host, nf, ids, scale=1, return_full=False):
    """还原相位并返回[有效电流行,18端口,nf频率]，对应原physics/extract接口。

此函数只处理一幅图案，多图案输出必须先按拓扑拆分。
输出float32先乘训练尺度，再转complex128乘正相位exp(+i2π(mx+ny))。
无效行显式置零；有效行编号和端口/频率布局不作转置猜测。
    """
    if len(host) != 18*nf:
        raise ValueError("One topology requires nf*18 fields")
    ids = np.asarray(ids,dtype=np.int64)
    grid = host_to_grid(host)
    real = (grid[:,:,0].reshape(-1,1152)*np.float32(scale)).astype(np.float32)
    imag = (grid[:,:,1].reshape(-1,1152)*np.float32(scale)).astype(np.float32)
    values = (real+1j*imag).astype(np.complex128)
    values *= phases(np.arange(1152)[None,:],np.tile(np.arange(18),nf)[:,None])
    active = np.zeros(1152,dtype=bool)
    active[ids] = True
    values[:,~active] = 0
    if return_full:
        return values
    return values[:,ids].reshape(nf,18,len(ids)).transpose(2,1,0)
