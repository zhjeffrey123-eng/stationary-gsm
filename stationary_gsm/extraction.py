"""对同一组近似电流执行三种 GSM 提取，提取本身不更新电流。"""
import numpy as np
from .physics import apply, assemble_gsm


def extract(a,Yp,Ym=None,method='stationary'):
    """普通转置 .T 用于互易双线性型，不能换成共轭转置。

    H+=(B−)^T Y+，H−=(B+)^T Y−。
    驻值 V+=H+ +(Y−)^T(B+−Z+Y+)。
    精确输出与 V+ 的差为 (X−−Y−)^T Z+ (X+−Y+)。
    这个恒等式针对固定离散模型，不是连续物理误差或筛选保优保证。
    """
    if Ym is None:
        if np.any(a.p['beta'] != 0):
            raise ValueError('斜入射需要两组横向波矢的电流')
        Ym = Yp
    Bp,Bm = a.p['B'][a.ids],a.m['B'][a.ids]
    Yp,Ym = np.asarray(Yp,complex),np.asarray(Ym,complex)
    if Yp.shape!=Bp.shape or Ym.shape!=Bm.shape or not np.all(np.isfinite([Yp,Ym])):
        raise ValueError('预测电流维度或数值不合法')
    Hp,Hm = Bm.T@Yp,Bp.T@Ym
    rp,rm = Bp-apply(a.plus,a.ids,Yp),Bm-apply(a.minus,a.ids,Ym)
    if method in ('field','projection'):
        Vp,Vm = Hp,Hm
    elif method=='reciprocal':
        Vp,Vm = (Hp+Hm.T)/2,(Hm+Hp.T)/2
    elif method=='stationary':
        Vp,Vm = Hp+Ym.T@rp,Hm+Yp.T@rm
    else:
        raise ValueError(method)
    N = a.p['N']
    Rp = -a.p['A']@a.p['J']@Vp/(N*N)
    Rm = -a.m['A']@a.m['J']@Vm/(N*N)
    return assemble_gsm(Rp),assemble_gsm(Rm),{'Rplus':Rp,'Rminus':Rm,'Vplus':Vp,'Vminus':Vm,
        'relative_plus':float(np.linalg.norm(rp)/max(np.linalg.norm(Bp),np.finfo(float).eps)),
        'relative_minus':float(np.linalg.norm(rm)/max(np.linalg.norm(Bm),np.finfo(float).eps))}
