"""电流文件接口与现场生成的单层PEC环形图案示例。"""
from pathlib import Path
import json
import numpy as np
from .physics import sheet, solve
from .extraction import extract

METHODS=('projection','reciprocal','stationary')

def save(target,geometry,Yp,Ym,reference=None,metadata=None):
    """三种提取共用Yp/Ym。输出为复数GSM，而非反射/透射功率。"""
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    arrays={'Yplus':Yp,'Yminus':Yp if Ym is None else Ym,'active_ids':geometry.ids}
    diagnostics={}
    for method in METHODS:
        plus,minus,detail=extract(geometry,Yp,Ym,method=method)
        assert plus.shape==minus.shape==(36,36)
        if not np.all(np.isfinite([plus,minus])):raise ArithmeticError('GSM中出现非有限值')
        arrays[method+'_plus_GSM']=plus;arrays[method+'_minus_GSM']=minus
        diagnostics[method]={'relative_EFIE_residual':detail['relative_plus']}
        if reference is not None:
            diagnostics[method]['relative_GSM_error']=float(np.linalg.norm(plus-reference)/np.linalg.norm(reference))
    if reference is not None:arrays['reference_GSM']=reference
    np.savez_compressed(target/'extracted_GSM.npz',**arrays)
    result={'same_currents_for_all_methods':True,'GSM_shape':[36,36],
            'methods':diagnostics,'output':str(target)}
    if metadata:result.update(metadata)
    (target/'extraction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result

def from_currents(input_path,target):
    """从NPZ读取用户自有近似电流。矩阵行次序必须对应active(mask,N)。"""
    with np.load(input_path,allow_pickle=False) as source:
        mask=np.asarray(source['mask'],bool)
        if mask.shape!=(8,8):raise ValueError('mask必须为8×8')
        N,Q,M=(int(source[name]) for name in ('N','Q','M'))
        if M!=1:raise ValueError('本命令的输出接口固定为M=1的36×36 GSM')
        f=float(source['frequency_GHz']);theta=float(source.get('theta_degrees',0.))
        phi=float(source.get('phi_degrees',30.))
        Yp=np.asarray(source['Yplus'],complex)
        Ym=np.asarray(source['Yminus'],complex) if 'Yminus' in source else None
    geometry=sheet(mask,f,N,Q,M,theta,phi)
    return save(target,geometry,Yp,Ym,metadata={'N':N,'Q':Q,'M':M,'frequency_GHz':f,
                  'incidence_degrees':theta,'azimuth_degrees':phi})

def demo(target,frequency=7.,N=24,Q=192,current_error=.02):
    """用于检查软件接口的单条件例子，不作为论文统计实验。

    先现场求解一个环形PEC图案，再用固定随机种子加入指定相对电流扰动。
    驻值/互易/投影均读取同一个扰动电流，不执行网络训练、样本筛选或调参。
    """
    if frequency<=0 or N<8 or N%8 or Q<1 or not 0<current_error<1:
        raise ValueError('frequency、网格或current-error参数不合法')
    mask=np.zeros((8,8),bool);mask[2:6,2:6]=True;mask[3:5,3:5]=False
    reference,X,_,check,geometry=solve(mask,frequency,N,Q,1,0.,30.,method='dense')
    rng=np.random.default_rng(20261012)
    perturb=rng.standard_normal(X.shape)+1j*rng.standard_normal(X.shape)
    perturb*=current_error*np.linalg.norm(X)/np.linalg.norm(perturb)
    Y=X+perturb
    return save(target,geometry,Y,None,reference,metadata={'example':'procedural_single_layer_ring',
        'N':N,'Q':Q,'M':1,'frequency_GHz':frequency,'incidence_degrees':0.,
        'relative_current_perturbation':current_error,
        'reference_maximum_port_residual':check['plus']['maximum_column_residual']})
