"""现场生成的环形像素面示例；不读取发布之外的测试/结果数据。"""
from pathlib import Path
import json
import numpy as np
from .data import load_model
from .network import predict
from .physics import solve
from .correction import refine
from .extraction import extract

def demo(model_path,target,frequency=7.,tolerance=.1,maximum_iterations=64):
    """参考解、近似电流及修正电流均在本次运行中生成。

    三个提取算子读取完全相同的近似电流；提取步骤本身不更新电流。
    这只是入口示例，不承诺新随机训练模型达到论文中的固定数值。
    """
    if frequency<=0 or tolerance<=0 or maximum_iterations<1:
        raise ValueError('frequency/tolerance/maximum_iterations必须为正')
    target=Path(target)
    target.mkdir(parents=True,exist_ok=False)
    model=load_model(model_path)
    N,Q,M=(int(model[k]) for k in ('N','Q','M'))
    mask=np.zeros((8,8),bool)
    mask[2:6,2:6]=True
    mask[3:5,3:5]=False
    reference,X,_,diagnostic,a=solve(mask,frequency,N=N,Q=Q,M=M,method='dense')
    Y=predict(model,mask,[frequency])[:,:,0]
    corrected,correction,_=refine(a,Y,tolerance=tolerance,maximum_iterations=maximum_iterations)
    arrays={'mask':mask,'reference_current':X,'raw_current':Y,'corrected_current':corrected,
            'reference_GSM':reference}
    errors={}
    for state,current in [('raw',Y),('corrected',corrected)]:
        errors[state]={}
        for method in ['projection','reciprocal','stationary']:
            S,_,residual=extract(a,current,method=method)
            assert np.all(np.isfinite(S)),(state,method)
            arrays[state+'_'+method+'_GSM']=S
            errors[state][method]={'relative_GSM_error':float(np.linalg.norm(S-reference)/np.linalg.norm(reference)),
                                  'relative_EFIE_residual':residual['relative_plus']}
    np.savez_compressed(target/'example.npz',**arrays)
    result={'example':'procedurally_generated_pixel_ring','N':N,'Q':Q,'M':M,'frequency_GHz':frequency,
            'reference_residual':diagnostic['plus']['maximum_column_residual'],
            'same_currents_for_all_three_methods':True,'errors':errors,
            'COCG_tolerance':tolerance,'COCG_maximum_iterations':maximum_iterations,
            'COCG_failed_ports':int(np.count_nonzero(correction['broken'])),
            'COCG_unconverged_ports':int(np.count_nonzero(~correction['converged'])),
            'output':str(target)}
    (target/'example.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
