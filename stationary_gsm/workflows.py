"""复算工作流。新结果必须写到新目录，原论文数据始终只读。"""
from pathlib import Path
from dataclasses import replace
import csv
import json
import time
import h5py
import numpy as np
from .data import ROOT,PAPER,Labels,load_bundle,save_bundle,load_masks
from .physics import sheet,solve,active,assemble_gsm,phase_solve,power_defect
from .extraction import extract
from .network import predict
from .correction import refine
from .stack import coupled,cascade,indices,reflectance,gap_amplitudes
from .fields import reconstruct


def new_folder(path):
    path = Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def write_rows(path,rows):
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_labels(target,root=ROOT,novel=False,count=None):
    """按发布的固定拓扑复算 EFIE 标签。避免随机重生成改变训练/测试结构。"""
    out = new_folder(target)
    if novel:
        with Labels(Path(root)/PAPER/'nn/novel_test_labels.h5') as d:
            masks,split = d.masks[:],None
    else:
        masks = load_masks(root)
        table = np.loadtxt(Path(root)/'output/pec8x8_trial/masks_and_split.csv',delimiter=',')
        split = table[:,1].astype(int)
    masks = masks[:count] if count is not None else masks
    N,Q,M,freq = 24,192,1,np.array([5.,7.,9.])
    templates = [sheet(np.zeros((8,8)),float(f),N,Q,M) for f in freq]
    with h5py.File(out/'labels.h5','w') as h:
        for k,v in {'N':N,'Q':Q,'M':M}.items():
            h.attrs[k] = v
        h.create_dataset('masks',data=masks)
        h.create_dataset('freq',data=freq)
        if split is not None:
            h.create_dataset('split',data=split[:len(masks)])
        h.create_dataset('active',shape=(len(masks),2*N*N),dtype=bool)
        h.create_dataset('currents',shape=(len(masks),3,2*N*N,18),dtype=np.complex64,
            chunks=(1,1,2*N*N,18),compression='gzip',compression_opts=1,shuffle=True)
        h.create_dataset('responses',shape=(len(masks),3,18,18),dtype=np.complex128)
        for t,mask in enumerate(masks):
            ids = active(mask,N)
            h['active'][t,ids] = True
            for fi,a in enumerate(templates):
                X,info = phase_solve(a.plus,ids,a.p['B'][ids])
                full = np.zeros((2*N*N,18),np.complex64)
                full[ids] = X
                R = a.p['C'][:,ids]@X
                if power_defect(assemble_gsm(R),a.p)>1e-8:
                    raise ArithmeticError('标签功率守恒检查失败')
                h['currents'][t,fi],h['responses'][t,fi] = full,R
            print(f'labels {t+1}/{len(masks)}',flush=True)
    return {'labels':str(out/'labels.h5'),'topologies':len(masks)}


def evaluate(target,root=ROOT,count=1000,correct=True,model_path=None):
    """冻结网络逐例评价。保留纯网络、指定迭代次数和真实残差阈值快照。

    每组三种提取共享电流。失败阈值行仍写出 NaN，并标记 criterion_attained=0。
    计时是当前 Python 环境实测，不覆盖原论文中原计算环境的计时。
    """
    out = new_folder(target)
    model = dict(np.load(model_path or Path(root)/'model.npz',allow_pickle=False))
    rows = []
    with Labels(Path(root)/PAPER/'nn/novel_test_labels.h5') as d:
        templates = [sheet(np.zeros((8,8)),float(f),d.N,d.Q,d.M) for f in d.freq]
        for t in range(min(count,len(d))):
            mask,ids,X,R = d.case(t)
            start = time.perf_counter()
            Y = predict(model,mask,d.freq)
            inference = time.perf_counter()-start
            for fi,template in enumerate(templates):
                a = replace(template,ids=ids)
                truth = assemble_gsm(R[:,:,fi])
                stages = [('network',0.,Y[:,:,fi],True,0,0.)]
                correction_info = None
                if correct:
                    _,correction_info,snap = refine(a,Y[:,:,fi])
                    for j,tau in enumerate((.1,.01,.001,.0001)):
                        attained = bool(np.all(correction_info['level_hit'][:,j]))
                        it = int(np.nanmax(correction_info['level_iterations'][:,j])) if attained else -1
                        stages.append(('residual',tau,snap[j],attained,it,float(correction_info['level_seconds'][j])))
                    for it,value in correction_info['fixed'].items():
                        if it==0:
                            continue  # 纯网络阶段已经对应第 0 次迭代，避免重复一组三法。
                        actual = min(it,correction_info['last_iteration'])
                        stages.append(('fixed',float(it),value,True,actual,float(correction_info['fixed_seconds'][it])))
                for kind,threshold,current,attained,iteration,prefix in stages:
                    for method in ('field','reciprocal','stationary'):
                        error,res,cur = np.nan,np.nan,np.nan
                        duration = np.nan
                        if attained and np.all(np.isfinite(current)):
                            timer = time.perf_counter()
                            S,_,state = extract(a,current,method=method)
                            duration = time.perf_counter()-timer
                            error = float(np.linalg.norm(S-truth)/np.linalg.norm(truth))
                            res = state['relative_plus']
                            cur = float(np.linalg.norm(current-X[:,:,fi])/np.linalg.norm(X[:,:,fi]))
                        rows.append({'topology':t+1,'frequency_GHz':float(d.freq[fi]),'stage':kind,
                            'threshold_or_iterations':threshold,'method':method,'criterion_attained':int(attained),
                            'relative_GSM_error':error,'relative_current_error':cur,'relative_PEC_residual':res,
                            'actual_snapshot_iteration':iteration,'inference_seconds_all_frequencies':inference,
                            'correction_prefix_seconds':prefix,'extraction_seconds':duration,
                            'breakdown_ports':int(np.sum(correction_info['broken'])) if correction_info else 0})
            write_rows(out/'quality.csv',rows)
            print(f'evaluation {t+1}/{min(count,len(d))}',flush=True)
    return {'rows':len(rows),'file':str(out/'quality.csv')}


def stack_validation(target,root=ROOT,full=False):
    out = new_folder(target)
    masks = load_masks(root)[700:703]
    rows = []
    Ns,freq,angles,ratios = ([16,24,32],[5.,7.,9.],[0.,20.,40.],[6,8]) if full else ([16],[7.],[0.,40.],[8])
    for N in Ns:
        for ratio in ratios:
            Q = ratio*N
            for f in freq:
                for theta in angles:
                    for L in (2,3):
                        gaps = [.008,.012][:L-1]
                        Sref,layerS,info = coupled(masks[:L],f,N,Q,3,theta,30.,gaps)
                        pt,P = info['ports'],len(info['ports']['kz'])
                        Scs = []
                        for M in range(4):
                            ix = indices(pt['orders'],M)
                            sel = np.r_[ix,P+ix]
                            layers = [s[np.ix_(sel,sel)] for s in layerS]
                            Sc,rc = cascade(layers,pt['kz'][ix],gaps)
                            Scs.append(Sc)
                            zero = np.flatnonzero(ix==pt['normal'][0])[0]+np.array([0,1])
                            fsel = np.r_[zero,len(ix)+zero]
                            ref = Sref[np.ix_(sel,sel)]
                            rows.append({'layers':L,'N':N,'Q':Q,'frequency_GHz':f,'theta_deg':theta,'M':M,
                                'relative_GSM_error':float(np.linalg.norm(Sc-ref)/np.linalg.norm(ref)),
                                'maximum_fundamental_amplitude_error':float(np.max(abs((Sc-ref)[np.ix_(fsel,fsel)]))),
                                'reference_power_defect':info['power_defect'],'feedback_rcond':rc})
                        save_bundle(out/f'stack_L{L}_N{N}_Q{Q}_f{f:g}_th{theta:g}.npz',
                            {'Sref':Sref,'layerS':layerS,'Scascade':Scs,'masks':masks[:L],'gaps':gaps,'info':info})
                        write_rows(out/'stack_validation.csv',rows)
                        print(f'stack L{L} N{N} Q{Q} {f:g} GHz {theta:g} deg',flush=True)
    return {'reference_conditions':len(rows)//4,'cascade_rows':len(rows)}


def fine_validation(target,root=ROOT,Ns=(48,),frequencies=(7.,),angles=(0.,40.),ratios=(8,)):
    """细网格矩阵自由参考计算。用户可显式选择 N48/64/80/96 和 6N/8N。"""
    out = new_folder(target)
    masks = load_masks(root)
    rows = []
    for topology in (701,702,703):
        mask = masks[topology-1]
        for N in Ns:
            for ratio in ratios:
                for f in frequencies:
                    for angle in angles:
                        start = time.perf_counter()
                        S,Xp,Xm,info,a = solve(mask,f,N,ratio*N,1,angle,30.,method='fft')
                        info['ids'] = a.ids
                        info['index_base'] = 0
                        save_bundle(out/f'reference_t{topology}_f{f:g}_a{angle:g}_N{N}_Q{ratio*N}_M1.npz',
                            {'Sref':S,'Xp':Xp,'Xm':Xm,'info':info,'mask':mask,'topology':topology})
                        rows.append({'topology':topology,'N':N,'Q':ratio*N,'frequency_GHz':f,'theta_deg':angle,
                            'residual':info['plus']['relative_residual'],'power_defect':info['power_defect'],
                            'seconds':time.perf_counter()-start,'output_coverage':'full GSM'})
                        write_rows(out/'fine_reference.csv',rows)
                        print(f'fine t{topology} N{N} {f:g} GHz {angle:g} deg',flush=True)
    return {'conditions':len(rows)}


def field_maps(target,root=ROOT):
    out = new_folder(target)
    folder = Path(root)/PAPER/'fine_reference'
    for N in (80,96):
        v = load_bundle(folder/f'reference_t701_f7_a0_N{N}_Q{8*N}_M1.npz')
        # 发布参考缓存保留原编号，新生成参考解则显式标记 index_base=0。
        base = int(v['info'].get('index_base',1))
        ids = np.asarray(v['info']['ids'],int)-base
        values = reconstruct(v['Xp'][:,8],ids,N)
        np.savez_compressed(out/f'fields_N{N}.npz',**values)
    return {'grids':[80,96],'directory':str(out)}
