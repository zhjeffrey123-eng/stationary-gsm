"""论文逐例统计、斜入射代数检验和 Python 转换回归测试。"""
from pathlib import Path
import csv
import json
import numpy as np
from .data import ROOT,PAPER,load_bundle,Labels,load_masks
from .physics import solve,sheet,block,apply,assemble_gsm,power_defect
from .extraction import extract
from .network import predict,features,geometry,gradients
from .stack import coupled,cascade,indices
from .fields import reconstruct


def statistics(root=ROOT):
    folder = Path(root)/PAPER/'nn'
    groups = {}
    shared = ['relative_current_error','relative_PEC_residual','criterion_attained',
        'actual_snapshot_iteration','trajectory_prefix_seconds','maximum_snapshot_port_residual',
        'median_snapshot_port_iterations','converged_ports','breakdown_ports','all_ports_converged']
    keys = ['topology','frequency_GHz','stopping_mode','requested_iterations','target_residual']
    count,topologies = 0,set()
    with (folder/'cocg_novel_test_quality.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in keys)
            method = row['method']
            if method in groups.get(key,{}):
                raise AssertionError('重复病例')
            groups.setdefault(key,{})[method] = row
            count += 1
            topologies.add(int(row['topology']))
    assert count==90000 and len(topologies)==1000 and len(groups)==30000
    errors = {(tau,m):[] for tau in (.1,.01) for m in ('field','reciprocal','stationary')}
    failures = {tau:[] for tau in (.1,.01)}
    for key,rows in groups.items():
        assert set(rows)=={'field','reciprocal','stationary'}
        ref = rows['field']
        for other in ('reciprocal','stationary'):
            assert all(ref[k]==rows[other][k] for k in shared),'没有使用同一电流'
        tau = float(ref['target_residual'])
        if int(ref['stopping_mode'])!=1 or tau not in (.1,.01):
            continue
        good = all(int(r['criterion_attained'])==1 and np.isfinite(float(r['relative_GSM_error'])) for r in rows.values())
        if good:
            for m,r in rows.items():
                errors[tau,m].append(float(r['relative_GSM_error']))
        else:
            failures[tau].append((int(ref['topology']),float(ref['frequency_GHz'])))
    summary = []
    for tau in (.1,.01):
        assert failures[tau]==[(954,5.),(954,7.),(954,9.)]
        for m in ('field','reciprocal','stationary'):
            e = errors[tau,m]
            assert len(e)==2997
            summary.append({'tau':tau,'method':m,'completed':2997,'failed':3,
                'median_error':float(np.median(e)),
                # MATLAB prctile 对应样本分位位置 (i−0.5)/n，使用 Hazen 定义。
                'p95_error':float(np.quantile(e,.95,method='hazen')),'maximum_error':float(max(e))})
    return {'rows':count,'topologies':1000,'same_current_comparison':True,'statistics':summary}


def demo():
    mask = np.zeros((8,8),bool)
    mask[2:6,2:6] = True
    ref,Xp,Xm,info,a = solve(mask,7.,16,128,1,40.,30.)
    rng = np.random.default_rng(1010)
    Dp = rng.normal(size=Xp.shape)+1j*rng.normal(size=Xp.shape)
    Dm = rng.normal(size=Xm.shape)+1j*rng.normal(size=Xm.shape)
    Dp *= np.linalg.norm(Xp)/np.linalg.norm(Dp)
    Dm *= np.linalg.norm(Xm)/np.linalg.norm(Dm)
    eps = np.logspace(-4,-1,7)
    methods = ['projection','reciprocal','stationary']
    errors,identity = np.zeros((len(eps),3)),[]
    for k,e in enumerate(eps):
        Yp,Ym = Xp+e*Dp,Xm+e*Dm
        for j,method in enumerate(methods):
            S,_,state = extract(a,Yp,Ym,method)
            errors[k,j] = np.linalg.norm(S-ref)/np.linalg.norm(ref)
        H = a.m['B'][a.ids].T@Xp
        Ep,Em = Xp-Yp,Xm-Ym
        defect = H-state['Vplus']-Em.T@apply(a.plus,a.ids,Ep)
        identity.append(np.linalg.norm(defect)/np.linalg.norm(H))
    slopes = [float(np.polyfit(np.log10(eps),np.log10(errors[:,j]),1)[0]) for j in range(3)]
    assert np.max(abs(np.array(slopes)-[1,1,2]))<.01
    assert max(identity)<1e-10 and info['power_defect']<1e-9
    return {'relative_current_perturbation':eps.tolist(),'relative_GSM_error':errors.tolist(),
            'methods':methods,'slopes':slopes,'quadratic_identity_maximum_defect':float(max(identity)),
            'power_defect':info['power_defect']}


def regression(root=ROOT):
    """测试实际算子、原参考解和冻结网络，不能用实现自比代替参考比对。"""
    root = Path(root)
    results = {'statistics':statistics(root),'oblique_demo':demo()}
    mask = load_masks(root)[700]
    a = sheet(mask,7.,16,128,1,40.,30.)
    Z = block(a.plus,a.ids,a.ids)
    rng = np.random.default_rng(83)
    x = rng.normal(size=(len(a.ids),3))+1j*rng.normal(size=(len(a.ids),3))
    y = apply(a.plus,a.ids,x)
    results['fft_dense_relative_error'] = float(np.linalg.norm(y-Z@x)/np.linalg.norm(y))
    Zm = block(a.minus,a.ids,a.ids)
    results['opposite_incidence_reciprocity_error'] = float(np.linalg.norm(Zm-Z.T)/np.linalg.norm(Z))
    assert results['fft_dense_relative_error']<1e-12 and results['opposite_incidence_reciprocity_error']<1e-12
    dense,_,_,_,_ = solve(mask,7.,16,128,1,40.,30.,method='dense')
    fft,_,_,finfo,_ = solve(mask,7.,16,128,1,40.,30.,method='fft')
    results['fft_reference_gsm_difference'] = float(np.linalg.norm(dense-fft)/np.linalg.norm(dense))
    assert results['fft_reference_gsm_difference']<1e-8
    results['fft_reference_residual'] = finfo['plus']['relative_residual']
    model = dict(np.load(root/'model.npz',allow_pickle=False))
    # 新测试标签提供原始精确响应，原始 CSV 提供旧程序的纯网络误差。
    with Labels(root/PAPER/'nn/novel_test_labels.h5') as d:
        mask,ids,X,R = d.case(0)
        Y = predict(model,mask,d.freq)
        expected = {}
        with (root/PAPER/'nn/local_novel_test_quality.csv').open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if int(row['topology'])==1:
                    expected[float(row['frequency_GHz']),row['method']] = float(row['relative_GSM_error'])
        differences = []
        for fi,f in enumerate(d.freq):
            a = sheet(mask,float(f),d.N,d.Q,d.M)
            truth = assemble_gsm(R[:,:,fi])
            for method in ('field','reciprocal','stationary'):
                S,_,_ = extract(a,Y[:,:,fi],method=method)
                error = np.linalg.norm(S-truth)/np.linalg.norm(truth)
                differences.append(abs(error-expected[float(f),method]))
        results['frozen_network_original_csv_maximum_difference'] = float(max(differences))
        # 单精度 erf/BLAS 末位可能不同，要求 GSM 误差指标差小于 2e−5。
        assert max(differences)<2e-5
    stack_errors = []
    masks = load_masks(root)[700:703]
    for L in (2,3):
        path = root/PAPER/f'stack_validation/stack_L{L}_N16_Q128_f7_th40.npz'
        old = load_bundle(path)
        S,sheets,info = coupled(masks[:L],7.,16,128,3,40.,30.,[.008,.012][:L-1])
        e = np.linalg.norm(S-old['Sref'])/np.linalg.norm(old['Sref'])
        stack_errors.append(float(e))
        assert e<1e-10 and info['power_defect']<1e-9
        Sc,_ = cascade(sheets,info['ports']['kz'],[.008,.012][:L-1])
        orig = np.asarray(old['Scascade'],object)
        # scipy 对 cell 解码返回 object array，每个元素就是完整 GSM。
        saved = old['Scascade'][-1] if isinstance(old['Scascade'],list) else orig.flat[-1]
        ce = np.linalg.norm(Sc-saved)/np.linalg.norm(saved)
        assert ce<1e-10
    results['original_two_three_layer_gsm_differences'] = stack_errors
    # 场图与原 N96 电流和原场数组逐点比对，检验单位、相位和网格顺序。
    folder = root/PAPER/'fine_reference'
    ref = load_bundle(folder/'reference_t701_f7_a0_N96_Q768_M1.npz')
    old_field = load_bundle(folder/'fine_fields_t701_f7_a0_N96_Q768.npz')
    ids = np.asarray(ref['info']['ids'],int)-1
    values = reconstruct(ref['Xp'][:,8],ids,96)
    field_errors = {k:float(np.linalg.norm(values[k]-old_field[k])/max(np.linalg.norm(old_field[k]),np.finfo(float).eps))
                    for k in ('jx','jy','ex','ey','ez','emag')}
    assert max(field_errors.values())<1e-10
    results['original_field_relative_differences'] = field_errors
    # 训练反向传播用有限差分检验少量权重，验证梯度方向和相对损失权重。
    tiny = {k:np.asarray(v,dtype=float) if k.startswith(('W','b')) else v for k,v in model.items()}
    u = rng.normal(size=(71,5)).astype(np.float32)
    target,weight = rng.normal(size=(2,5)).astype(np.float32),np.ones(5,np.float32)
    _,grad = gradients(tiny,u,target,weight)
    h = 1e-3
    grads = []
    for k,ix in [('W3',(0,0)),('b2',(3,)),('W1',(2,3))]:
        value = tiny[k][ix]
        tiny[k][ix] = value+h
        plus,_ = gradients(tiny,u,target,weight)
        tiny[k][ix] = value-h
        minus,_ = gradients(tiny,u,target,weight)
        tiny[k][ix] = value
        finite = (plus-minus)/(2*h)
        grads.append(float(abs(finite-grad[k][ix])/max(1.,abs(finite),abs(grad[k][ix]))))
    assert max(grads)<.005
    results['training_gradient_maximum_scaled_difference'] = max(grads)
    return results
