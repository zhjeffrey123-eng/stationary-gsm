"""现场重建扩展训练标签；物理算子与已发布版本保持一致。"""
from pathlib import Path
import csv,ctypes,hashlib,json,os,time,traceback
from ctypes import wintypes
import h5py
import numpy as np
from scipy.linalg import solve
from . import physics
N,Q,M,FREQ=24,192,1,(5.,7.,9.)
SEED=2026101101
ADDED=4400

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

class PMC(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
              ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
              ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
              ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
              ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]

def memory():
    c=PMC();c.cb=ctypes.sizeof(c)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    api=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    api.GetProcessMemoryInfo.restype=wintypes.BOOL
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_MiB=c.WorkingSetSize/2**20,
                peak_working_set_MiB=c.PeakWorkingSetSize/2**20,
                private_commit_MiB=c.PagefileUsage/2**20,
                peak_private_commit_MiB=c.PeakPagefileUsage/2**20)

def initialise_h5(masks,protocol):
    if OUTPUT.exists():raise FileExistsError('不覆盖候选H5')
    with h5py.File(SOURCE,'r') as s,h5py.File(OUTPUT,'w') as h:
        for field in ('N','Q','M'):h.attrs[field]=s.attrs[field]
        for field,value in dict(period_m=.020,theta_degrees=0.,phi_degrees=30.,
            frequency_unit='GHz',current_scaling='eta0 J',time_convention='exp(-i omega t)').items():
            h.attrs[field]=value
        h.attrs['scope']='5000 training patterns and 100 unchanged validation patterns; no test labels'
        h.attrs['training_patterns']=5000;h.attrs['validation_patterns']=100
        h.attrs['generation_seed']=SEED;h.attrs['layout']='train[0:5000],validation[5000:5100]'
        h.create_dataset('freq',data=s['freq'][:],dtype=s['freq'].dtype)
        for name in ('masks','active','currents','responses','split'):
            shape=(5100,*s[name].shape[1:]);kwargs={}
            if name=='currents':kwargs=dict(chunks=(1,1,1152,18),compression='gzip',compression_opts=1,shuffle=True)
            if name=='responses':kwargs=dict(chunks=(1,1,18,18),compression='gzip',compression_opts=1,shuffle=True)
            h.create_dataset(name,shape=shape,dtype=s[name].dtype,**kwargs)
            for old0,new0,count in ((0,0,600),(5000,5000,100)):
                for j in range(count):h[name][new0+j]=s[name][old0+j]
        h['masks'][600:5000]=masks;h['split'][600:5000]=1
        for j,g in enumerate(masks):h['active'][600+j,physics.active(g,N)]=True
    verify_source_identity()

def verify_source_identity():
    report={}
    with h5py.File(SOURCE,'r') as s,h5py.File(OUTPUT,'r') as h:
        assert np.array_equal(s['freq'][:],h['freq'][:]) and s['freq'].dtype==h['freq'].dtype
        for label,old0,new0,count in (('original_training_600',0,0,600),('original_validation_100',5000,5000,100)):
            report[label]={}
            for name in ('masks','active','currents','responses','split'):
                a=hashlib.sha256();b=hashlib.sha256();assert s[name].dtype==h[name].dtype
                for j in range(count):
                    old=s[name][old0+j];new=h[name][new0+j]
                    assert np.array_equal(old,new),(label,name,j)
                    a.update(np.asarray(old).tobytes(order='C'));b.update(np.asarray(new).tobytes(order='C'))
                assert a.hexdigest()==b.hexdigest()
                report[label][name]=dict(dtype=str(s[name].dtype),shape=[count,*s[name].shape[1:]],
                    source_array_sha256=a.hexdigest(),candidate_array_sha256=b.hexdigest(),exact_equal=True)
        report['frequency_exact_equal']=True
    dump(HERE/'source_array_identity.json',report)
    return report

def run():
    protocol=json.loads((HERE/'generation_protocol.json').read_text(encoding='utf-8'))
    assert protocol['generator_sha256']==sha(__file__) and protocol['published_physics_sha256']==sha(Path(physics.__file__))
    assert protocol['source_training_h5_sha256']==sha(SOURCE)
    assert protocol['new_masks_file_sha256']==sha(HERE/'new_training_masks.npy')
    masks=np.load(HERE/'new_training_masks.npy',allow_pickle=False)
    baseline=memory() if os.name=='nt' else {};start=time.perf_counter();initialise_h5(masks,protocol)
    caches=[];operator_timer=time.perf_counter()
    for f in FREQ:
        a=physics.sheet(np.zeros((8,8),bool),f,N,Q,M,0.,30.)
        all_ids=np.arange(2*N*N);Z=physics.block(a.plus,all_ids,all_ids)
        path=HERE/f'full_Z_f{f:g}_N24_Q192_M1.npy';np.save(path,Z,allow_pickle=False)
        del Z
        caches.append((a,np.load(path,mmap_mode='r'),path))
        print('OPERATOR_CACHED',f,'memory',memory() if os.name=='nt' else {},flush=True)
    operator_seconds=time.perf_counter()-operator_timer
    dump(HERE/'operator_cache_manifest.json',dict(published_physics_sha256=protocol['published_physics_sha256'],
        operator_wall_seconds=operator_seconds,files=[dict(path=str(path),sha256=sha(path),shape=list(z.shape),dtype=str(z.dtype)) for a,z,path in caches]))
    log_fields=['new_pattern','target_zero_based_index','frequency_GHz','active_rows','incident_columns',
        'solve_seconds','case_wall_seconds','relative_Frobenius_residual','maximum_column_residual','power_defect',
        'peak_working_set_MiB','private_commit_MiB','status']
    maxima=dict(maximum_column_residual=0.,relative_Frobenius_residual=0.,power_defect=0.)
    solves=[];count=0;prefix_start=time.perf_counter()
    with h5py.File(OUTPUT,'r+') as h,(HERE/'label_cases.csv').open('w',newline='',encoding='utf-8') as lf:
        writer=csv.DictWriter(lf,fieldnames=log_fields);writer.writeheader();lf.flush()
        for first,last in ((0,10),(10,ADDED)):
            for fi,(a,Z,path) in enumerate(caches):
                for j in range(first,last):
                    current_case=dict(new_pattern=j+1,target_zero_based_index=600+j,frequency_GHz=FREQ[fi])
                    case_start=time.perf_counter()
                    try:
                        ids=physics.active(masks[j],N);B=a.p['B'][ids];Zi=Z[np.ix_(ids,ids)]
                        solve_start=time.perf_counter();X=solve(Zi,B,assume_a='gen');solve_seconds=time.perf_counter()-solve_start
                        residual=physics.apply(a.plus,ids,X)-B
                        columns=np.linalg.norm(residual,axis=0)/np.maximum(np.linalg.norm(B,axis=0),np.finfo(float).tiny)
                        maximum=float(np.max(columns));fro=float(np.linalg.norm(residual)/np.linalg.norm(B))
                        R=a.p['C'][:,ids]@X;power=physics.power_defect(physics.assemble_gsm(R),a.p)
                        if not np.all(np.isfinite(X)) or not np.isfinite(maximum) or not np.isfinite(power) or maximum>1e-9 or power>1e-9:
                            raise ArithmeticError(f'Label verification failed: residual={maximum}, power={power}')
                        full=np.zeros((1152,18),np.complex64);full[ids]=X
                        h['currents'][600+j,fi]=full;h['responses'][600+j,fi]=R;h.flush()
                        mem=memory() if os.name=='nt' else {};row=dict(**current_case,active_rows=len(ids),incident_columns=18,
                            solve_seconds=solve_seconds,case_wall_seconds=time.perf_counter()-case_start,
                            relative_Frobenius_residual=fro,maximum_column_residual=maximum,power_defect=power,
                            peak_working_set_MiB=mem.get('peak_working_set_MiB'),private_commit_MiB=mem.get('private_commit_MiB'),status='accepted')
                        writer.writerow(row);lf.flush();solves.append(solve_seconds);count+=1
                        maxima['maximum_column_residual']=max(maxima['maximum_column_residual'],maximum)
                        maxima['relative_Frobenius_residual']=max(maxima['relative_Frobenius_residual'],fro)
                        maxima['power_defect']=max(maxima['power_defect'],power)
                        if count%100==0:print('LABELS',count,13200,'frequency',FREQ[fi],'new_pattern',j+1,'maxima',maxima,'memory',mem,flush=True)
                    except BaseException:
                        dump(HERE/'numerical_failure.json',dict(**current_case,traceback=traceback.format_exc(),completed_cases=count,stop_without_geometry_deletion=True))
                        raise
            if last==10:
                prefix=dict(status='passed',new_patterns=10,frequency_cases=30,all_frequencies_GHz=list(FREQ),
                    wall_seconds=time.perf_counter()-prefix_start,operator_cache_wall_seconds=operator_seconds,
                    offline_elapsed_seconds=time.perf_counter()-start,mean_solve_seconds=float(np.mean(solves)),
                    solve_seconds=solves,maxima=dict(maxima),baseline_memory=baseline,measured_memory=memory() if os.name=='nt' else {},
                    protocol_sha256=sha(HERE/'generation_protocol.json'),case_csv_sha256=sha(HERE/'label_cases.csv'))
                dump(HERE/'prefix10_audit.json',prefix);print('PREFIX10_PASS',json.dumps(prefix),flush=True)
        h.attrs['generation_complete']=True;h.flush()
    identity=verify_source_identity()
    case_rows=list(csv.DictReader((HERE/'label_cases.csv').open(encoding='utf-8')))
    assert len(case_rows)==13200 and len({(int(r['new_pattern']),float(r['frequency_GHz'])) for r in case_rows})==13200
    assert {(int(r['new_pattern']),float(r['frequency_GHz'])) for r in case_rows}=={(j,f) for j in range(1,4401) for f in FREQ}
    assert all(r['status']=='accepted' for r in case_rows)
    final=dict(status='passed',new_training_patterns=4400,total_training_patterns=5000,unchanged_validation_patterns=100,
        verified_new_frequency_cases=count,maxima=maxima,offline_total_wall_seconds=time.perf_counter()-start,
        operator_cache_wall_seconds=operator_seconds,mean_case_solve_seconds=float(np.mean(solves)),
        total_dense_solve_seconds=float(np.sum(solves)),baseline_memory=baseline,measured_memory=memory() if os.name=='nt' else {},
        original_700_array_identity=True,mask_exclusion_audit_sha256=sha(HERE/'mask_exclusion_audit.json'),
        source_identity_audit_sha256=sha(HERE/'source_array_identity.json'),case_log_sha256=sha(HERE/'label_cases.csv'),
        protocol_sha256=sha(HERE/'generation_protocol.json'),training_h5_sha256=sha(OUTPUT),training_h5_bytes=OUTPUT.stat().st_size,
        offline_wall_boundary='Before copying unchanged labels through final array-identity verification; excludes mask generation and final H5 digest calculation',
        separate_from_network_training_cost=True,numerical_formula_changes=False)
    dump(HERE/'generation_final_audit.json',final);print('GENERATION_PASS',json.dumps(final),flush=True)

def generate_labels(template_path,target):
    """用模板中已提供的4400新增几何重新计算标签，旧600训练/100验证标签原样复制。

    模板不提供测试集；本入口不重新采样几何，因此不需要未公开测试轨道列表。
    同一4400几何、同一18列物理解和独立FFT/传播功率验证，不替换失败几何。
    """
    global SOURCE,OUTPUT,HERE
    SOURCE=Path(template_path)
    HERE=Path(target)
    if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
    HERE.mkdir(parents=True,exist_ok=False)
    OUTPUT=HERE/'training_dataset_5000.h5'
    with h5py.File(SOURCE,'r') as source:
        if len(source['masks'])!=5100 or not np.all(source['split'][:5000]==1) or not np.all(source['split'][5000:]==2):
            raise ValueError('模板必须为5000训练+100验证布局')
        if tuple(int(source.attrs[k]) for k in ('N','Q','M'))!=(N,Q,M) or not np.array_equal(source['freq'][:],FREQ):
            raise ValueError('模板网格必须为N24/Q192/M1、5/7/9GHz')
        masks=source['masks'][600:5000].astype(bool)
    np.save(HERE/'new_training_masks.npy',masks,allow_pickle=False)
    dump(HERE/'mask_exclusion_audit.json',dict(geometry_source='supplied training-template rows600:5000; no random resampling or test-file input',new_training_patterns=ADDED))
    protocol=dict(status='frozen_before_label_solves',generation_seed=SEED,
        geometry_source='supplied training-template rows600:5000',
        generator_sha256=sha(__file__),published_physics_sha256=sha(Path(physics.__file__)),
        source_training_h5_sha256=sha(SOURCE),new_masks_file_sha256=sha(HERE/'new_training_masks.npy'),
        maximum_column_residual=1e-9,maximum_propagating_power_defect=1e-9,
        current_dtype='complex64',response_dtype='complex128',N=N,Q=Q,M=M,frequencies_GHz=list(FREQ),
        stop_on_any_numerical_failure=True,generation_cost_is_offline_and_separate_from_network_training=True)
    dump(HERE/'generation_protocol.json',protocol)
    run()
    return json.loads((HERE/'generation_final_audit.json').read_text(encoding='utf-8'))
