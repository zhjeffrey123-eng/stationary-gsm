"""固定训练子集、磁盘映射标签和进程资源计量。"""
from pathlib import Path
import csv, ctypes, hashlib, json, os, platform, sys, time
from ctypes import wintypes
import h5py
import numpy as np
import scipy
from threadpoolctl import threadpool_info
from . import network as net
from .current_cache import build_current_cache, sample_labels

SIZES=(100,300,600,1000,2000,5000)
EPOCHS=(10,20,40,60,80)

def sha(path):
    """流式计算文件校验值，不一次载入大 H5。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

class ProcessMemoryCounters(ctypes.Structure):
    # SIZE_T 在 64 位 Windows 上必须是指针宽度，不能用 DWORD 替代。
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t)]

def memory():
    """实际进程工作集和 OS 累计峰值，均以 MiB 表示。"""
    if os.name != 'nt':
        raise RuntimeError('资源计量使用 Windows OS PeakWorkingSetSize')
    c = ProcessMemoryCounters()
    c.cb = ctypes.sizeof(c)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_MiB': c.WorkingSetSize / 2**20,
            'peak_working_set_MiB': c.PeakWorkingSetSize / 2**20,
            'private_commit_MiB': c.PagefileUsage / 2**20,
            'peak_private_commit_MiB': c.PeakPagefileUsage / 2**20}

def training_configuration(labels_path,n,hidden,epochs,batches,batch,seed,subset_seed,
                           validation_interval,checkpoint_epochs):
    """只读取split/网格元数据，先冻结子集和验证图案，不读取任何测试文件。"""
    if hidden not in (64,128) or min(n,epochs,batches,batch,validation_interval)<1:
        raise ValueError('hidden为64或128，其余预算为正整数')
    if len(set(checkpoint_epochs))!=len(checkpoint_epochs) or any(e<1 or e>epochs for e in checkpoint_epochs):
        raise ValueError('检查点为互异的1..epochs整数')
    with h5py.File(labels_path,'r') as d:
        split=d['split'][:]
        pool=np.flatnonzero(split==1)
        validation=np.flatnonzero(split==2)
        if len(pool)!=5000 or len(validation)!=100 or len(split)!=5100 or np.any(~np.isin(split,[1,2])):
            raise ValueError('扩展标签必须只含5000训练及100验证图案')
        if not np.array_equal(pool,np.arange(5000)) or not np.array_equal(validation,np.arange(5000,5100)):
            raise ValueError('布局必须为train0:5000、validation5000:5100')
        if not bool(d.attrs.get('generation_complete',False)):
            raise ValueError('标签尚未完整生成')
        if tuple(int(d.attrs[k]) for k in ('N','Q','M'))!=(24,192,1) or not np.array_equal(d['freq'][:],[5.,7.,9.]):
            raise ValueError('标签网格必须为N24/Q192/M1、5/7/9GHz')
        if n>len(pool):raise ValueError('training_count不能超过训练池')
        # N100/N300保持旧600训练图案上的嵌套置换，较大规模按原训练顺序取前缀。
        permutation=np.random.default_rng(subset_seed).permutation(pool[:600])
        training=permutation[:n] if n<600 else pool[:n]
    return dict(architecture=[71,hidden,hidden,2],train_seed=seed,subset_seed=subset_seed,
        training_count=n,terminal_epoch=epochs,batches_per_epoch=batches,coefficients_per_batch=batch,
        checkpoint_epochs=list(checkpoint_epochs),validation_interval=validation_interval,
        training_zero_based_indices=training.tolist(),validation_zero_based_indices=validation.tolist(),
        loss='mean field-relative squared current coefficient error',
        normalization='selected-training-only global RMS; analytic incident phase removed',
        time_scope='perf_counter starts before H5/cache loading; includes geometry, active-current disk cache creation and flush, RMS, initialization, training, scheduled validation, checkpoint saves and earlier routine logging; excludes imports, initial split/protocol freezing, label generation and test evaluation',
        memory_scope='Windows GetProcessMemoryInfo PeakWorkingSetSize in MiB, cumulative process peak including imports, protocol setup, labels/cache, training, validation and saves; baseline before training H5 loading; each size uses a fresh process')

def train_profile(labels_path,target,training_count=600,hidden=64,epochs=80,batches=128,
                  batch=4096,seed=120026,subset_seed=20261011,validation_interval=5,
                  checkpoint_epochs=EPOCHS):
    """默认预算与固定检查点；验证最佳权重另存，不替换预定检查点。

    相位剥离、逐拓扑RMS双精度归约、AdamW与全局梯度裁剪保持原计算顺序。
    磁盘缓存只替换标签存储和取数途径，不改变所抽取的complex64系数。
    """
    if os.name!='nt':raise RuntimeError('train-profile资源计量要求Windows')
    n=int(training_count)
    checkpoint_epochs=tuple(checkpoint_epochs)
    cfg=training_configuration(labels_path,n,hidden,epochs,batches,batch,seed,subset_seed,
                               validation_interval,checkpoint_epochs)
    target=Path(target)
    target.mkdir(parents=True,exist_ok=False)
    dump(target/'training_protocol.json',cfg)
    baseline = memory()
    rng = np.random.default_rng(cfg['train_seed'])
    training = np.asarray(cfg['training_zero_based_indices'], int)
    validation = np.asarray(cfg['validation_zero_based_indices'], int)
    cache_ids = np.r_[training, validation]
    timer = time.perf_counter()
    # 以下第一步就是打开H5，完整训练成本从这一行之前计时。
    with h5py.File(labels_path, 'r') as d:
        masks = d['masks'][:]
        g = net.geometry(masks)
        frequencies = d['freq'][:]
        N, Q, M = (int(d.attrs[k]) for k in ('N','Q','M'))
        rows = [np.flatnonzero(a) for a in d['active'][:]]
        counts = np.asarray([len(a) for a in rows])
        packed,offsets,X,rms=build_current_cache(d,cache_ids,rows,target/'active_current_cache.bin')
    scale = np.float32(np.sqrt(np.mean(rms[training]**2)))
    model = dict(N=np.asarray(N),Q=np.asarray(Q),M=np.asarray(M),output_scale=np.asarray(scale))
    widths = cfg['architecture']
    for j in range(3):
        model[f'W{j+1}'] = (rng.standard_normal((widths[j+1],widths[j])) * np.sqrt(2/widths[j])).astype(np.float32)
        model[f'b{j+1}'] = np.zeros(widths[j+1], np.float32)
    names = ['W1','b1','W2','b2','W3','b3']
    mom,vel = ({k:np.zeros_like(model[k]) for k in names} for _ in range(2))
    iteration,best,best_epoch,history,resources,manifest = 0,np.inf,0,[],[],[]
    validation_rows=[];validation_summary=[]
    print('TRAIN_BEGIN', n, 'cache',len(cache_ids), 'baseline_MiB',baseline['working_set_MiB'],
          'threads',threadpool_info(),flush=True)
    for epoch in range(1,epochs+1):
        lr = .0001+(.001-.0001)*(1+np.cos(np.pi*(epoch-1)/max(epochs-1,1)))/2
        accumulated = 0.
        for _ in range(batches):
            topo = rng.choice(training,batch)
            pos = np.floor(rng.random(batch)*counts[topo]).astype(int)
            row = np.asarray([rows[t][p] for t,p in zip(topo,pos)])
            fi,port = rng.integers(0,len(frequencies),batch),rng.integers(0,18,batch)
            u,phase = net.features(g,row,topo,fi,port,frequencies,N)
            label = sample_labels(packed,offsets,counts,topo,fi,pos,port)/phase
            target_value = np.vstack([label.real,label.imag]).astype(np.float32)/scale
            weight = (float(scale)/np.maximum(rms[topo,fi,port],1e-6))**2
            loss,grad = net.gradients(model,u,target_value,weight.astype(np.float32))
            norm = np.sqrt(sum(np.sum(v.astype(float)**2) for v in grad.values()))
            fac = min(1.,5/(norm+1e-12))
            iteration += 1
            for k in names:
                value = grad[k]*fac
                mom[k] = .9*mom[k]+.1*value
                vel[k] = .999*vel[k]+.001*value*value
                model[k] = (model[k]*(1-lr*1e-5)-lr*(mom[k]/(1-.9**iteration))/
                    (np.sqrt(vel[k]/(1-.999**iteration))+1e-8)).astype(np.float32)
            accumulated += loss
        val = float('nan');val_current_median=float('nan');matrix_errors=[]
        if epoch==1 or epoch%validation_interval==0 or epoch==epochs or epoch in checkpoint_epochs:
            errors=[]
            for validation_index,t in enumerate(validation):
                truth = np.transpose(X[t], (1,2,0))
                pred = net.predict(model,masks[t],frequencies,g[:,:,t:t+1])
                errors.extend((np.sum(abs(pred-truth)**2,axis=0)/
                    np.maximum(np.sum(abs(truth)**2,axis=0),np.finfo(float).eps)).ravel())
                if epoch in checkpoint_epochs:
                    for fi,f in enumerate(frequencies):
                        denominator=float(np.linalg.norm(truth[:,:,fi]))
                        error=float(np.linalg.norm(pred[:,:,fi]-truth[:,:,fi])/denominator)
                        finite=bool(np.isfinite(error))
                        if not finite:error=float('inf')
                        matrix_errors.append(error)
                        validation_rows.append(dict(training_count=n,epoch=epoch,
                            validation_index=int(validation_index),
                            validation_H5_zero_based_index=int(t),frequency_GHz=float(f),
                            current_error=error,finite=finite,reference_current_norm_F=denominator,
                            active_rows=int(counts[t]),incident_columns=18))
            val = float(np.mean(errors))
            if val<best:
                best,best_epoch=val,epoch
                np.savez_compressed(target/'validation_best.npz',**model,bestEpoch=np.asarray(epoch))
            if epoch in checkpoint_epochs:
                val_current_median=float(np.median(matrix_errors))
        if epoch in checkpoint_epochs:
            path = target/f'checkpoint_{epoch}.npz'
            np.savez_compressed(path,**model,saved_epoch=np.asarray(epoch),training_count=np.asarray(n))
        elapsed = time.perf_counter()-timer
        mem = memory()
        record = dict(training_count=n,epoch=epoch,training_wall_seconds=elapsed,
            peak_working_set_MiB=mem['peak_working_set_MiB'],working_set_MiB=mem['working_set_MiB'],
            baseline_working_set_MiB=baseline['working_set_MiB'],
            private_commit_MiB=mem['private_commit_MiB'],peak_private_commit_MiB=mem['peak_private_commit_MiB'])
        resources.append(record)
        history.append(dict(training_count=n,epoch=epoch,train_relative_squared_current=accumulated/batches,
            validation_relative_squared_current=val,validation_current_median_relative=val_current_median,
            learning_rate=lr,training_wall_seconds=elapsed))
        if epoch in checkpoint_epochs:
            # 哈希和CSV审计写入不放在刚刚截取的checkpoint训练计时中。
            manifest.append(dict(**record,checkpoint=path.name,checkpoint_sha256=sha(path)))
            write_csv(target/'checkpoints.csv',manifest)
            validation_summary.append(dict(training_count=n,epoch=epoch,
                current_median_relative=val_current_median,current_p95_relative=float(np.percentile(matrix_errors,95)),
                n_conditions=len(matrix_errors),nonfinite_failures=int(np.sum(~np.isfinite(matrix_errors))),
                checkpoint_sha256=sha(path)))
            write_csv(target/'validation_condition_errors.csv',validation_rows)
            write_csv(target/'validation_checkpoint_summary.csv',validation_summary)
        write_csv(target/'training_history.csv',history)
        write_csv(target/'resource_history.csv',resources)
        print('N',n,'epoch',epoch,'train',history[-1]['train_relative_squared_current'],
              'validation',val,'wall',round(elapsed,3),'peakMiB',round(mem['peak_working_set_MiB'],3),flush=True)
    summary=dict(training_count=n,architecture=widths,parameter_count=sum(model[k].size for k in names),
        epochs=epochs,updates=iteration,coefficients_sampled=iteration*batch,
        train_indices=training.tolist(),validation_indices=validation.tolist(),
        normalization_scale=float(scale),validation_best_epoch=best_epoch,
        validation_best_squared_error=best,protocol_sha256=sha(target/'training_protocol.json'),
        source_sha256=sha(__file__),network_source_sha256=sha(Path(net.__file__)),
        environment=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,
                         scipy=scipy.__version__,h5py=h5py.__version__,threadpools=threadpool_info()),
        checkpoint_records=manifest,status='training_complete',test_evaluated_during_training=False,
        validation_checkpoint_records=validation_summary,cache_file_bytes=int(packed.nbytes),
        cache_source_sha256=sha(Path(__file__).with_name('current_cache.py')),
        time_scope=cfg['time_scope'],memory_scope=cfg['memory_scope'])
    dump(target/'training_summary.json',summary)
    print('TRAIN_COMPLETE',n,flush=True)
    return summary
