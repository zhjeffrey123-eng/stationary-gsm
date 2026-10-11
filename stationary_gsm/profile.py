"""固定嵌套训练子集与真实Windows进程资源计量，不读取测试/归档结果。

训练循环来自实际训练规模实验。子集仅由固定seed决定，验证图案不变。
每次CLI调用须新Python进程。PeakWorkingSetSize为OS累计峰值，不由数组估计。
"""
from pathlib import Path
import csv, ctypes, hashlib, json, os, platform, sys, time
from ctypes import wintypes
import h5py
import numpy as np
import scipy
from threadpoolctl import threadpool_info
from . import network as net

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
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
        raise RuntimeError('本次资源实验固定使用 Windows OS PeakWorkingSetSize')
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


def train_profile(labels_path,target,training_count=600,hidden=64,epochs=80,batches=128,
                  batch=4096,seed=120026,subset_seed=20261011,validation_interval=5,
                  checkpoint_epochs=(10,20,40,60,80)):
    """固定预算检查点，与验证最佳权重分别保存；不以测试误差选权重。

    wall从打开H5前开始，包括加载、缓存、RMS、初始化、训练、验证和检查点保存。
    checkpoint采样在权重写盘后；后续累计时间包含先前日志/哈希/CSV写盘开销。
    wall不含Python imports或任何测试评价；OS峰值内存包含imports及整个进程。
    """
    if os.name!='nt':
        raise RuntimeError('train-profile使用Windows实际PeakWorkingSetSize；其他平台可使用train')
    if hidden not in (64,128) or min(training_count,epochs,batches,batch,validation_interval)<1:
        raise ValueError('hidden必须为64或128，其余预算必须为正整数')
    if len(set(checkpoint_epochs))!=len(checkpoint_epochs) or any(e<1 or e>epochs for e in checkpoint_epochs):
        raise ValueError('checkpoint_epochs必须为互异的1..epochs整数')
    n=int(training_count)
    target=Path(target)
    target.mkdir(parents=True,exist_ok=False)
    cfg=dict(architecture=[71,hidden,hidden,2],train_seed=seed,subset_seed=subset_seed,
        training_count=n,terminal_epoch=epochs,batches_per_epoch=batches,coefficients_per_batch=batch,
        checkpoint_epochs=list(checkpoint_epochs),validation_interval=validation_interval,
        loss='mean field-relative squared current coefficient error',
        normalization='global RMS of selected training patterns only; analytic incident phase removed',
        schedule='cosine 0.001 to 0.0001 over terminal_epoch',
        time_scope='cumulative perf_counter from before H5 opening through checkpoint weight save; includes label/cache loading, initialization, training, validation, checkpoint saves, and prior routine logging/hash/CSV overhead; excludes imports and test evaluation',
        memory_scope='Windows GetProcessMemoryInfo PeakWorkingSetSize, process lifetime peak in MiB; includes interpreter/imports, labels/cache, training/validation and saves; baseline WorkingSetSize before H5 opening; no sampled or estimated peak')
    baseline = memory()
    rng = np.random.default_rng(cfg['train_seed'])
    timer = time.perf_counter()
    # 以下第一步就是打开H5，完整训练成本从这一行之前计时。
    with h5py.File(labels_path, 'r') as d:
        split=d['split'][:]
        pool=np.flatnonzero(split==1)
        validation=np.flatnonzero(split==2)
        if not len(validation) or np.any(~np.isin(split,[1,2])) or n>len(pool):
            raise ValueError('标签只允许训练split1/验证split2，training_count不能超过训练池')
        permutation=np.random.default_rng(subset_seed).permutation(pool)
        training=pool if n==len(pool) else permutation[:n]
        cache_ids=np.r_[training,validation]
        masks = d['masks'][:]
        g = net.geometry(masks)
        frequencies = d['freq'][:]
        N, Q, M = (int(d.attrs[k]) for k in ('N','Q','M'))
        rows = [np.flatnonzero(a) for a in d['active'][:]]
        counts = np.asarray([len(a) for a in rows])
        X = [None] * len(masks)
        rms = np.full((len(masks),len(frequencies),18), np.nan)
        for t in cache_ids:
            current = d['currents'][int(t)][:,rows[t],:]
            X[t] = current
            rms[t] = np.sqrt(np.sum(abs(current.astype(complex))**2,axis=1)/counts[t])
    scale = np.float32(np.sqrt(np.mean(rms[training]**2)))
    model = dict(N=np.asarray(N),Q=np.asarray(Q),M=np.asarray(M),output_scale=np.asarray(scale))
    widths = cfg['architecture']
    for j in range(3):
        model[f'W{j+1}'] = (rng.standard_normal((widths[j+1],widths[j])) * np.sqrt(2/widths[j])).astype(np.float32)
        model[f'b{j+1}'] = np.zeros(widths[j+1], np.float32)
    names = ['W1','b1','W2','b2','W3','b3']
    mom,vel = ({k:np.zeros_like(model[k]) for k in names} for _ in range(2))
    iteration,best,best_epoch,history,resources,manifest = 0,np.inf,0,[],[],[]
    cfg['training_zero_based_indices']=training.tolist()
    cfg['validation_zero_based_indices']=validation.tolist()
    dump(target/'training_protocol.json',cfg)
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
            label = np.asarray([X[t][f,p,j] for t,f,p,j in zip(topo,fi,pos,port)],complex)/phase
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
        val = float('nan')
        if epoch==1 or epoch%validation_interval==0 or epoch==epochs:
            errors=[]
            for t in validation:
                truth = np.transpose(X[t], (1,2,0))
                pred = net.predict(model,masks[t],frequencies,g[:,:,t:t+1])
                errors.extend((np.sum(abs(pred-truth)**2,axis=0)/
                    np.maximum(np.sum(abs(truth)**2,axis=0),np.finfo(float).eps)).ravel())
            val = float(np.mean(errors))
            if val<best:
                best,best_epoch=val,epoch
                np.savez_compressed(target/'validation_best.npz',**model,bestEpoch=np.asarray(epoch))
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
            validation_relative_squared_current=val,learning_rate=lr,training_wall_seconds=elapsed))
        if epoch in checkpoint_epochs:
            # 哈希和CSV审计写入不放在刚刚截取的checkpoint训练计时中。
            manifest.append(dict(**record,checkpoint=str(path.resolve()),checkpoint_sha256=sha(path)))
            write_csv(target/'checkpoints.csv',manifest)
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
        time_scope=cfg['time_scope'],memory_scope=cfg['memory_scope'])
    dump(target/'training_summary.json',summary)
    print('TRAIN_COMPLETE',n,flush=True)
    return summary
