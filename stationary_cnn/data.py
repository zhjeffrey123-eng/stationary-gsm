"""通用H5→file-backed去相位缓存。所有currents读取都限定单幅图案。

未压缩全电流数组约2.5GB，不允许source['currents'][:N]整批载入。去相位后的active电流
存为complex64 memmap；每个频率的真解norm及train-only RMS用float64逐图计算。
验证原电流另存小型raw memmap，使验证不把量化后的target当成真解。
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from . import indexing
from .objective import phase_table


def file_sha256(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):result.update(block)
    return result.hexdigest()


def load_dataset(data_path,output_dir,train_count,validation_start=5000,validation_count=100,expected_data_sha256=None):
    import h5py
    data_path=Path(data_path);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    if train_count<=0 or train_count>validation_start:raise ValueError("Train and validation must be disjoint")
    # 外层可先校验source SHA。这里不从路径名推断数据身份。
    data_sha=expected_data_sha256 or file_sha256(data_path)
    selected=np.r_[np.arange(train_count),np.arange(validation_start,validation_start+validation_count)]
    phase=phase_table();phase_sha=hashlib.sha256(phase.tobytes()).hexdigest()
    demod_path=output_dir/"demod_current_cache.bin";rawval_path=output_dir/"validation_reference.bin";meta_path=output_dir/"cache_metadata.npz"
    with h5py.File(data_path,"r") as source:
        masks=source["masks"][selected];active=source["active"][selected]
        rows=[np.flatnonzero(value) for value in active];counts=np.asarray([len(ids) for ids in rows],dtype=np.int64)
        frequencies=source["freq"][:];metadata={key:int(source.attrs[key]) for key in ("N","Q","M")}
        if tuple(metadata[key] for key in ("N","M","Q"))!=(24,1,192) or not np.array_equal(frequencies,np.asarray([5.,7.,9.])):raise ValueError("Fixed paper discretization/frequency mismatch")
        if np.any(counts==0):raise ValueError("No active-current case allowed")
        sizes=counts*54;offsets=np.r_[0,np.cumsum(sizes)]
        val_offsets=np.r_[0,np.cumsum(sizes[train_count:])]
        reusable=False
        if meta_path.exists() and demod_path.exists() and rawval_path.exists():
            with np.load(meta_path,allow_pickle=False) as meta:
                reusable=(str(meta["data_sha256"])==data_sha and str(meta["phase_sha256"])==phase_sha and
                    str(meta["loader_sha256"])==file_sha256(__file__) and np.array_equal(meta["selected"],selected) and
                    demod_path.stat().st_size==int(offsets[-1])*8 and rawval_path.stat().st_size==int(val_offsets[-1])*8)
                if reusable:norm_squared=meta["norm_squared"].copy();scale=np.float32(meta["scale"])
        if not reusable:
            # 两个文件的每个切片只指向当前图案，赋值后不保留大匿名数组。
            cache=np.memmap(demod_path,mode="w+",dtype=np.complex64,shape=(int(offsets[-1]),))
            rawval=np.memmap(rawval_path,mode="w+",dtype=np.complex64,shape=(int(val_offsets[-1]),))
            norm_squared=np.empty((len(selected),3),np.float64);scale_sum=0.
            for local,original in enumerate(selected):
                ids=rows[local]
                # 必须是单个整数图案索引：不能替换为[:train_count]。
                full=source["currents"][int(original)]
                truth=full[:,ids,:].transpose(1,0,2)
                truth128=truth.astype(np.complex128)
                norm_squared[local]=np.sum(np.abs(truth128)**2,axis=(0,2))
                if not np.all(np.isfinite(norm_squared[local])) or not np.all(norm_squared[local]>0):raise ArithmeticError("Invalid source full-case norm")
                demod=truth128/phase[ids,None,:]
                view=cache[offsets[local]:offsets[local+1]].reshape(len(ids),3,18)
                view.real=demod.real.astype(np.float32);view.imag=demod.imag.astype(np.float32)
                if local<train_count:scale_sum+=float(np.sum(norm_squared[local]/(len(ids)*18)))
                else:rawval[val_offsets[local-train_count]:val_offsets[local-train_count+1]]=truth.ravel()
                del view,demod,truth128,truth,full
            scale=np.float32(np.sqrt(scale_sum/(train_count*3)))
            cache.flush();rawval.flush();del cache,rawval
            np.savez_compressed(meta_path,data_sha256=np.asarray(data_sha),phase_sha256=np.asarray(phase_sha),loader_sha256=np.asarray(file_sha256(__file__)),
                selected=selected,counts=counts,offsets=offsets,val_offsets=val_offsets,norm_squared=norm_squared,scale=np.asarray(scale),**metadata)
    cache=np.memmap(demod_path,mode="r",dtype=np.complex64,shape=(int(offsets[-1]),))
    rawval=np.memmap(rawval_path,mode="r",dtype=np.complex64,shape=(int(val_offsets[-1]),))
    demod_views=[cache[offsets[i]:offsets[i+1]].reshape(counts[i],3,18) for i in range(len(selected))]
    validation_views=[rawval[val_offsets[i]:val_offsets[i+1]].reshape(counts[train_count+i],3,18) for i in range(validation_count)]
    return dict(masks=masks,rows=rows,counts=counts,selected=selected,demod_memmap=cache,demod_views=demod_views,
        validation_memmap=rawval,validation_views=validation_views,scale=scale,norm_squared=norm_squared,
        host_active=np.stack([indexing.active_host_mask(ids) for ids in rows]),query54=indexing.queries(frequencies),phase=phase,
        frequencies=frequencies,n_train=train_count,validation_count=validation_count,
        cache_reused=reusable,demod_cache_bytes=demod_path.stat().st_size,raw_validation_cache_bytes=rawval_path.stat().st_size,**metadata)
