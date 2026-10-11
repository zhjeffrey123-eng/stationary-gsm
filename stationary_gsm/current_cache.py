"""逐拓扑读取并打包有效金属电流，避免将所有训练标签占为私有内存。

缓存是本次训练在输出目录现场生成的磁盘文件，不改动 H5 源数据。
有效电流顺序、complex64 数值、逐拓扑 RMS 的归约顺序均沿用原训练代码。
随机批次仍先抽拓扑、再抽该拓扑内的有效电流行、频率和入射端口。
"""
from pathlib import Path
import numpy as np


def build_current_cache(file, cache_ids, rows, target):
    """返回磁盘映射、拓扑起始偏移、原布局视图与 RMS。

    每个拓扑的连续块以 [有效金属行,频率,端口] 的 C 次序保存，
    再转置成原代码使用的 [频率,有效金属行,端口] 视图。
    原 NumPy 高级索引正好使用这种物理布局，因此验证误差归约也保持原顺序。
    每个规模只缓存它真正使用的 N 个训练图案及固定 100 个验证图案。
    """
    cache_ids=np.asarray(cache_ids,int)
    counts=np.asarray([len(v) for v in rows],int)
    nf=len(file['freq']);ports=file['currents'].shape[-1]
    offsets=np.full(len(rows),-1,np.int64)
    sizes=counts[cache_ids]*nf*ports
    offsets[cache_ids]=np.r_[0,np.cumsum(sizes[:-1])]
    size=int(np.sum(sizes))
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    packed=np.memmap(target,mode='w+',dtype=np.complex64,shape=(size,))
    views=[None]*len(rows)
    rms=np.full((len(rows),nf,ports),np.nan)
    for t in cache_ids:
        current=file['currents'][int(t)][:,rows[t],:]
        first=int(offsets[t]);last=first+int(counts[t]*nf*ports)
        packed[first:last]=current.transpose(1,0,2).ravel(order='C')
        views[t]=packed[first:last].reshape((counts[t],nf,ports),order='C').transpose(1,0,2)
        # 与旧缓存逐位相同，不能用不同分块/单精度累加改变归一化。
        rms[t]=np.sqrt(np.sum(abs(current.astype(complex))**2,axis=1)/counts[t])
    packed.flush()
    return packed,offsets,views,rms


def sample_labels(packed,offsets,counts,topology,frequency,local_row,port):
    """批次只改变取数途径，仍逐位获取原 complex64 标签后转 complex128。"""
    index=offsets[topology]+(local_row*3+frequency)*18+port
    return np.asarray(packed[index],complex)
