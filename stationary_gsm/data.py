"""只读 HDF5 训练/验证标签，逐图案读取电流，索引从零开始。"""
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]

def load_model(path):
    """读取用户新训练的模型，不反序列化 Python 对象。"""
    with np.load(path, allow_pickle=False) as f:
        return {key:f[key] for key in f.files}

class Labels:
    """masks[T,y,x]；currents[T,f,dof,port]；split=1训练、2验证。

    全电流标签为 η₀J 的原生 complex64，空气行保留为零。
    入射18端口按(m,n)每阶Ex/Ey排列，数据N24/Q192/M1、法向入射。
    """
    def __init__(self,path):
        self.file=h5py.File(path,'r')
        self.N,self.Q,self.M=(int(self.file.attrs[k]) for k in ('N','Q','M'))
        self.freq=self.file['freq'][:]
        self.masks=self.file['masks']
        self.split=self.file['split'][:]
    def __len__(self): return len(self.masks)
    def case(self,index):
        ids=np.flatnonzero(self.file['active'][index])
        currents=self.file['currents'][index][:,ids,:].transpose(1,2,0)
        responses=self.file['responses'][index].transpose(1,2,0)
        return self.masks[index],ids,currents,responses
    def close(self): self.file.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
