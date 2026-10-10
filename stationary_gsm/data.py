"""Python 原生数据接口。标签用 HDF5，权重及参考结果用无 pickle 的 NPZ。"""
from pathlib import Path
import json
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path('output/awpl/revision20_metal_paper')


def load_bundle(path):
    """读取嵌套结果树。NPZ 数组可单独检查，JSON 描述变量名和嵌套结构。

    allow_pickle=False 防止对象反序列化。字典、列表、数组和标量分别保留。
    根目录的普通权重 NPZ 没有 __schema__，直接返回按名称索引的字典。
    """
    with np.load(path,allow_pickle=False) as archive:
        if '__schema__' not in archive.files:
            return {k:archive[k] for k in archive.files}
        def decode(node):
            kind = node['kind']
            if kind=='array':
                return archive[node['key']]
            if kind=='dict':
                return {k:decode(v) for k,v in node['items'].items()}
            if kind=='list':
                return [decode(v) for v in node['items']]
            if kind=='object_array':
                arr = np.empty(node['shape'],object)
                for i,item in enumerate(node['items']):
                    arr.flat[i] = decode(item)
                return arr
            if kind=='structured_array':
                descr = [tuple([item[0],item[1]]+([tuple(item[2])] if len(item)>2 else []))
                         for item in node['dtype']]
                arr = np.empty(node['shape'],dtype=np.dtype(descr))
                for name,item in node['fields'].items():
                    arr[name] = decode(item)
                return arr
            if kind=='scalar':
                return node['value']
            raise ValueError(kind)
        return decode(json.loads(str(archive['__schema__'])))


def save_bundle(path,values):
    """保存可审计的数值数组及 JSON 结构描述，不写入 Python pickle。"""
    arrays = {}
    def encode(value,logical_path='root'):
        if isinstance(value,dict):
            return {'kind':'dict','items':{k:encode(v,logical_path+'/'+k) for k,v in value.items()}}
        if isinstance(value,(list,tuple)):
            return {'kind':'list','items':[encode(v,logical_path+'/'+str(i)) for i,v in enumerate(value)]}
        if isinstance(value,np.ndarray):
            if value.dtype.fields:
                return {'kind':'structured_array','shape':value.shape,'dtype':value.dtype.descr,
                        'fields':{name:encode(value[name],logical_path+'/'+name) for name in value.dtype.names}}
            if value.dtype==object:
                return {'kind':'object_array','shape':value.shape,'items':[
                    encode(v,logical_path+'/'+str(i)) for i,v in enumerate(value.flat)]}
            key = logical_path.removeprefix('root/').replace('/','__')
            if key in arrays:
                key += '__'+str(len(arrays))
            arrays[key] = value
            return {'kind':'array','key':key}
        if isinstance(value,np.generic):
            return encode(np.asarray(value),logical_path)
        if isinstance(value,complex) or (isinstance(value,float) and not np.isfinite(value)):
            return encode(np.asarray(value),logical_path)
        if value is None or isinstance(value,(str,bool,int,float)):
            return {'kind':'scalar','value':value}
        raise TypeError(type(value))
    schema = encode(values)
    arrays['__schema__'] = np.asarray(json.dumps(schema,ensure_ascii=False))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)


class Labels:
    """按拓扑读取标签，不把所有参考电流同时读入内存。

    HDF5 布局为 masks[T,y,x]、active[T,dof]、currents[T,f,dof,port]、
    responses[T,f,out_port,in_port]，复数用 HDF5 原生 complex dtype。
    原始拓扑编号从 1 开始，Python 数组索引从 0 开始，两个数据集编号分开。
    """
    def __init__(self,path):
        self.file = h5py.File(path,'r')
        self.N,self.Q,self.M = (int(self.file.attrs[k]) for k in ('N','Q','M'))
        self.freq = self.file['freq'][:]
        self.masks = self.file['masks']
        self.split = self.file['split'][:] if 'split' in self.file else None
    def __len__(self):
        return len(self.masks)
    def case(self,index):
        ids = np.flatnonzero(self.file['active'][index])
        # h5py 先按单一拓扑读取，再取有效金属行，返回 [dof,port,f]。
        currents = self.file['currents'][index][:,ids,:].transpose(1,2,0)
        responses = self.file['responses'][index].transpose(1,2,0)
        return self.masks[index],ids,currents,responses
    def close(self):
        self.file.close()
    def __enter__(self):
        return self
    def __exit__(self,*_):
        self.close()


def load_masks(root=ROOT):
    """CSV 像素采用原始列优先排列，恢复后行是 y，列是 x。"""
    table = np.loadtxt(Path(root)/'output/pec8x8_trial/masks_and_split.csv',delimiter=',')
    return np.stack([row[2:].reshape((8,8),order='F') for row in table]).astype(bool)
