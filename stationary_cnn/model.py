"""论文唯一 CNN 的可发布实现。Torch 由调用方传入，静态导入不加载 GPU。

输入 mask 为原 y,x 的8×8金属图，query依次为(f−7)/2、m、n、±1偏振。
几何分支五层周期3×3卷积，query分支为各层提供gamma/beta。
输出36通道的编号及24网格回排统一由同包indexing定义。
"""
from __future__ import annotations
import hashlib
import numpy as np
from . import indexing

ARCHITECTURE="CNN32_periodic_matrix"


def specifications():
    result=[]
    for i in range(5):
        channels=1 if i==0 else 32
        result += [(f"convs.{i}.weight",(32,channels,3,3),channels*9),(f"convs.{i}.bias",(32,),None)]
    result += [("query_layers.0.weight",(32,4),4),("query_layers.0.bias",(32,),None),
               ("query_layers.1.weight",(320,32),32),("query_layers.1.bias",(320,),None),
               ("head.weight",(36,32,1,1),32),("head.bias",(36,),None)]
    return result


def parameter_count():return int(sum(np.prod(shape) for _,shape,_ in specifications()))


def initial_state(seed=120026):
    # 独立NumPy RNG。最终调制投影置零，使首个前向gamma=beta=0。
    rng=np.random.default_rng(seed);arrays={}
    for key,shape,fan_in in specifications():
        if fan_in is None or key.startswith("query_layers.1."):arrays[key]=np.zeros(shape,np.float32)
        else:arrays[key]=(rng.standard_normal(shape)*np.sqrt(2/fan_in)).astype(np.float32)
    return arrays,rng.bit_generator.state


def core_digest(arrays):
    value=hashlib.sha256()
    for key,_,_ in specifications():
        array=np.ascontiguousarray(arrays[key]);value.update(key.encode());value.update(str(array.shape).encode())
        value.update(array.dtype.str.encode());value.update(array.tobytes())
    return value.hexdigest()


def make_model(torch,arrays,device="cuda"):
    class PeriodicCurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs=torch.nn.ModuleList([torch.nn.Conv2d(1 if i==0 else 32,32,3,padding=1,padding_mode="circular") for i in range(5)])
            self.query_layers=torch.nn.ModuleList([torch.nn.Linear(4,32),torch.nn.Linear(32,320)])
            self.head=torch.nn.Conv2d(32,36,1)
        def forward(self,mask,query,active):
            modulation=self.query_layers[1](torch.nn.functional.gelu(self.query_layers[0](query),approximate="none")).reshape(-1,5,2,32)
            hidden=mask
            for i,conv in enumerate(self.convs):
                gamma,beta=modulation[:,i,0,:,None,None],modulation[:,i,1,:,None,None]
                hidden=torch.nn.functional.gelu((1+gamma)*conv(hidden)+beta,approximate="none")
            return self.head(hidden)*active
    model=PeriodicCurrent();state=model.state_dict()
    if set(state)!=set(arrays) or sum(p.numel() for p in model.parameters())!=49220:raise ValueError("CNN schema mismatch")
    model.load_state_dict({key:torch.from_numpy(np.ascontiguousarray(arrays[key])) for key in state},strict=True)
    return model.to(device=device,dtype=torch.float32)


class Predictor:
    """predict(mask,frequencies)返回原物理接口的active_rows×18×nf复电流。

    active_rows回调由公开stationary_gsm.physics.active或调用方提供；此核心
    不搜索本机目录、不含私有数据路径。缩放及相位遵循统一数值约定。
    """
    def __init__(self,torch,net,output_scale,active_rows,device="cuda"):
        self.torch,self.net,self.scale,self.active_rows,self.device=torch,net,np.float32(output_scale),active_rows,device
    def predict(self,mask,frequencies):
        mask=np.asarray(mask,dtype=bool);frequencies=np.asarray(frequencies,dtype=float).reshape(-1)
        if mask.shape!=(8,8):raise ValueError("8x8 y,x mask required")
        ids=np.asarray(self.active_rows(mask),dtype=np.int64);fields=18*len(frequencies)
        inputs=dict(mask=np.repeat((mask.astype(np.float32)*2-1)[None,None],fields,axis=0),query=indexing.queries(frequencies),
                    active=np.repeat(indexing.active_host_mask(ids)[None],fields,axis=0))
        with self.torch.no_grad():
            batch={key:self.torch.from_numpy(np.ascontiguousarray(array)).to(self.device) for key,array in inputs.items()}
            host=self.net(batch["mask"],batch["query"],batch["active"]).cpu().numpy()
        return indexing.predict_from_host(host,len(frequencies),ids,self.scale)


def export_model(net,output_scale,N=24,M=1,Q=192):
    result={key:value.detach().cpu().numpy().copy() for key,value in net.state_dict().items()}
    result.update(N=np.asarray(N),M=np.asarray(M),Q=np.asarray(Q),output_scale=np.asarray(output_scale),architecture=np.asarray(ARCHITECTURE))
    return result


def load_predictor(path,torch,active_rows,device="cuda"):
    with np.load(path,allow_pickle=False) as stored:
        if str(stored["architecture"])!=ARCHITECTURE or tuple(int(stored[key]) for key in ("N","M","Q"))!=(24,1,192):raise ValueError("CNN model metadata mismatch")
        arrays={key:stored[key].copy() for key,_,_ in specifications()};scale=np.float32(stored["output_scale"])
    model=make_model(torch,arrays,device);model.eval()
    return Predictor(torch,model,scale,active_rows,device)
