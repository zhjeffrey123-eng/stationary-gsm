"""几何对齐的 71–128–128–2 共享电流网络，NumPy 前向、反向和 AdamW。

所有基函数共用权重。监督目标去除解析入射相位，再用训练集全局 RMS 缩放。
网络训练仅含相对电流 MSE，不含 PEC 残差。后续物理修正由独立模块完成。
"""
from pathlib import Path
import csv
import time
import numpy as np
from scipy.special import erf
from .physics import active
from .data import Labels


def geometry(masks):
    masks = np.asarray(masks)
    if masks.ndim==2:
        masks = masks[None]
    g = np.empty((64,64,len(masks)),np.float32)
    for t,mask in enumerate(masks):
        for ay in range(8):
            for ax in range(8):
                shifted = np.roll(mask,(-ay,-ax),axis=(0,1))
                g[:,ax+8*ay,t] = 2*shifted.ravel(order='C').astype(np.float32)-1
    return g


def features(g,row,topology,fi,port,frequencies,N):
    """全部索引是零基。每列表示某拓扑、频率、入射端口的一个电流基。"""
    row,topology,fi,port = (np.asarray(v,int).ravel() for v in (row,topology,fi,port))
    component = row // (N*N)
    r = row % (N*N)
    ix,iy,s = r % N,r // N,N//8
    x,y = (ix+.5*(component==1))/N,(iy+.5*(component==0))/N
    anchor = ix//s + 8*(iy//s)
    mode = port//2
    # Floquet 整数次序同求解器，m 最快变化，每阶连续排列 Ex/Ey。
    mm,nn = mode % 3-1,mode//3-1
    # 原训练编码中 Ex=-1、Ey=+1，不能按物理方向自行反转符号。
    pol = 2*(port % 2)-1
    phase = np.exp(1j*2*np.pi*(mm*x+nn*y))
    ox = 2*((ix % s)+.5*(component==1))/s-1
    oy = 2*((iy % s)+.5*(component==0))/s-1
    u = np.vstack([g[:,anchor,topology],ox,oy,(np.asarray(frequencies)[fi]-7)/2,
                   mm,nn,pol,2*component-1]).astype(np.float32)
    return u,phase


def forward(model,u,cache=False):
    x = np.asarray(u,np.float32)
    z1 = model['W1']@x+model['b1'].reshape(-1,1)
    a1 = np.float32(.5)*z1*(np.float32(1)+erf(z1/np.float32(np.sqrt(2))))
    z2 = model['W2']@a1+model['b2'].reshape(-1,1)
    a2 = np.float32(.5)*z2*(np.float32(1)+erf(z2/np.float32(np.sqrt(2))))
    out = model['W3']@a2+model['b3'].reshape(-1,1)
    return (out,(x,z1,a1,z2,a2)) if cache else out


def predict(model,mask,frequencies,g=None):
    N = int(model['N'])
    if int(model['M'])!=1:
        raise ValueError('冻结网络仅支持 M=1、法向入射，不外推到斜入射')
    g = geometry(mask) if g is None else g
    if g.ndim==2:
        g = g[:,:,None]
    ids = active(mask,N)
    nr,nf,P = len(ids),len(frequencies),18
    row = np.tile(ids,P*nf)
    port = np.repeat(np.tile(np.arange(P),nf),nr)
    fi = np.repeat(np.arange(nf),P*nr)
    u,phase = features(g,row,np.zeros_like(row),fi,port,frequencies,N)
    raw = np.empty((2,len(row)),np.float32)
    for first in range(0,len(row),8192):
        raw[:,first:first+8192] = forward(model,u[:,first:first+8192])*np.float32(model['output_scale'])
    values = (raw[0]+1j*raw[1]).astype(complex)*phase
    return values.reshape((nr,P,nf),order='F')


def gelu_derivative(z):
    return np.float32(.5)*(1+erf(z/np.float32(np.sqrt(2))))+z*np.exp(-z*z/2)/np.float32(np.sqrt(2*np.pi))


def gradients(model,u,target,weight):
    """逐场相对 MSE 梯度。weight=(训练全局 RMS/当前标签场 RMS)²。"""
    pred,(x,z1,a1,z2,a2) = forward(model,u,True)
    diff = pred-target
    loss = np.sum(np.sum(diff*diff,axis=0)*weight)/u.shape[1]
    go = 2*diff*weight[None,:]/u.shape[1]
    g2 = (model['W3'].T@go)*gelu_derivative(z2)
    g1 = (model['W2'].T@g2)*gelu_derivative(z1)
    return float(loss),{'W3':go@a2.T,'b3':go.sum(axis=1),'W2':g2@a1.T,
                        'b2':g2.sum(axis=1),'W1':g1@x.T,'b1':g1.sum(axis=1)}


def train(labels_path,target,epochs=80,batches=128,batch=4096,seed=120026,validation_interval=5):
    """重新训练，写入新目录。随机数库不同，不承诺重得原始冻结权重。

    原论文结果对应发布的冻结权重。此函数使用相同结构、监督损失和优化设置，
    以独立验证集误差选择权重。默认 80 epochs，单精度训练，余弦学习率。
    """
    target = Path(target)
    target.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(seed)
    with Labels(labels_path) as d:
        if d.split is None:
            raise ValueError('训练文件必须包含固定 split')
        masks = d.masks[:]
        g = geometry(masks)
        training,validation = np.flatnonzero(d.split==1),np.flatnonzero(d.split==2)
        rows = [np.flatnonzero(v) for v in d.file['active'][:]]
        counts = np.array([len(v) for v in rows])
        # 只缓存有效金属行，避免同时保留全部空气零行和整份 complex128 副本。
        # 每个拓扑缓存布局为 [频率,局部有效电流行,端口]。
        X = []
        rms = np.empty((len(masks),len(d.freq),18))
        for t,ids in enumerate(rows):
            current = d.file['currents'][t][:,ids,:]
            X.append(current)
            rms[t] = np.sqrt(np.sum(abs(current.astype(complex))**2,axis=1)/counts[t])
        scale = np.float32(np.sqrt(np.mean(rms[training]**2)))
        model = {'N':np.asarray(d.N),'Q':np.asarray(d.Q),'M':np.asarray(d.M),'output_scale':np.asarray(scale)}
        widths = [71,128,128,2]
        for j in range(3):
            model[f'W{j+1}'] = (rng.standard_normal((widths[j+1],widths[j]))*np.sqrt(2/widths[j])).astype(np.float32)
            model[f'b{j+1}'] = np.zeros(widths[j+1],np.float32)
        names = ['W1','b1','W2','b2','W3','b3']
        mom,vel = ({k:np.zeros_like(model[k]) for k in names} for _ in range(2))
        best,best_epoch,iteration,history,start = np.inf,0,0,[],time.perf_counter()
        for epoch in range(1,epochs+1):
            lr = .0001+(.001-.0001)*(1+np.cos(np.pi*(epoch-1)/max(epochs-1,1)))/2
            accumulated = 0.
            for _ in range(batches):
                topo = rng.choice(training,batch)
                pos = np.floor(rng.random(batch)*counts[topo]).astype(int)
                row = np.array([rows[t][p] for t,p in zip(topo,pos)])
                fi,port = rng.integers(0,len(d.freq),batch),rng.integers(0,18,batch)
                u,phase = features(g,row,topo,fi,port,d.freq,d.N)
                label = np.array([X[t][f,p,j] for t,f,p,j in zip(topo,fi,pos,port)],complex)/phase
                target_value = np.vstack([label.real,label.imag]).astype(np.float32)/scale
                weight = (float(scale)/np.maximum(rms[topo,fi,port],1e-6))**2
                loss,grad = gradients(model,u,target_value,weight.astype(np.float32))
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
            if epoch==1 or epoch % validation_interval==0 or epoch==epochs:
                errors = []
                for t in validation:
                    mask,ids,truth,_ = d.case(t)
                    pred = predict(model,mask,d.freq,g[:,:,t:t+1])
                    errors.extend((np.sum(abs(pred-truth)**2,axis=0)/
                        np.maximum(np.sum(abs(truth)**2,axis=0),np.finfo(float).eps)).ravel())
                val = float(np.mean(errors))
                if val<best:
                    best,best_epoch = val,epoch
                    np.savez_compressed(target/'model.npz',**model,bestEpoch=np.asarray(epoch))
            history.append([epoch,accumulated/batches,val,lr,time.perf_counter()-start])
            print(f'epoch {epoch} train {history[-1][1]:.6g} validation {val:.6g}',flush=True)
        with (target/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_relative_squared_current','validation_relative_squared_current','learning_rate','elapsed_seconds'])
            writer.writerows(history)
        return {'best_epoch':best_epoch,'validation_relative_squared_current':best}
