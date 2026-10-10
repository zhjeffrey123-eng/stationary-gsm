"""多层金属面的全耦合 EFIE 和包含倏逝阶的 GSM 连接。"""
import numpy as np
from scipy.linalg import solve,block_diag
from .physics import active,kernel,block,ports,assemble_gsm


def star(A,B):
    """Redheffer 星积，A 位于 B 前方，用反馈矩阵求解内部模态幅度。"""
    P = len(A)//2
    A11,A12,A21,A22 = A[:P,:P],A[:P,P:],A[P:,:P],A[P:,P:]
    B11,B12,B21,B22 = B[:P,:P],B[:P,P:],B[P:,:P],B[P:,P:]
    F = np.eye(P)-B11@A22
    # 用 1-范数条件数诊断反馈敏感性。不是 MATLAB rcond 的估计器。
    rcond = float(1/abs(np.linalg.cond(F,1)))
    if rcond<=np.finfo(float).eps:
        raise ArithmeticError('级联反馈矩阵接近奇异')
    X = solve(F,np.c_[B11@A21,B12])
    X1,X2 = X[:,:P],X[:,P:]
    return np.block([[A11+A12@X1,A12@X2],
                     [B21@(A21+A22@X1),B22+B21@A22@X2]]),rcond


def cascade(sheets,kz,gaps):
    if not sheets or len(gaps)!=len(sheets)-1 or np.any(np.asarray(gaps)<0):
        raise ValueError('层数与间隙不一致')
    P = len(kz)
    S,minimum = sheets[0],1.
    for l in range(1,len(sheets)):
        D = np.diag(np.exp(1j*np.asarray(kz)*gaps[l-1]))
        O = np.zeros((P,P))
        S,c1 = star(S,np.block([[O,D],[D,O]]))
        S,c2 = star(S,sheets[l])
        minimum = min(minimum,c1,c2)
    return S,minimum


def indices(orders,M):
    k = np.flatnonzero(np.all(abs(orders)<=M,axis=1))
    return np.c_[2*k,2*k+1].ravel()


def coupled(masks,fGHz,N,Q,M,theta,phi,gaps):
    """直接组装层间 Green 块，独立于 GSM 连接，但共享基函数和 Green 模型。"""
    masks = np.asarray(masks)
    if masks.ndim!=3 or masks.shape[1:]!=(8,8) or len(gaps)!=len(masks)-1:
        raise ValueError('masks 布局须为 [层,y,x]')
    L = len(masks)
    z = np.r_[0,np.cumsum(gaps)]
    height = z[-1]
    pt = ports(fGHz,N,M,theta,phi)
    P = len(pt['kz'])
    self_kernel = kernel(fGHz,N,Q,theta,phi)
    ids = [active(mask,N) for mask in masks]
    offsets = np.r_[0,np.cumsum([len(v) for v in ids])]
    nu = offsets[-1]
    Z,B,C = np.zeros((nu,nu),complex),np.zeros((nu,2*P),complex),np.zeros((2*P,nu),complex)
    layerS = []
    for l in range(L):
        rr = slice(offsets[l],offsets[l+1])
        Zll = block(self_kernel,ids[l],ids[l])
        Z[rr,rr] = Zll
        Bl,Cl = pt['B'][ids[l]],pt['C'][:,ids[l]]
        Xl = solve(Zll,Bl) if len(ids[l]) else np.zeros_like(Bl)
        layerS.append(assemble_gsm(Cl@Xl))
        front,rear = np.exp(1j*pt['kz']*z[l]),np.exp(1j*pt['kz']*(height-z[l]))
        B[rr] = np.c_[Bl*front[None,:],Bl*rear[None,:]]
        C[:,rr] = np.r_[front[:,None]*Cl,rear[:,None]*Cl]
    for distance in np.unique(abs(z[:,None]-z)):
        if distance==0:
            continue
        g = kernel(fGHz,N,Q,theta,phi,float(distance))
        for l in range(L):
            for r in range(L):
                if l!=r and abs(abs(z[l]-z[r])-distance)<1e-14:
                    Z[offsets[l]:offsets[l+1],offsets[r]:offsets[r+1]] = block(g,ids[l],ids[r])
    X = solve(Z,B) if nu else np.zeros_like(B)
    D,O = np.diag(np.exp(1j*pt['kz']*height)),np.zeros((P,P))
    S = np.block([[O,D],[D,O]])+C@X
    q,G = pt['normal'],pt['G']
    Gin = G[np.ix_(q,q)]
    defects = []
    for R,T in ((S[:P,q],S[P:,q]),(S[P:,P+q],S[:P,P+q])):
        defects.append(np.linalg.norm(R.conj().T@G@R+T.conj().T@G@T-Gin,2)/np.linalg.norm(Gin,2))
    return S,layerS,{'ports':pt,'ids':ids,'z':z,'currents':X,
        'relative_residual':float(np.linalg.norm(Z@X-B)/max(np.linalg.norm(B),np.finfo(float).eps)),
        'power_defect':float(max(defects))}


def gap_amplitudes(sheets,kz,gaps,incident):
    """求各间隙两端的实际前向和后向幅度，任意 z 处另乘传播因子。"""
    L,P = len(sheets),len(kz)
    S = block_diag(*sheets)
    T = np.zeros_like(S)
    for l,gap in enumerate(gaps):
        right,left = slice(2*P*l+P,2*P*(l+1)),slice(2*P*(l+1),2*P*(l+1)+P)
        D = np.diag(np.exp(1j*np.asarray(kz)*gap))
        T[left,right],T[right,left] = D,D
    u = np.zeros(2*P*L,complex)
    u[:P] = incident
    b = solve(np.eye(len(u))-S@T,S@u)
    return [b[2*P*l+P:2*P*(l+1)] for l in range(L-1)], [
        b[2*P*(l+1):2*P*(l+1)+P] for l in range(L-1)]


def reflectance(S,pt,polarization='TM'):
    """包含全部传播阶的反射、透射功率，输入 TE/TM 采用实际切向幅度。"""
    phi,theta = np.deg2rad(pt['phi']),np.deg2rad(pt['theta'])
    v = np.array([-np.sin(phi),np.cos(phi)]) if polarization=='TE' else np.cos(theta)*np.array([np.cos(phi),np.sin(phi)])
    P,q,G = len(pt['G']),pt['normal'],pt['G']
    incident = np.zeros(P,complex)
    incident[q] = v
    R,T = S[:P,:P]@incident,S[P:,:P]@incident
    Pin = np.real(v.conj()@G[np.ix_(q,q)]@v)
    return float(np.real(R.conj()@G@R)/Pin),float(np.real(T.conj()@G@T)/Pin)
