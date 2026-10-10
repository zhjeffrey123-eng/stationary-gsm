"""谱域 Galerkin EFIE，零厚度 PEC，自由空间，时间因子 exp(-iωt)。

掩模按 [y,x] 储存。电流按 [x方向N²个基，y方向N²个基] 排列，
每一块内 x 最快变化。Python 使用从零开始的索引，读取原数据时须转换。
电流系数为 η₀J，端口系数为切向 Ex/Ey 电场，不能直接当作功率坐标。
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve as linear_solve
from scipy.sparse.linalg import LinearOperator, gmres

PERIOD = .020
C0 = 299792458.
ETA0 = 376.730313668


def active(mask, N):
    """只保留整个屋顶基支撑均在金属内的未知量，周期边缘也检查邻居。"""
    mask = np.asarray(mask)
    if mask.shape != (8, 8) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError('mask 必须为 8×8 二值矩阵')
    if N < 8 or N % 8:
        raise ValueError('N 必须是 8 的正整数倍')
    metal = np.repeat(np.repeat(mask.astype(bool), N // 8, 0), N // 8, 1).T
    ax = metal & np.roll(metal, 1, axis=0)
    ay = metal & np.roll(metal, 1, axis=1)
    return np.r_[np.flatnonzero(ax.ravel(order='F')),
                 N*N + np.flatnonzero(ay.ravel(order='F'))]


def outgoing_kz(k0, kx, ky):
    """出射分支 Im(kz)≥0。截止点不作人工正则化，应另行处理其极限。"""
    kz = np.sqrt((k0*k0 - kx*kx - ky*ky).astype(complex))
    if np.any(np.abs(kz) <= 1e-9*k0):
        raise ValueError('Wood/Rayleigh 截止点，kz 接近零')
    return kz


def kernel(fGHz, N, Q, theta=0., phi=30., gap=0.):
    """分块求有限 Green 谱和，并按整数模 N 折叠为四个 FFT 核。

    Q 控制 Green 谱，M 控制 GSM 端口，两者是不同的截断。
    gap>0 时加入层间传播 exp(i*kz*gap)，不会丢弃倏逝阶。
    """
    if fGHz <= 0 or N < 8 or N % 8 or Q < 0 or gap < 0 or not 0 <= theta < 90:
        raise ValueError('物理参数不合法')
    k0 = 2*np.pi*fGHz*1e9/C0
    d = PERIOD/N
    beta = k0*np.sin(np.deg2rad(theta))*np.array([
        np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))])
    W = np.zeros((4, N*N), complex)
    mm = np.arange(-Q, Q+1)
    for first in range(-Q, Q+1, 32):
        m, n = np.meshgrid(mm, np.arange(first, min(first+32, Q+1)), indexing='ij')
        m, n = m.ravel(order='F'), n.ravel(order='F')
        kx, ky = beta[0]+2*np.pi*m/PERIOD, beta[1]+2*np.pi*n/PERIOD
        kz = outgoing_kz(k0, kx, ky)
        # numpy.sinc(x)=sin(πx)/(πx)，因此此处输入除以 π。
        sx = np.sinc(kx*d/(2*np.pi))**2*np.sinc(ky*d/(2*np.pi))
        sy = np.sinc(kx*d/(2*np.pi))*np.sinc(ky*d/(2*np.pi))**2
        prop = np.exp(1j*kz*gap)
        xx = (k0*k0-kx*kx)/(2*k0*kz)*prop
        yy = (k0*k0-ky*ky)/(2*k0*kz)*prop
        xy = -kx*ky/(2*k0*kz)*prop
        alias = m % N + N*(n % N)
        values = [xx*sx*sx, xy*sx*sy*np.exp(1j*d*(-kx+ky)/2),
                  xy*sx*sy*np.exp(1j*d*(kx-ky)/2), yy*sy*sy]
        for j, value in enumerate(values):
            # bincount 只处理实数，分别累加实部和虚部，保持原谱积分定义。
            W[j] += np.bincount(alias, weights=value.real, minlength=N*N)
            W[j] += 1j*np.bincount(alias, weights=value.imag, minlength=N*N)
    W = np.stack([v.reshape((N,N), order='F') for v in W])
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    phase = np.exp(1j*d*(beta[0]*i+beta[1]*j))
    return {'W': W, 'K': np.fft.ifft2(W, axes=(1,2)), 'phase': phase,
            'N': N, 'Q': Q, 'beta': beta, 'k0': k0, 'gap': gap}


def block(g, row_ids, col_ids):
    """只组装有效金属基之间的阻抗块，避免构造空气区域的未知量。"""
    N = g['N']
    rr, cc = np.asarray(row_ids, int), np.asarray(col_ids, int)
    ri, ci = rr % (N*N), cc % (N*N)
    rx, ry, cx, cy = ri % N, ri // N, ci % N, ci // N
    rt, ct = rr // (N*N), cc // (N*N)
    Z = np.zeros((len(rr), len(cc)), complex)
    for t in range(2):
        rows = np.flatnonzero(rt == t)
        for first in range(0, len(rows), 64):
            rows_chunk = rows[first:first+64]
            for u in range(2):
                cols = np.flatnonzero(ct == u)
                di, dj = rx[rows_chunk,None]-cx[cols], ry[rows_chunk,None]-cy[cols]
                # 相位必须使用真实整数差，不能使用取模后的坐标差。
                phase = np.exp(1j*(PERIOD/N)*(g['beta'][0]*di+g['beta'][1]*dj))
                Z[np.ix_(rows_chunk,cols)] = g['K'][2*t+u][di % N,dj % N]*phase
    return Z


def apply(g, ids, x, inverse=False, adjoint=False):
    """FFT 矩阵自由乘法。inverse 是全周期逆预条件，不是图案的精确逆。"""
    N = g['N']
    x = np.asarray(x, complex)
    vector = x.ndim == 1
    if vector:
        x = x[:,None]
    full = np.zeros((2*N*N,x.shape[1]), complex)
    full[ids] = x
    phase = g['phase'][:,:,None]
    ux = full[:N*N].reshape((N,N,-1), order='F')*phase.conj()
    uy = full[N*N:].reshape((N,N,-1), order='F')*phase.conj()
    Fx, Fy = np.fft.fft2(ux, axes=(0,1)), np.fft.fft2(uy, axes=(0,1))
    W = g['W']
    if inverse:
        det = W[0]*W[3]-W[1]*W[2]
        if np.any(np.abs(det) <= np.finfo(float).eps*np.max(np.abs(W[0]*W[3])+np.abs(W[1]*W[2]))):
            raise ArithmeticError('背景谱逆奇异')
        W = np.stack([W[3]/det,-W[1]/det,-W[2]/det,W[0]/det])
    if adjoint:
        W = W[[0,2,1,3]].conj()
    vx = np.fft.ifft2(W[0,:,:,None]*Fx+W[1,:,:,None]*Fy, axes=(0,1))*phase
    vy = np.fft.ifft2(W[2,:,:,None]*Fx+W[3,:,:,None]*Fy, axes=(0,1))*phase
    result = np.vstack([vx.reshape((N*N,-1), order='F'),
                        vy.reshape((N*N,-1), order='F')])[ids]
    return result[:,0] if vector else result


def ports(fGHz, N, M=1, theta=0., phi=30.):
    """每个 (m,n) 阶连续排列 Ex/Ey，功率采用传播阶的完整导纳 Gram。"""
    k0, d = 2*np.pi*fGHz*1e9/C0, PERIOD/N
    beta = k0*np.sin(np.deg2rad(theta))*np.array([np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))])
    i,j = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    i,j = i.ravel(order='F'),j.ravel(order='F')
    cx,cy = np.c_[i*d,(j+.5)*d], np.c_[(i+.5)*d,j*d]
    m,n = np.meshgrid(np.arange(-M,M+1),np.arange(-M,M+1),indexing='ij')
    orders = np.c_[m.ravel(order='F'),n.ravel(order='F')]
    qxy = beta + 2*np.pi*orders/PERIOD
    kz = outgoing_kz(k0,qxy[:,0],qxy[:,1])
    P = 2*len(orders)
    B,A,G,J = np.zeros((2*N*N,P),complex),np.zeros((P,P),complex),np.zeros((P,P)),np.zeros((P,P))
    for l,q in enumerate(qxy):
        sl = slice(2*l,2*l+2)
        sx,sy = np.sinc(q*d/(2*np.pi))
        B[:N*N,2*l] = sx*sx*sy*np.exp(1j*(cx@q))
        B[N*N:,2*l+1] = sx*sy*sy*np.exp(1j*(cy@q))
        A[sl,sl] = (k0*k0*np.eye(2)-np.outer(q,q))/(2*k0*kz[l])
        if abs(kz[l].imag)<1e-10*k0 and kz[l].real>0:
            G[sl,sl] = ((kz[l]**2*np.eye(2)+np.outer(q,q))/(k0*kz[l])).real
        rev = np.flatnonzero(np.all(orders == -orders[l],axis=1))[0]
        J[sl,2*rev:2*rev+2] = np.eye(2)
    zero = np.flatnonzero(np.all(orders==0,axis=1))[0]
    return {'B':B,'A':A,'C':-A@B.conj().T/(N*N),'G':G,'J':J,
            'orders':orders,'kz':np.repeat(kz,2),'normal':np.array([2*zero,2*zero+1]),
            'beta':beta,'N':N,'M':M,'fGHz':fGHz,'theta':theta,'phi':phi,'cx':cx,'cy':cy}


@dataclass
class Sheet:
    """同一结构的正反横向入射波矢算子与端口，索引均从零开始。"""
    plus: dict
    minus: dict
    p: dict
    m: dict
    ids: np.ndarray


def sheet(mask,fGHz,N=24,Q=None,M=1,theta=0.,phi=30.):
    Q = 8*N if Q is None else int(Q)
    if Q < M:
        raise ValueError('Q 不能小于 M')
    kp = kernel(fGHz,N,Q,theta,phi)
    pp = ports(fGHz,N,M,theta,phi)
    # 反相位等于方位角加 180°，不是改为另一侧照射。
    km = kp if theta==0 else kernel(fGHz,N,Q,theta,phi+180)
    pm = pp if theta==0 else ports(fGHz,N,M,theta,phi+180)
    # Green 谱内若有保留端口范围以外的传播阶，能量检查会遗漏功率。
    m,n = np.meshgrid(np.arange(-Q,Q+1),np.arange(-Q,Q+1),indexing='ij')
    for pt in (pp,pm):
        kx,ky = pt['beta'][0]+2*np.pi*m/PERIOD,pt['beta'][1]+2*np.pi*n/PERIOD
        propagating = kx*kx+ky*ky < kp['k0']**2
        if np.any(propagating & ((abs(m)>M)|(abs(n)>M))):
            raise ValueError('M 遗漏传播 Floquet 阶')
    return Sheet(kp,km,pp,pm,active(mask,N))


def phase_solve(g,ids,B,method='dense',tol=1e-9,restart=80,max_cycles=80):
    """参考电流求解。最后总是独立检查未预条件的真实 EFIE 残差。"""
    if len(ids)==0:
        return np.zeros_like(B), {'relative_residual':0.,'iterations':[]}
    if method=='dense':
        X = linear_solve(block(g,ids,ids),B,assume_a='gen')
        iterations = []
    elif method=='fft':
        A = LinearOperator((len(ids),len(ids)),matvec=lambda x:apply(g,ids,x),dtype=complex)
        P = LinearOperator(A.shape,matvec=lambda x:apply(g,ids,x,inverse=True),dtype=complex)
        X = np.zeros_like(B)
        iterations = []
        for j,b in enumerate(B.T):
            if np.linalg.norm(b)==0:
                iterations.append(0)
                continue
            count = [0]
            def count_iteration(_):
                count[0] += 1
            x,status = gmres(A,b,M=P,rtol=min(1e-12,tol/100),atol=0,
                restart=min(restart,len(ids)),maxiter=max_cycles,callback=count_iteration,callback_type='pr_norm')
            residual = np.linalg.norm(A@x-b)/np.linalg.norm(b)
            if residual > tol:
                raise ArithmeticError(f'端口 {j} 真实残差 {residual:.3g}，求解状态 {status}')
            X[:,j] = x
            iterations.append(count[0])
    else:
        raise ValueError(method)
    R = apply(g,ids,X)-B
    columns = np.linalg.norm(R,axis=0)/np.maximum(np.linalg.norm(B,axis=0),np.finfo(float).tiny)
    if np.max(columns)>tol:
        raise ArithmeticError('参考解的真实 EFIE 残差未达标')
    return X,{'relative_residual':float(np.linalg.norm(R)/max(np.linalg.norm(B),np.finfo(float).eps)),
              'maximum_column_residual':float(np.max(columns)),'iterations':iterations}


def assemble_gsm(R):
    """共同参考面上自由空间零厚度 PEC 满足 T=I+R。"""
    T = np.eye(len(R))+R
    return np.block([[R,T],[T,R]])


def power_defect(S,pt):
    P = len(pt['G'])
    q,G = pt['normal'],pt['G']
    R,T = S[:P,q],S[P:,q]
    Gin = G[np.ix_(q,q)]
    return float(np.linalg.norm(R.conj().T@G@R+T.conj().T@G@T-Gin,2)/np.linalg.norm(Gin,2))


def solve(mask,fGHz,N=24,Q=None,M=1,theta=0.,phi=30.,method='dense',tol=1e-9):
    a = sheet(mask,fGHz,N,Q,M,theta,phi)
    Xp,dp = phase_solve(a.plus,a.ids,a.p['B'][a.ids],method,tol)
    Xm,dm = (Xp,dp) if theta==0 else phase_solve(a.minus,a.ids,a.m['B'][a.ids],method,tol)
    Sp,Sm = assemble_gsm(a.p['C'][:,a.ids]@Xp),assemble_gsm(a.m['C'][:,a.ids]@Xm)
    return Sp,Xp,Xm,{'plus':dp,'minus':dm,'power_defect':max(power_defect(Sp,a.p),power_defect(Sm,a.m))},a
