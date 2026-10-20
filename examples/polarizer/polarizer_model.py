"""Dual-input five-layer linear-to-circular PEC polarizer reference model.

The patterns and square lattice rotate together by45degrees. All sheets have
the same transverse center. The20mm core period is mapped exactly to the
13.333333mm physical period for the free-space zero-thickness PEC model.
"""
import os,time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
from scipy.linalg import solve
from stationary_gsm import physics as ph,extraction
SCALE=2/3
PERIOD_M=.020*SCALE
ROT=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
if not np.isclose(ph.PERIOD,.020):raise ValueError("Expected20mm reference core period")




def translated_ports(f,N,M,shift):
    p=ph.ports(f*SCALE,N,M,0,0)
    q=2*np.pi*p['orders']/ph.PERIOD
    phase=np.repeat(np.exp(1j*(q@shift)),2)
    p['B']=p['B']*phase[None,:];p['C']=phase.conj()[:,None]*p['C']
    p['origin_shift_ref_m']=np.array(shift)
    return p

def shift_gsm(S,pt,shift):
    phase=np.repeat(np.exp(1j*(2*np.pi*pt['orders']/ph.PERIOD@shift)),2)
    d=np.tile(phase,2)
    return d.conj()[:,None]*S*d[None,:]

def rotated_fundamental(S):
    W=np.kron(np.eye(2),ROT)
    return W@S@W.T

def rotated_gsm(S):
    W=np.kron(np.eye(len(S)//2),ROT)
    return W@S@W.T

def metrics(S):
    out={}
    for pol,p in [('x',0),('y',1)]:
        e=S[2:,p];r=S[:2,p];T=float(np.vdot(e,e).real);R=float(np.vdot(r,r).real)
        axes=np.linalg.svd(np.c_[e.real,-e.imag],compute_uv=False)
        ar=float(20*np.log10(max(1.,axes[0]/max(axes[1],1e-30))))
        s3=float(2*np.imag(e[0]*e[1].conj()))
        out[pol]={'T':T,'R':R,'AR_dB':ar,'IL_dB':float(-10*np.log10(max(T,1e-30))),
                  'circularity':s3/max(T,1e-30),'phase_Ey_minus_Ex_deg':float(np.angle(e[1]*e[0].conj(),deg=True)),
                  'P_x_plus_i_y':float(abs((e[0]-1j*e[1])/np.sqrt(2))**2),
                  'P_x_minus_i_y':float(abs((e[0]+1j*e[1])/np.sqrt(2))**2),'power_balance':T+R-1.}
    out['T_min']=min(out['x']['T'],out['y']['T']);out['AR_max_dB']=max(out['x']['AR_dB'],out['y']['AR_dB'])
    out['opposite_circularity']=bool(out['x']['circularity']*out['y']['circularity']<0)
    return out

def join(A,B,D):
    P=len(D);a,b,c,d=A[:P,:P],A[:P,P:]*D[None,:],D[:,None]*A[P:,:P],D[:,None]*A[P:,P:]*D[None,:]
    e,ff,g,h=B[:P,:P],B[:P,P:],B[P:,:P],B[P:,P:]
    x=solve(np.eye(P)-e@d,np.c_[e@c,ff],check_finite=False);x1,x2=x[:,:P],x[:,P:]
    return np.block([[a+b@x1,b@x2],[g@(c+d@x1),h+g@d@x2]])

def cascade(lib,seq,gaps,f,N=24,M=1):
    pt=ph.ports(f*SCALE,N,M,0,0);s=lib[seq[0]]
    for j in range(1,5):s=join(s,lib[seq[j]],np.exp(1j*pt['kz']*gaps[j-1]/SCALE))
    P=len(pt['kz']);ix=np.r_[pt['normal'],P+pt['normal']]
    return s,rotated_fundamental(s[np.ix_(ix,ix)])

def shifted_kernel(f,N,Q,gap,shift):
    """Full Green sum with actual lateral source/test-origin difference before aliasing."""
    if np.linalg.norm(shift)<1e-15:return ph.kernel(f*SCALE,N,Q,0,0,gap/SCALE)
    k0=2*np.pi*f*SCALE*1e9/ph.C0;d=ph.PERIOD/N;W=np.zeros((4,N*N),complex);mm=np.arange(-Q,Q+1)
    for first in range(-Q,Q+1,32):
        m,n=np.meshgrid(mm,np.arange(first,min(first+32,Q+1)),indexing='ij');m,n=m.ravel(order='F'),n.ravel(order='F')
        kx,ky=2*np.pi*m/ph.PERIOD,2*np.pi*n/ph.PERIOD;kz=ph.outgoing_kz(k0,kx,ky)
        sx=np.sinc(kx*d/(2*np.pi))**2*np.sinc(ky*d/(2*np.pi));sy=np.sinc(kx*d/(2*np.pi))*np.sinc(ky*d/(2*np.pi))**2
        prop=np.exp(1j*kz*gap/SCALE+1j*(kx*shift[0]+ky*shift[1]))
        xx=(k0*k0-kx*kx)/(2*k0*kz)*prop;yy=(k0*k0-ky*ky)/(2*k0*kz)*prop;xy=-kx*ky/(2*k0*kz)*prop;alias=m%N+N*(n%N)
        for j,value in enumerate([xx*sx*sx,xy*sx*sy*np.exp(1j*d*(-kx+ky)/2),xy*sx*sy*np.exp(1j*d*(kx-ky)/2),yy*sy*sy]):
            W[j]+=np.bincount(alias,weights=value.real,minlength=N*N)+1j*np.bincount(alias,weights=value.imag,minlength=N*N)
    W=np.stack([v.reshape((N,N),order='F')for v in W])
    return {'W':W,'K':np.fft.ifft2(W,axes=(1,2)),'phase':np.ones((N,N),complex),'N':N,'Q':Q,'beta':np.zeros(2),'k0':k0,'gap':gap/SCALE}

def coupled(masks,shifts,gaps,f,N,Q=None):
    start=time.perf_counter();Q=8*N if Q is None else Q;z=np.r_[0,np.cumsum(gaps)];h=z[-1]
    pt=ph.ports(f*SCALE,N,0,0,0);g=ph.kernel(f*SCALE,N,Q,0,0);ids=[ph.active(m,N)for m in masks];off=np.r_[0,np.cumsum([len(i)for i in ids])]
    Z=np.zeros((off[-1],off[-1]),complex);B=np.zeros((off[-1],4),complex);C=np.zeros((4,off[-1]),complex);k0=2*np.pi*f*1e9/ph.C0
    for j in range(5):
        sl=slice(off[j],off[j+1]);Z[sl,sl]=ph.block(g,ids[j],ids[j]);bl,cl=pt['B'][ids[j]],pt['C'][:,ids[j]]
        front,rear=np.exp(1j*k0*z[j]),np.exp(1j*k0*(h-z[j]));B[sl,:2]=bl*front;B[sl,2:]=bl*rear;C[:2,sl]=cl*front;C[2:,sl]=cl*rear
    for j in range(5):
        for l in range(j+1,5):
            kg=shifted_kernel(f,N,Q,float(z[l]-z[j]),shifts[j]-shifts[l]);val=ph.block(kg,ids[j],ids[l]);Z[off[j]:off[j+1],off[l]:off[l+1]]=val;Z[off[l]:off[l+1],off[j]:off[j+1]]=val.T
    # Rotate the actual source and readout columns before solving: both labx
    # and laby inputs are explicit RHSs, not inferred from a copied CP curve.
    W=np.kron(np.eye(2),ROT);B=B@W.T;C=W@C
    assembled=time.perf_counter();X=solve(Z,B,check_finite=False);solved=time.perf_counter();D=np.exp(1j*k0*h)*np.eye(2);O=np.zeros((2,2));S=np.block([[O,D],[D,O]])+C@X
    Suv=W.T@S@W;res=np.linalg.norm(Z@X-B,axis=0)/np.linalg.norm(B,axis=0)
    return S,{'N':N,'Q':Q,'unknowns':int(off[-1]),'max_true_column_residual':float(max(res)),'power_defect':float(np.linalg.norm(S.conj().T@S-np.eye(4),2)),
              'principal_crosspol_max':float(np.max(abs(Suv[np.ix_([0,2],[1,3])]))),'assembly_seconds':assembled-start,'solve_seconds':solved-assembled,'total_seconds':time.perf_counter()-start},Suv
