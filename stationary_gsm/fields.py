"""从真实 EFIE 电流重建表面电流和透射侧近场，不用 GSM 曲线生成场图。"""
import numpy as np
from .physics import PERIOD,ETA0,C0,outgoing_kz


def series(cx,cy,coordinate,z,fGHz=7.,Q=96):
    """法向 x 极化 1 V/m 入射。屋顶基 Fourier 积分采用解析 sinc 因子。"""
    N,p = len(cx),PERIOD
    k,d = 2*np.pi*fGHz*1e9/C0,p/N
    orders = np.arange(-Q,Q+1)
    kt,centres = 2*np.pi*orders/p,np.arange(N)*d
    E,Ec = np.exp(-1j*np.outer(kt,centres)),np.exp(-1j*np.outer(kt,centres+.5*d))
    s = np.sinc(kt*d/(2*np.pi))
    Jx = np.outer(s*s,s)*(E@cx@Ec.T)/(N*N)
    Jy = np.outer(s,s*s)*(Ec@cy@E.T)/(N*N)
    kx,ky = np.meshgrid(kt,kt,indexing='ij')
    kz = outgoing_kz(k,kx,ky)
    Axx,Ayy,Axy = (k*k-kx*kx)/(2*k*kz),(k*k-ky*ky)/(2*k*kz),-kx*ky/(2*k*kz)
    Ex,Ey = -(Axx*Jx+Axy*Jy),-(Axy*Jx+Ayy*Jy)
    Ez = -(kx*Ex+ky*Ey)/kz
    phase,F = np.exp(1j*kz*z),np.exp(1j*np.outer(coordinate,kt))
    ex,ey,ez = F@(Ex*phase)@F.T+np.exp(1j*k*z),F@(Ey*phase)@F.T,F@(Ez*phase)@F.T
    return ex,ey,ez,np.array([Ex[Q,Q],Ey[Q,Q]])


def reconstruct(currents,ids,N,fGHz=7.,z=.002,points=81,Q=96):
    full = np.zeros(2*N*N,complex)
    full[ids] = currents
    cx,cy = full[:N*N].reshape((N,N),order='F'),full[N*N:].reshape((N,N),order='F')
    coordinate = np.linspace(0,PERIOD,points)
    xx,yy = np.meshgrid(coordinate,coordinate,indexing='ij')
    tx,ty = (xx/(PERIOD/N)) % N,(yy/(PERIOD/N)) % N
    ix,iy = np.floor(tx).astype(int),np.floor(ty).astype(int)
    fx,fy = tx-ix,ty-iy
    jx = ((1-fx)*cx[ix,iy]+fx*cx[(ix+1)%N,iy])/ETA0
    jy = ((1-fy)*cy[ix,iy]+fy*cy[ix,(iy+1)%N])/ETA0
    ex,ey,ez,zero = series(cx,cy,coordinate,z,fGHz,Q)
    return {'x_mm':coordinate*1000,'y_mm':coordinate*1000,'z_mm':z*1000,
            'jx':jx,'jy':jy,'jmag':np.sqrt(abs(jx)**2+abs(jy)**2),'ex':ex,'ey':ey,'ez':ez,
            'emag':np.sqrt(abs(ex)**2+abs(ey)**2+abs(ez)**2),'zero_reflection':zero}
