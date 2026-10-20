"""Zero-start GMRES snapshots checked at every true-residual iterate."""
from . import polarizer_model as c
import numpy as np,time

def first_true_gmres(a,tolerance=.1,maximum_iterations=512,restart=64):
    start=time.perf_counter();B=a.p['B'][a.ids];Z=c.ph.block(a.plus,a.ids,a.ids)
    Y=np.zeros_like(B);residuals=[];iterations=[];history=[]
    for p,b in enumerate(B.T):
        bnorm=np.linalg.norm(b);x=np.zeros(len(b),complex);count=0;hit=False;port_history=[]
        if bnorm==0:residuals.append(0.);iterations.append(0);history.append([0.]);continue
        while count<maximum_iterations and not hit:
            r=b-Z@x;initial=np.linalg.norm(r)/bnorm
            if initial<=tolerance:hit=True;break
            rpre=c.ph.apply(a.plus,a.ids,r,inverse=True);beta=np.linalg.norm(rpre)
            m=min(restart,len(b),maximum_iterations-count);V=np.zeros((len(b),m+1),complex);H=np.zeros((m+1,m),complex);V[:,0]=rpre/beta
            rhs=np.zeros(m+1,complex);rhs[0]=beta
            for j in range(m):
                w=c.ph.apply(a.plus,a.ids,c.ph.apply(a.plus,a.ids,V[:,j]),inverse=True)
                # Two orthogonalization passes keep the small Arnoldi basis stable.
                for _ in range(2):
                    projection=V[:,:j+1].conj().T@w;H[:j+1,j]+=projection;w-=V[:,:j+1]@projection
                H[j+1,j]=np.linalg.norm(w)
                if abs(H[j+1,j])>1e-30:V[:,j+1]=w/H[j+1,j]
                coeff=np.linalg.lstsq(H[:j+2,:j+1],rhs[:j+2],rcond=None)[0]
                trial=x+V[:,:j+1]@coeff;actual=float(np.linalg.norm(b-Z@trial)/bnorm);count+=1;port_history.append(actual)
                if actual<=tolerance:
                    x=trial;hit=True;break
                if abs(H[j+1,j])<1e-14:
                    x=trial;break
            if not hit:x=trial
        actual=float(np.linalg.norm(b-Z@x)/bnorm)
        if not hit:raise ArithmeticError(f'GMRES port{p} failed, true residual{actual}, iterations{count}')
        Y[:,p]=x;residuals.append(actual);iterations.append(count);history.append(port_history)
    return Y,{'max_true_column_residual':max(residuals),'true_column_residuals':residuals,'first_hit_iterations':iterations,
              'true_residual_history_per_port':history,'seconds':time.perf_counter()-start,'initialization':'zero, non-learning',
              'operator':'FFT matrix-free Galerkin EFIE with full-period inverse left preconditioner','verification':'Unpreconditioned active denseZ, every Arnoldi iterate',
              'tolerance':tolerance,'maximum_iterations':maximum_iterations,'restart':restart}
