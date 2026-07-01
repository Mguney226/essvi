"""GPU de-Americanization via a custom Metal kernel (MLX).

One GPU thread per contract runs the full control-variate Leisen-Reimer binomial + bisection
internally (incremental node spots, no per-level launch). fp32 - the control variate (exact Black
leg + small tree-estimated early-exercise premium) absorbs the precision loss, so sigma* matches the
CPU fp64 binomial to ~0.01 bp median at n=191. Saturates at ~3.5M contracts/s on the M3 Ultra GPU
(~4x the 28-core CPU at the de-Am, ~120x a single CPU thread).

Drop-in replacement for `deam.deam_iv_batch` (same signature). Enabled by `VSE_GPU_DEAM=1`.
Each thread is independent (no atomics/reductions) -> deterministic re-runs. `seeds` is ignored
(robust bracketed bisection; ITERS=22 is the fp32-justified count). For a fresh single-producer
backfill the fp32 surface is one of the equally-valid arb-free fits (see PERF_FINDINGS.md).
"""
from __future__ import annotations
import os
import numpy as np

_ITERS = int(os.environ.get("VSE_GPU_DEAM_ITERS", "22"))
_kernels: dict[int, object] = {}
_mx = None

def _build(n: int):
    global _mx
    import mlx.core as mx
    _mx = mx
    header = f"""
inline float myerf(float x){{ float s=x<0.0f?-1.0f:1.0f; x=fabs(x); float t=1.0f/(1.0f+0.3275911f*x);
 float y=1.0f-(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-x*x); return s*y; }}
inline float pp_inv(float z,int n){{ float c=z>=0.0f?1.0f:-1.0f; float den=(float)n+0.33333333f+0.1f/((float)n+1.0f);
 float t=z/den; float a=t*t*((float)n+0.16666667f); float p=0.5f+c*0.5f*sqrt(max(1.0f-exp(-a),0.0f)); return clamp(p,1e-7f,0.9999999f); }}
inline float black76(float F,float K,float T,float sig,float D,float isc){{ float sw=sig*sqrt(T); float d1=(log(F/K)+0.5f*sw*sw)/sw; float d2=d1-sw;
 float nd1=0.5f*(1.0f+myerf(d1*0.70710678f)); float nd2=0.5f*(1.0f+myerf(d2*0.70710678f));
 return isc>0.5f?D*(F*nd1-K*nd2):D*(K*(1.0f-nd2)-F*(1.0f-nd1)); }}
inline float amprice(float S,float K,float T,float r,float q,float sig,float isc){{
 int n={n}; float dt=T/(float)n; float sw=sig*sqrt(T); float d1=(log(S/K)+(r-q+0.5f*sig*sig)*T)/sw; float d2=d1-sw;
 float p=pp_inv(d2,n); float pbar=pp_inv(d1,n); float erqdt=exp((r-q)*dt); float u=erqdt*pbar/p; float d=(erqdt-p*u)/(1.0f-p); float disc=exp(-r*dt);
 float uod=u/d; float dn=pow(d,(float)n); float am[{n+1}]; float eu[{n+1}]; float st=S*dn;
 for(int j=0;j<=n;++j){{ float pay=isc>0.5f?max(st-K,0.0f):max(K-st,0.0f); am[j]=pay; eu[j]=pay; st*=uod; }}
 float di=dn/d;
 for(int i=n-1;i>=0;--i){{ float sn=S*di; for(int k=0;k<=i;++k){{ am[k]=disc*(p*am[k+1]+(1.0f-p)*am[k]); eu[k]=disc*(p*eu[k+1]+(1.0f-p)*eu[k]);
   float ex=isc>0.5f?max(sn-K,0.0f):max(K-sn,0.0f); if(ex>am[k]) am[k]=ex; sn*=uod; }} di/=d; }}
 float F=S*exp((r-q)*T); float D=exp(-r*T); return black76(F,K,T,sig,D,isc)+(am[0]-eu[0]); }}
"""
    body = f"""
 uint e=thread_position_in_grid.x; if(e>=(uint)price_shape[0]) return;
 float prc=price[e]; float K=Kk[e]; float T=Tt[e]; float r=rr[e]; float q=qq[e]; float isc=iscf[e]; float Sv=Sp[0];
 float intr=isc>0.5f?max(Sv-K,0.0f):max(K-Sv,0.0f); if(isnan(prc)||prc<intr-1e-9f){{ sig[e]=NAN; return; }}
 float lo=0.01f,hi=5.0f; float plo=amprice(Sv,K,T,r,q,lo,isc)-prc; float phi=amprice(Sv,K,T,r,q,hi,isc)-prc;
 if(plo>0.0f||phi<0.0f||isnan(plo)||isnan(phi)){{ sig[e]=NAN; return; }}
 for(int it=0;it<{_ITERS};++it){{ float mid=0.5f*(lo+hi); float fm=amprice(Sv,K,T,r,q,mid,isc)-prc; if(fm>0.0f) hi=mid; else lo=mid; }}
 sig[e]=0.5f*(lo+hi);
"""
    return mx.fast.metal_kernel(name=f"deam_cv_{n}", input_names=["price","Sp","Kk","Tt","rr","qq","iscf"],
                                output_names=["sig"], header=header, source=body)

def deam_iv_batch_gpu(prices, S, Ks, Ts, rs, qs, is_calls, seeds, n):
    """GPU de-Am: invert American prices -> Black-equivalent sigma* on the Metal GPU. Same contract as
    deam.deam_iv_batch (NaN on sub-intrinsic / no-bracket). fp32; returns float64 ndarray."""
    M = int(np.asarray(prices).shape[0])
    if M == 0:
        return np.empty(0)
    k = _kernels.get(int(n))
    if k is None:
        k = _kernels[int(n)] = _build(int(n))
    mx = _mx
    def A(x): return mx.array(np.ascontiguousarray(x, dtype=np.float32))
    out = k(inputs=[A(prices), A([float(S)]), A(Ks), A(Ts), A(rs), A(qs), A(np.asarray(is_calls, dtype=np.float32))],
            grid=(M, 1, 1), threadgroup=(min(256, M), 1, 1),
            output_shapes=[(M,)], output_dtypes=[mx.float32])[0]
    return np.array(out, dtype=np.float64)
