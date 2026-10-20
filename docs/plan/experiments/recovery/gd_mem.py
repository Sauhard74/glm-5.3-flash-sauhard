import time, sys, mlx.core as mx
from mlx_vlm.models.gated_delta import gated_delta_update
mx.set_memory_limit(12<<30)
Hk,Hv,D = int(sys.argv[1]), int(sys.argv[2]), 128
for T in (256,1024,2048):
    q=mx.random.normal((1,T,Hk,D)).astype(mx.bfloat16)*0.1; k=mx.random.normal((1,T,Hk,D)).astype(mx.bfloat16)*0.1
    v=mx.random.normal((1,T,Hv,D)).astype(mx.bfloat16); a=mx.random.normal((1,T,Hv)).astype(mx.bfloat16); b=mx.random.normal((1,T,Hv)).astype(mx.bfloat16)
    A=mx.zeros((Hv,)); dt=mx.zeros((Hv,))
    def f(q,k,v,a,b):
        y,_=gated_delta_update(q,k,v,a,b,A,dt,use_kernel=False); return y.astype(mx.float32).sum()
    mx.eval(q,k,v,a,b); mx.clear_cache(); mx.reset_peak_memory()
    base=mx.get_active_memory()
    t=time.time(); g=mx.grad(f,argnums=(0,1,2,3,4))(q,k,v,a,b); mx.eval(g); dtm=time.time()-t
    pk=(mx.get_peak_memory()-base)/2**20
    print(f"Hk={Hk} Hv={Hv} T={T} peak_extra={pk:.0f} MiB  per_token={pk/T:.2f} MiB  fwd+bwd {dtm*1e3/T:.2f} ms/token")
