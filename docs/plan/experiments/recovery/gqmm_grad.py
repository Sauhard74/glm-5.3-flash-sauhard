import mlx.core as mx, time
mx.set_memory_limit(2<<30)
E,O,I,T,k=64,640,2560,512,10   # Qwen-like expert shape, 64 experts
w=mx.random.normal((E,O,I)).astype(mx.bfloat16)*0.02
wq,s,b=mx.quantize(w,group_size=64,bits=3)
x=mx.random.normal((T,1,I)).astype(mx.bfloat16)
idx=mx.random.randint(0,E,(T,k)).astype(mx.uint32)
def f(s,b,x):
    y=mx.gather_qmm(x[:,None],wq,s,b,rhs_indices=idx[:,None] if False else idx,transpose=True,group_size=64,bits=3)
    return y.astype(mx.float32).square().sum()
mx.eval(wq,s,b,x); del w; mx.clear_cache(); mx.reset_peak_memory(); base=mx.get_active_memory()
try:
    t=time.time(); g=mx.grad(f,argnums=(0,1,2))(s,b,x); mx.eval(g); print("grad ok", [a.shape for a in g], f"{time.time()-t:.3f}s", f"peak_extra {(mx.get_peak_memory()-base)/2**20:.0f} MiB", "dense W bf16 would be", E*O*I*2/2**20, "MiB")
except Exception as e: print("ERR", type(e).__name__, str(e)[:300])
