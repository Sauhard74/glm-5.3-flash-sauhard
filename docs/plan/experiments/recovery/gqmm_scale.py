import mlx.core as mx
E,O,I,k=8,256,1024,4
w=mx.random.normal((E,O,I)).astype(mx.bfloat16)*0.02
wq,s,b=mx.quantize(w,group_size=64,bits=3); mx.eval(wq,s,b)
for T in (8,32,128):
  for sorted_ in (False,True):
    x=mx.random.normal((T*k,1,I)).astype(mx.bfloat16)
    idx=mx.random.randint(0,E,(T*k,)).astype(mx.uint32)
    if sorted_: idx=mx.sort(idx)
    def f(s,b): return mx.gather_qmm(x,wq,s,b,rhs_indices=idx,transpose=True,group_size=64,bits=3,sorted_indices=sorted_).astype(mx.float32).square().sum()
    mx.eval(x,idx); mx.clear_cache(); mx.reset_peak_memory(); base=mx.get_active_memory()
    g=mx.grad(f,argnums=(0,1))(s,b); mx.eval(g)
    pk=(mx.get_peak_memory()-base)/2**20
    print(f"rows={T*k} sorted={sorted_} peak_extra={pk:.1f} MiB  rows*O*I*4B={(T*k*O*I*4)/2**20:.1f} MiB  E*O*I*4B={E*O*I*4/2**20:.1f} MiB")
