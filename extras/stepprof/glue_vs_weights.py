import re,sys,collections
W=re.compile(r'w8a16|GroupProblemShape|cutlass_80_wmma')
def short(n):
    n=re.sub(r'^void ','',n)
    for k in ('index_put','index_kernel_impl','indexSelect','direct_copy','FillFunctor','where','CatArray','Memcpy','memcpy32','scatter_gather'):
        if k in n:return 'torch:'+k
    if n.startswith('at::native'):return 'torch:elementwise/other'
    return re.split(r'[<(]',n)[0][:60]
for fn in sys.argv[1:]:
    ph=None;D=collections.defaultdict(list)
    for l in open(fn):
        if l.startswith('##'):ph=l.split()[1];continue
        p=l.split(None,6);D[ph].append((float(p[0]),float(p[1]),p[6].strip()))
    for ph,ks in D.items():
        ev=[]
        for t,d,n in ks:ev+= [(t,0,n),(t+d,1,n)]
        ev.sort(key=lambda x:(x[0],-x[1]));act=collections.Counter();last=ev[0][0]
        wt=gl=idle=0.0;gby=collections.Counter()
        for t,k,n in ev:
            dt=t-last
            if dt>0:
                names=[a for a in act if act[a]>0]
                if any(W.search(a) for a in names):wt+=dt
                elif names:
                    gl+=dt
                    for a in names:gby[short(a)]+=dt/len(names)
                else:idle+=dt
            last=t
            if k==0:act[n]+=1
            else:act[n]-=1
        span=ev[-1][0]-ev[0][0];nW=sum(1 for _,_,n in ks if W.search(n))
        print(f"\n== {fn.split('/')[-1]} {ph}: span {span:.0f}us kernels {len(ks)} (weight {nW}) | weight-kernel wall {wt:.0f} ({wt/span*100:.0f}%) glue-only wall {gl:.0f} ({gl/span*100:.0f}%) idle {idle:.0f} ({idle/span*100:.0f}%)")
        if ph=='SP:verify':
            for a,v in gby.most_common(28):print(f"   {v:7.1f}  {a}")
