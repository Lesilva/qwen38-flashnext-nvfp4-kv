import json,re,collections,sys
t=json.load(open(sys.argv[1]));ev=t["traceEvents"];N=int(sys.argv[3])
kern=sorted([e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")],key=lambda e:e["ts"])
rt={(e.get("args") or {}).get("correlation"):e for e in ev if e.get("ph")=="X" and e.get("cat") in ("cuda_runtime","cuda_driver")}
ann=sorted([e for e in ev if e.get("ph")=="X" and e.get("cat")=="user_annotation" and e["name"] in("SP:verify","SP:draft","SP:extend")],key=lambda e:e["ts"])
b=collections.OrderedDict((k,[0,0.0]) for k in ("<1.5","1.5-3","3-6","6-12","12-25",">25"))
for e in kern:
    d=e["dur"];k="<1.5" if d<1.5 else "1.5-3" if d<3 else "3-6" if d<6 else "6-12" if d<12 else "12-25" if d<25 else ">25"
    b[k][0]+=1;b[k][1]+=d
print("HIST per step:",{k:(round(v[0]/N,1),round(v[1]/N,1)) for k,v in b.items()})
print("streams:",collections.Counter((e.get("args") or {}).get("stream") for e in kern).most_common(6))
def seq(a):
    return [e for e in kern if (lambda r:r is not None and a["ts"]<=r["ts"]<=a["ts"]+a["dur"])(rt.get((e.get("args") or {}).get("correlation")))]
with open(sys.argv[2],"w") as f:
    for nm in ("SP:draft","SP:verify","SP:extend"):
        a=[x for x in ann if x["name"]==nm][N//2];s=seq(a)
        f.write("## %s n=%d sum=%.1f span=%.1f\n"%(nm,len(s),sum(e["dur"] for e in s),s[-1]["ts"]+s[-1]["dur"]-s[0]["ts"]))
        prev=None
        for e in s:
            gap=(e["ts"]-prev) if prev is not None else 0;a_=e.get("args") or {}
            f.write("%9.1f %7.2f gap%8.2f st%s g%s b%s %s\n"%(e["ts"]-s[0]["ts"],e["dur"],gap,a_.get("stream"),a_.get("grid"),a_.get("block"),re.sub(r"\s+"," ",e["name"])[:120]))
            prev=max(prev or 0,e["ts"]+e["dur"])
print("ok")
