import json,pathlib,statistics,sys,collections,bisect,re
# Step-profile analysis (2026-11-14). Usage: analyze.py <prof dir> -> prints one JSON document.
D=pathlib.Path(sys.argv[1]);out={}
med=statistics.median
def phases(rows):
 r={}
 for bs in sorted({x[0] for x in rows}):
  v=[x for x in rows if x[0]==bs]
  g=lambda f:round(med([f(x) for x in v])/1e6,3)
  r[str(bs)]={'n':len(v),'draft_ms':g(lambda x:x[2]-x[1]),'verify_ms':g(lambda x:x[3]-x[2]),'extend_ms':g(lambda x:x[4]-x[3]),'inside_ms':g(lambda x:x[4]-x[1]),
              'outside_ms':round(med([x[1]-x[5] for x in v if x[5]])/1e6,3) if any(x[5] for x in v) else None,
              'period_ms':round(med([b[1]-a[1] for a,b in zip(v,v[1:])])/1e6,3) if len(v)>1 else None}
 return r
def trace(path,steps):
 t=json.loads(path.read_text());ev=t['traceEvents'] if isinstance(t,dict) else t
 cats=collections.Counter(e.get('cat') for e in ev if e.get('ph')=='X')
 kern=[e for e in ev if e.get('ph')=='X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
 rt={}
 for e in ev:
  if e.get('ph')=='X' and e.get('cat') in ('cuda_runtime','cuda_driver'):
   c=(e.get('args') or {}).get('correlation')
   if c is not None:rt[c]=e
 ann=[e for e in ev if e.get('ph')=='X' and e.get('cat')=='user_annotation' and str(e.get('name','')).startswith(('SP:','M:'))]
 sp=sorted([e for e in ann if e['name'].startswith('SP:') and e['name']!='SP:step'],key=lambda e:e['ts'])
 mods=sorted([e for e in ann if e['name'].startswith('M:')],key=lambda e:e['ts'])
 sp_ts=[e['ts'] for e in sp];mod_ts=[e['ts'] for e in mods]
 def find_sp(ts):
  i=bisect.bisect_right(sp_ts,ts)-1
  if i>=0 and sp[i]['ts']<=ts<=sp[i]['ts']+sp[i]['dur']:return sp[i]['name']
  return 'SP:none'
 def find_mod(ts):
  # innermost = latest-starting module range that still contains ts (look back a bounded window)
  i=bisect.bisect_right(mod_ts,ts)-1;k=0
  while i>=0 and k<400:
   e=mods[i]
   if e['ts']+e['dur']>=ts:return e['name']
   i-=1;k+=1
  return None
 by_sp=collections.defaultdict(lambda:[0,0.0]);by_mod=collections.defaultdict(lambda:[0,0.0]);by_k=collections.defaultdict(lambda:[0,0.0]);by_spk=collections.defaultdict(lambda:[0,0.0]);by_modk=collections.defaultdict(lambda:[0,0.0]);unlinked=0;launch=collections.Counter()
 for e in kern:
  c=(e.get('args') or {}).get('correlation');r=rt.get(c)
  if r is None:unlinked+=1;ts=None;s='SP:unlinked'
  else:ts=r['ts'];s=find_sp(ts);launch[r['name']]+=1
  name=re.sub(r'\s+',' ',e['name'])[:140]
  by_sp[s][0]+=1;by_sp[s][1]+=e['dur'];by_k[name][0]+=1;by_k[name][1]+=e['dur'];by_spk[(s,name)][0]+=1;by_spk[(s,name)][1]+=e['dur']
  if mods and ts is not None and s!='SP:none':
   m=find_mod(ts) or s+'/<no module>';by_mod[m][0]+=1;by_mod[m][1]+=e['dur'];by_modk[(m,name)][0]+=1;by_modk[(m,name)][1]+=e['dur']
 # GPU busy union inside the profiled span
 iv=sorted((e['ts'],e['ts']+e['dur']) for e in kern);busy=0.0;gaps=[];cur=None
 for a,b in iv:
  if cur is None:cur=[a,b]
  elif a<=cur[1]:cur[1]=max(cur[1],b)
  else:busy+=cur[1]-cur[0];gaps.append(a-cur[1]);cur=[a,b]
 if cur:busy+=cur[1]-cur[0]
 span=(iv[-1][1]-iv[0][0]) if iv else 0
 f=lambda d,n=60:[{'name':k if isinstance(k,str) else ' | '.join(k),'count_per_step':round(v[0]/steps,2),'us_per_step':round(v[1]/steps,1)} for k,v in sorted(d.items(),key=lambda kv:-kv[1][1])[:n]]
 gs=sorted(gaps)
 return {'categories':dict(cats),'kernels':len(kern),'kernels_per_step':round(len(kern)/steps,1),'kernel_us_per_step':round(sum(e['dur'] for e in kern)/steps,1),'unlinked':unlinked,'launch_calls':dict(launch.most_common(8)),
         'gpu_span_us_per_step':round(span/steps,1),'gpu_busy_union_us_per_step':round(busy/steps,1),'gap_count_per_step':round(len(gaps)/steps,1),'gap_us_median':gs[len(gs)//2] if gs else None,'gap_us_p90':gs[int(len(gs)*.9)] if gs else None,'gap_us_sum_per_step':round(sum(gaps)/steps,1),
         'sp_ranges_us_per_step':{k:round(sum(e['dur'] for e in ann if e['name']==k)/steps,1) for k in sorted({e['name'] for e in ann if e['name'].startswith('SP:')})},
         'by_phase':f(by_sp),'by_module':f(by_mod,80),'by_kernel':f(by_k,70),'by_phase_kernel':f(by_spk,90),'by_module_kernel':f(by_modk,120)}
for p in sorted(D.glob('*.json')):
 if p.name.startswith('ctl') or p.name.endswith('.trace.json'):continue
 r=json.loads(p.read_text());rid=r['cfg']['id'];o={'cfg':r['cfg'],'phases':phases(r['rows'])}
 for k in ('events_error','trace_error','key_averages_seconds','trace_seconds'):
  if k in r:o[k]=r[k]
 if 'events' in r:
  n=len(r['rows']);ev=r['events']
  o['ranges']=[{'name':e['name'],'count':e['count'],'cpu_ms_per_step':round(e['cpu_total_us']/n/1e3,3),'dev_ms_per_step':round((e['dev_total_us'] or 0)/n/1e3,3)} for e in ev if e['name'].startswith('SP:')]
  dk=[e for e in ev if 'CUDA' in e['device_type']];o['ka_kernel_us_per_step']=round(sum(e['self_dev_us'] or 0 for e in dk)/n,1);o['ka_kernel_calls_per_step']=round(sum(e['count'] for e in dk)/n,1)
 tp=D/(rid+'.trace.json')
 if tp.exists():
  try:o['trace']=trace(tp,len(r['rows']));o['trace_bytes']=tp.stat().st_size
  except Exception as e:o['trace_parse_error']=repr(e)
 out[rid]=o
print(json.dumps(out,indent=1,ensure_ascii=False))
