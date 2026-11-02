"""Targeted analysis of the nine serial diagnosis measurements."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import median

os.environ['MPLCONFIGDIR'] = '/tmp/meta-early-write-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load')
OUT = ROOT/'validation/meta-early-write-end-20261102'
GIB = 1024**3
COLORS = {'cpu':'#a34425','meta':'#087fa0','meta-ballast':'#7847a0'}
NAMES = {'cpu':'CPU','meta':'Meta · 64 MiB ring','meta-ballast':'Meta + 6.05 GiB ballast'}


def load(p):
    return json.loads(p.read_text())


def dump(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def describe(x):
    a=np.asarray(x,dtype=float)
    if not len(a):
        return dict(count=0,sum=0,median=0,p95=0,max=0)
    return dict(count=len(a),sum=float(a.sum()),median=float(np.median(a)),
                p95=float(np.percentile(a,95)),max=float(a.max()))


def intersection_events(pwrites):
    boundaries=[]
    for rank,events in enumerate(pwrites):
        for s,d,_ in events:
            if d>20_000_000:
                boundaries.extend([(s,rank,1),(s+d,rank,-1)])
    active=set(); begin=None; result=[]
    for t,r,delta in sorted(boundaries):
        if delta==1:
            active.add(r)
        else:
            active.discard(r)
        if len(active)==4 and begin is None:
            begin=t
        elif len(active)<4 and begin is not None:
            if t-begin>=20_000_000:
                result.append((begin,t))
            begin=None
    return result


def analyze(case):
    condition=case.name.split('-',1)[1]; repeat=int(case.name[1])
    sam=[json.loads(s) for s in (case/'vmstat.jsonl').read_text().splitlines()]
    ranks=[]; pws=[]; stages=[]; starts=[]; ends=[]; signatures=[]; io_origins=[]
    for rank in range(4):
        report=load(case/'save'/f'rank-{rank:03d}.json')
        starts.append(report['stats']['dcp_started_ns']); ends.append(report['stats']['dcp_ended_ns'])
        io_origins.append(report['io']['origin_ns'])
        pw=sorted((e['start_ns'],e['duration_ns'],e['completed'])
                  for e in report['payload_events'] if e['op']=='pwrite')
        sig=[(e['file'],e['offset'],e['requested'],e['completed'],e['success'])
             for e in report['payload_events'] if e['op']=='pwrite']
        signatures.append(hashlib.sha256(json.dumps(sig,separators=(',',':')).encode()).hexdigest())
        assert all(e['success'] and e['completed']==e['requested']
                   for e in report['payload_events'] if e['op']=='pwrite')
        pws.append(pw)
        stage=defaultdict(list); first2=defaultdict(list)
        for line in (case/'save'/f'rank-{rank:03d}.io.jsonl').open():
            event=json.loads(line)
            if event.get('io_layer')=='staging':
                stage[event['op']].append(event['duration_ns']/1e9)
                absolute=event['start_ns']+io_origins[-1]
                if pw[0][0]<=absolute<pw[0][0]+2_000_000_000:
                    first2[event['op']].append(event['duration_ns']/1e9)
        stages.append({k:describe(v) for k,v in stage.items()})
        gaps=[max(0,b[0]-a[0]-a[1])/1e9 for a,b in zip(pw,pw[1:])]
        fast=[d/1e9 for _,d,_ in pw if d<1_000_000]
        stall=[d/1e9 for _,d,_ in pw if d>20_000_000]
        ranks.append(dict(rank=rank,pwrite_calls=len(pw),payload_bytes=sum(b for _,_,b in pw),
                          stalled_pwrite=describe(stall),gap_seconds=describe(gaps),
                          fast_pwrite_seconds=describe(fast),
                          stages=stages[-1],first_2s_stages={k:describe(v) for k,v in first2.items()}))
        del report
    first=min(pw[0][0] for pw in pws); last=max(s+d for pw in pws for s,d,_ in pw)
    begin=min(starts); end=max(ends)
    active=[s for s in sam if first<=s['absolute_ns']<=last]
    timed=[s for s in sam if begin<=s['absolute_ns']<=end]
    assert sam[0]['absolute_ns']<begin and sam[-1]['absolute_ns']>end
    diag=load(case/'diagnostic.json')
    if condition=='meta-ballast':
        assert diag['ballast']['fully_faulted_ns']<begin
    def vm(k,ss=active):
        return np.array([s['vmstat'][k] for s in ss],float)
    def near(t):
        return min(sam,key=lambda s:abs(s['absolute_ns']-t))
    journal=[]; prev=None
    for s in sam:
        n=int(s['journal_info'].split()[0])
        if prev is not None and n>prev:
            journal.append(dict(time_s=(s['absolute_ns']-first)/1e9,count=n-prev,absolute_ns=s['absolute_ns']))
        prev=n
    common=[]
    for a,b in intersection_events(pws):
        nn=near(b)
        closest=min(journal,key=lambda j:abs(j['absolute_ns']-b)) if journal else None
        common.append(dict(start_s=(a-first)/1e9,end_s=(b-first)/1e9,duration_ms=(b-a)/1e6,
                           dirty_gib=nn['vmstat']['nr_dirty']*4096/GIB,
                           dirty_threshold_gib=nn['vmstat']['nr_dirty_threshold']*4096/GIB,
                           nearest_commit_delta_ms=(closest['absolute_ns']-b)/1e6 if closest else None))
    firstslow=min((s for pw in pws for s,d,_ in pw if d>20_000_000),default=None)
    bins=np.arange(0,(end-first)/1e9+0.1,0.05)
    times=np.array([(s+d-first)/1e9 for pw in pws for s,d,_ in pw])
    sizes=np.array([b/1e6 for pw in pws for _,_,b in pw])
    mb,_=np.histogram(times,bins=bins,weights=sizes)
    smooth=np.convolve(mb,np.ones(10)/10,'valid')
    low=next((float(bins[i]) for i,x in enumerate(smooth) if bins[i]>=2 and x<35),None)
    freerun=(vm('nr_dirty_threshold')+vm('nr_dirty_background_threshold'))/2
    crosses=np.flatnonzero(vm('nr_dirty')+vm('nr_writeback')>=freerun)
    threshold0=near(first)['vmstat']['nr_dirty_threshold']*4096/GIB
    result=dict(case=case.name,repeat=repeat,condition=condition,
                absolute_first_pwrite_ns=first,absolute_save_start_ns=begin,absolute_save_end_ns=end,
                write_end_s=(last-first)/1e9,write_end_from_save_start_s=(last-begin)/1e9,
                total_save_s=(end-begin)/1e9,post_write_s=(end-last)/1e9,
                first_pwrite_from_save_start_s=(first-begin)/1e9,
                payload_bytes=sum(r['payload_bytes'] for r in ranks),
                stalled_pwrite_s_per_rank=median(r['stalled_pwrite']['sum'] for r in ranks),
                gaps_s_per_rank=median(r['gap_seconds']['sum'] for r in ranks),
                host_copy_s_per_rank=median(s.get('host_copy',{}).get('sum',0) for s in stages),
                host_materialize_s_per_rank=median(s.get('host_materialize',{}).get('sum',0) for s in stages),
                pwrite_signature_sha256_by_rank=signatures,
                dirty_threshold_first_gib=threshold0,
                dirty_threshold_median_gib=float(np.median(vm('nr_dirty_threshold')))*4096/GIB,
                dirty_threshold_min_gib=float(vm('nr_dirty_threshold').min())*4096/GIB,
                dirty_threshold_max_gib=float(vm('nr_dirty_threshold').max())*4096/GIB,
                background_threshold_median_gib=float(np.median(vm('nr_dirty_background_threshold')))*4096/GIB,
                dirty_peak_gib=float(vm('nr_dirty').max())*4096/GIB,
                dirty_peak_fraction_threshold=float((vm('nr_dirty')/vm('nr_dirty_threshold')).max()),
                dirty_at_write_end_gib=near(last)['vmstat']['nr_dirty']*4096/GIB,
                dirty_plus_writeback_at_write_end_gib=(near(last)['vmstat']['nr_dirty']+
                                                      near(last)['vmstat']['nr_writeback'])*4096/GIB,
                disk_written_by_write_end_gib=(int(near(last)['diskstats'][9])-
                                              int(near(first)['diskstats'][9]))*512/GIB,
                disk_written_during_save_gib=(int(near(end)['diskstats'][9])-
                                             int(near(begin)['diskstats'][9]))*512/GIB,
                anon_median_gib=float(np.median(vm('nr_anon_pages')))*4096/GIB,
                free_plus_file_median_gib=float(np.median(vm('nr_free_pages')+vm('nr_active_file')+
                                                         vm('nr_inactive_file')))*4096/GIB,
                first_slow_pwrite_s=(firstslow-first)/1e9 if firstslow else None,
                throughput_drop_s=low,
                first_freerun_crossing_s=(active[int(crosses[0])]['absolute_ns']-first)/1e9 if len(crosses) else None,
                first_freerun_crossing_from_save_start_s=(active[int(crosses[0])]['absolute_ns']-begin)/1e9 if len(crosses) else None,
                sampler_read_ms=describe([s['read_duration_ns']/1e6 for s in sam]),
                sampler_interval_ms=describe(np.diff([s['absolute_ns'] for s in sam])/1e6),
                all_rank_stalls=common,journal_commits=journal,ranks=ranks,
                samples=len(sam),ballast=diag.get('ballast'))
    # Avoid loading summary.json's duplicate copies of every event.
    memory=[json.loads(s) for s in (case/'save'/'memory.jsonl').read_text().splitlines()]
    result['job_peak_pss_gib']=max(m['total_pss_bytes'] for m in memory)/GIB
    result['job_peak_anon_pss_gib']=max(m['total_anon_pss_bytes'] for m in memory)/GIB
    dump(case/'analysis.json',result)
    series=dict(time_s=[(s['absolute_ns']-first)/1e9 for s in timed],
                dirty_gib=[s['vmstat']['nr_dirty']*4096/GIB for s in timed],
                threshold_gib=[s['vmstat']['nr_dirty_threshold']*4096/GIB for s in timed],
                throughput_time_s=((bins[:-1]+bins[1:])/2).tolist(),throughput_mb_50ms=mb.tolist())
    return result,series


def figure(runs,series,path,title):
    fig,axes=plt.subplots(2,1,figsize=(12,6.8),sharex=True,layout='constrained')
    for row in runs:
        c=row['condition']; a=series[row['case']]; color=COLORS[c]
        axes[0].plot(a['throughput_time_s'],a['throughput_mb_50ms'],color=color,lw=0.85,
                     alpha=.83,label=NAMES[c])
        axes[1].plot(a['time_s'],a['dirty_gib'],color=color,lw=1.3,label=NAMES[c]+' · dirty')
        axes[1].plot(a['time_s'],a['threshold_gib'],color=color,lw=1.3,ls='--',
                     label=NAMES[c]+' · threshold')
    axes[0].set_ylabel('Completed pwrite MB / 50 ms')
    axes[1].set_ylabel('GiB (pages × 4096)')
    axes[1].set_xlabel('Elapsed real seconds from first payload pwrite (0 = first write)')
    axes[0].set_title(title+'\nReal elapsed time · full save interval')
    for ax in axes:
        ax.grid(alpha=.2); ax.set_ylim(bottom=0); ax.legend(fontsize=8,ncol=3,loc='upper right')
    axes[1].set_xlim(left=min((r['absolute_save_start_ns']-r['absolute_first_pwrite_ns'])/1e9 for r in runs)-.1,
                     right=max((r['absolute_save_end_ns']-r['absolute_first_pwrite_ns'])/1e9 for r in runs)+.2)
    fig.savefig(path.with_suffix('.png'),dpi=160)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    progress=load(OUT/'progress.json')
    assert progress['status'] in ('stage-complete','complete') and len(progress['completed'])==9
    rows=[]; series={}
    for case in sorted((OUT/'measurements'/'3b').iterdir()):
        row,data=analyze(case); rows.append(row); series[row['case']]=data
        print(row['case'], {k:round(row[k],4) if isinstance(row[k],float) else row[k] for k in
              ('write_end_s','total_save_s','stalled_pwrite_s_per_rank','dirty_threshold_median_gib',
               'dirty_at_write_end_gib','first_slow_pwrite_s','throughput_drop_s',
               'host_copy_s_per_rank','host_materialize_s_per_rank','gaps_s_per_rank')},flush=True)
    reference=rows[0]['pwrite_signature_sha256_by_rank']
    assert all(r['pwrite_signature_sha256_by_rank']==reference for r in rows)
    assert all(r['payload_bytes']==6794207232 for r in rows)
    numeric=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in
             ('repeat','absolute_first_pwrite_ns','absolute_save_start_ns','absolute_save_end_ns')]
    medians={c:{k:median(r[k] for r in rows if r['condition']==c and r[k] is not None)
                for k in numeric if all(r[k] is not None for r in rows if r['condition']==c)}
             for c in COLORS}
    dump(OUT/'analysis.json',dict(runs=rows,medians=medians,requests_identical_all_nine=True))
    dump(OUT/'plot-series.json',series)
    with (OUT/'run-metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['case','condition','repeat']+numeric); w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    with (OUT/'all-rank-stalls.csv').open('w') as f:
        fields=['case','start_s','end_s','duration_ms','dirty_gib','dirty_threshold_gib',
                'nearest_commit_delta_ms']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in rows:
            w.writerows(dict(case=row['case'],**e) for e in row['all_rank_stalls'])
    plots=OUT/'plots'; plots.mkdir(exist_ok=True)
    for repeat in range(1,4):
        figure([r for r in rows if r['repeat']==repeat],series,plots/f'round-{repeat}',f'3B · 4 ranks · shared safetensors · round {repeat}')
    for row in rows:
        figure([row],series,plots/row['case'],f"3B · 4 ranks · {row['case']}")
    print('MEDIANS',json.dumps(medians,indent=2),flush=True)


if __name__=='__main__':
    main()
