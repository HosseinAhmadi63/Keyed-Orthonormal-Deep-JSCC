from pathlib import Path
import csv
import numpy as np
from keyed_jscc.public_sampling import theoretical_moments
from .common import actual_observations, save_npz, write_json

def statistics(u):
    x=np.asarray(u,dtype=np.float64)
    return np.column_stack((np.mean(x**2,1),np.mean(x**4,1),np.mean(x**6,1),np.mean(x,1),np.mean(x<0,1)))

def run(case,cfg,outdir,device,smoke=False):
    outdir=Path(outdir);outdir.mkdir(exist_ok=True,parents=True)
    count=min(16 if smoke else 3600,len(case['test_u']))
    u=case['test_u'][:count];labels=case['test_labels'][:count]
    names=['second_moment','fourth_moment','sixth_moment','signed_mean','negative_fraction']
    arrays=dict(labels=labels,latent_statistics=statistics(u))
    summaries={}
    for snr in (20.0,float('inf')):
        obs,_=actual_observations(u,case,'diagnostics',snr)
        actual=statistics(obs[:,0]);key='noiseless' if np.isinf(snr) else 'snr20'
        theory=theoretical_moments(u,0 if np.isinf(snr) else 10**(-snr/10))
        arrays[key+'_observable_statistics']=actual
        arrays[key+'_expected_moments']=np.column_stack([theory[n] for n in ('second','fourth','sixth')])
        summaries[key]={str(int(c)):dict(n=int(np.sum(labels==c)),mean=actual[labels==c].mean(0),standard_deviation=actual[labels==c].std(0)) for c in np.unique(labels)}
    save_npz(outdir/'observable_statistics.npz',**arrays)
    summary=dict(case=case['name'],statistics=names,class_conditional=summaries,unit='one original image; moments across 256 real latent coordinates',purpose='distribution diagnostics; no model fitting or selection on test data',scientific_results=not smoke)
    write_json(outdir/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for row,domain in enumerate(('latent_statistics','noiseless_observable_statistics')):
        for col,stat in enumerate((1,2,4)):
            ax=axes[row,col]
            values=[arrays[domain][labels==c,stat] for c in np.unique(labels)]
            ax.boxplot(values,showfliers=False)
            ax.set(xlabel='CIFAR-10 class index',ylabel=names[stat].replace('_',' '),title=('Encoder latent' if row==0 else 'Scrambled noiseless observation'))
            ax.set_xticklabels([str(c) for c in np.unique(labels)])
    fig.savefig(outdir/'statistical_diagnostics.png',dpi=200)
    fig.savefig(outdir/'statistical_diagnostics.pdf')
    plt.close(fig)
    return summary

def training_summary(cfg,outdir):
    source=Path(cfg['assets_dir'])/'codec_training_curves.csv'
    with source.open(newline='') as stream:
        records=list(csv.DictReader(stream))
    cases={}
    for name in sorted(set(r['case'] for r in records)):
        rows=[r for r in records if r['case']==name][-5:]
        cases[name]={field:float(np.mean([float(r[field]) for r in rows])) for field in ('kl_uniform','online_cross_entropy','fitting_mse','validation_mse')}
    write_json(Path(outdir)/'training_loss_summary.json',dict(last_epochs=5,cases=cases,uniform_cross_entropy=float(np.log(10)),source='retained executed codec training records',scope='online ten-class defense head training statistics; frozen codec performance assessed separately'))
