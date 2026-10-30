from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

COLORS = ['#0072B2', '#D55E00', '#009E73', '#7C5DA8', '#CC4662', '#66727D']
CASES = [f'seed{s}_lambda{l}' for s in (1001, 1002, 1003) for l in ('0', '0.2')]
ARMS = ['seed1001_lambda0', 'seed1001_lambda0.2']


def read_table(directory, name):
    def value(x):
        if x == '': return None
        if x in ('True', 'False'): return x == 'True'
        try: return float(x)
        except ValueError: return x
    with (directory / name).open(encoding='utf-8-sig', newline='') as stream:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def choose(rows, **values):
    return [row for row in rows if all(row.get(k) == v for k, v in values.items())]


def mean(rows, key):
    a = np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=float)
    if not len(a): raise ValueError('No values for ' + key)
    return float(a.mean())


def range_line(ax, x, groups, key, color, label, scale=1, linestyle='-', marker='o'):
    arrays = [np.asarray([r[key] for r in group if r.get(key) is not None], dtype=float) * scale for group in groups]
    if not all(len(a) for a in arrays): raise ValueError('An operating point has no records')
    y = [a.mean() for a in arrays]
    ax.plot(x, y, color=color, label=label, marker=marker, linestyle=linestyle, markersize=3.5)
    ax.fill_between(x, [a.min() for a in arrays], [a.max() for a in arrays], color=color, alpha=.13, linewidth=0)


def layout(rows, cols, size):
    fig, axes = plt.subplots(rows, cols, figsize=size, layout='constrained')
    for ax in np.asarray(axes).flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.32, color='#aeb6c0', linewidth=.5)
        ax.set_axisbelow(True)
    return fig, np.asarray(axes).reshape(rows, cols)


def title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', fontweight='bold', fontsize=9)


def save(fig, output, name, notes):
    fig.savefig(output / (name + '.png'), dpi=230, facecolor='white', metadata={'Software': 'Matplotlib'})
    fig.savefig(output / (name + '.svg'), facecolor='white', metadata={'Date': None, 'Description': notes})
    plt.close(fig)
    return {'figure': name, 'formats': ['png', 'svg'], 'interpretation': notes}


def primary(data, output):
    rows, controls = data['attack_selected.csv'], data['attack_controls.csv']
    prior = mean(choose(data['image_priors.csv'], model='fitting_mean_image'), 'mean_ssim_valid')
    fig, a = layout(2, 2, (7.6, 5.1)); x = np.arange(6)
    for index, criterion in enumerate(('metric', 'loss')):
        y = [mean(choose(rows, case=c, task='reconstruction', criterion=criterion), 'mean_ssim_valid') for c in CASES]
        a[0, 0].plot(x + (index-.5)*.13, y, 'o', color=COLORS[index], label='By SSIM' if index==0 else 'By MSE', markersize=4)
        y = [100*mean(choose(rows, case=c, task='classification', criterion=criterion), 'accuracy') for c in CASES]
        a[0, 1].plot(x + (index-.5)*.13, y, 'o', color=COLORS[index], label='By accuracy' if index==0 else 'By cross-entropy', markersize=4)
    labels=[f'{s}\n$\\lambda={l}$' for s in (1001,1002,1003) for l in ('0','0.2')]
    for ax in a[0]: ax.set_xticks(x, labels); ax.legend(fontsize=7.5, loc='best')
    a[0,0].axhline(prior, color=COLORS[5], linestyle=':', linewidth=1, label='Image prior');a[0,0].set_ylabel('Valid-window SSIM');a[0,0].legend(fontsize=7.5,loc='best')
    a[0,1].axhline(10, color=COLORS[5], linestyle=':', linewidth=1);a[0,1].set_ylabel('Accuracy (%)');a[0,1].set_ylim(7.8,11.8)
    for col, (task, key, suffix) in enumerate([('reconstruction','mean_ssim_valid','SSIM'),('classification','accuracy','Accuracy (%)')]):
        scale=100 if col else 1
        for interface,color in [('identity',COLORS[2]),('fixed_transform',COLORS[3]),('unique_nonce',COLORS[0])]:
            groups=[choose(rows,case=c,task=task,criterion='metric') if interface=='unique_nonce' else choose(controls,case=c,control=interface+'_'+task,criterion='metric') for c in CASES]
            range_line(a[1,col],x,groups,key,color,{'identity':'Identity','fixed_transform':'Reused transform','unique_nonce':'Unique nonce'}[interface],scale=scale)
        a[1,col].set_xticks(x,labels);a[1,col].set_ylabel(suffix);a[1,col].set_ylim((0,1) if col==0 else (0,65));a[1,col].legend(fontsize=7,loc='best')
    for ax,l,t in zip(a.flat,'abcd',['Reconstruction selection','Class-label selection','Reconstruction controls','Class-inference controls']):title(ax,l,t)
    return save(fig,output,'evaluation_primary','Primary points are validation-selected conditional results. Stationary controls are independently fitted. Dotted references are the fitting-image prior or ten-class chance accuracy; plotted points are not confidence intervals.')


def attacks(data, output):
    fig,a=layout(2,3,(9.0,5.7));budget=data['attack_budget.csv'];snr=data['attack_snr.csv'];sets=data['attack_packet_sets.csv'];binary=data['attack_binary.csv'];attrs=data['attack_attributes.csv']
    for arm,color,label in zip(ARMS,COLORS,['$\\lambda=0$','$\\lambda=0.2$']):
        groups=[[r for r in budget if r['case']==arm and r['criterion']=='metric' and 'cnn' in r['model'] and r['budget_epochs']==b] for b in (30,60,120)]
        range_line(a[0,0],[30,60,120],groups,'mean_ssim_valid',color,label)
        for ax,model,key,style in [(a[0,1],'cnn256','mean_ssim_valid','-'),(a[0,2],'invariant','accuracy','-'),(a[0,2],'deepsets','accuracy','--')]:
            groups=[[r for r in snr if r['case']==arm and model in r['model'] and (('noiseless' in r['model']) if level=='noiseless' else r['snr_eve']==level)] for level in (20,30,40,'noiseless')]
            range_line(ax,np.arange(4),groups,key,color,label+(' DeepSets' if model=='deepsets' else ' invariant' if model=='invariant' else ''),scale=100 if key=='accuracy' else 1,linestyle=style,marker='s' if model=='deepsets' else 'o')
        counts=[1,2,4,8,16]
        groups=[[r for r in sets if r['case']==arm and r['count']==count and r['accuracy'] is not None] for count in counts]
        range_line(a[1,0],np.arange(5),groups,'accuracy',color,label,scale=100)
        values=[100*r['balanced_accuracy'] for r in binary if r['case']==arm and r['criterion']=='metric']
        j=ARMS.index(arm);a[1,1].plot(np.full(len(values),j)+np.linspace(-.06,.06,len(values)),values,'o',color=color,markersize=4)
        a[1,1].plot([j-.13,j+.13],[np.mean(values)]*2,color=color,lw=2)
        names=['mean_luminance','mean_chroma_range','finite_difference_luminance_energy']
        groups=[[r for r in attrs if r['case']==arm and r['criterion']=='metric' and r['target']==name] for name in names]
        yy=[100*mean(g,'mse_improvement_over_training_mean')/mean(g,'prior_mse') for g in groups]
        lo=[100*min(r['mse_improvement_over_training_mean']/r['prior_mse'] for r in g) for g in groups];hi=[100*max(r['mse_improvement_over_training_mean']/r['prior_mse'] for r in g) for g in groups]
        a[1,2].errorbar(np.arange(3)+(j-.5)*.13,yy,yerr=[np.array(yy)-lo,np.array(hi)-yy],fmt='o',capsize=2,color=color,label=label,markersize=4)
    a[0,0].set(xlabel='Maximum fitting epochs',ylabel='Valid-window SSIM',xticks=[30,60,120]);a[0,0].legend(fontsize=7);a[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.5f'));a[0,0].yaxis.set_major_locator(MaxNLocator(4))
    for ax in a[0,1:]:ax.set_xticks(range(4),['20','30','40','Noiseless']);ax.set_xlabel('Eve SNR (dB)')
    a[0,1].set_ylabel('Valid-window SSIM');a[0,2].set_ylabel('Accuracy (%)');a[0,2].axhline(10,color=COLORS[5],linestyle=':');a[0,2].legend(fontsize=6.1,loc='best')
    a[1,0].set(xticks=range(5),xticklabels=[str(x) for x in (1,2,4,8,16)],xlabel='Fresh packets of one image',ylabel='Accuracy (%)');a[1,0].axhline(10,color=COLORS[5],linestyle=':');a[1,0].legend(fontsize=7)
    a[1,1].set(xticks=[0,1],xticklabels=['$\\lambda=0$','$\\lambda=0.2$'],ylabel='Balanced accuracy (%)');a[1,1].axhline(50,color=COLORS[5],linestyle=':')
    a[1,2].set(xticks=range(3),xticklabels=['Luminance','Chroma','Difference\nenergy'],ylabel='MSE improvement over prior (%)');a[1,2].axhline(0,color=COLORS[5],linestyle=':')
    for ax,l,t in zip(a.flat,'abcdef',['Reconstruction budget','High-SNR reconstruction','High-SNR class labels','Joint packet inference','Animal/vehicle labels','Image attributes']):title(ax,l,t)
    return save(fig,output,'evaluation_attacks','Budget shading spans attacker starts; high-SNR shading spans recorded keys/noise draws; attribute bars span two starts. These ranges are not confidence intervals. Joint packet points are count-specific selections for repeated transmissions of the same image. Binary points are separate starts; short lines are their means.')


def natural(data, output):
    fig,a=layout(2,2,(7.6,5.3));rows=data['natural_summary.csv'];sels=data['natural_selections.csv'];frames=data['natural_frames.csv'];cases=['frozen_seed42','seed1001_lambda0.2'];labels=['Prior','Bob','Tile attack','Context attack','Identity control']
    for j,(case,color) in enumerate(zip(cases,[COLORS[5],COLORS[0]])):
        selected={r['family']:r['model'] for r in sels if r['case']==case and r['family_selected']}
        models=['mean_prior','bob',selected['tile'],selected['context'],selected['identity_context']]
        vals=[choose(rows,case=case,model=m)[0] for m in models];y=np.arange(5)+(j-.5)*.16
        xx=np.array([r['mean_ssim_valid'] for r in vals]);lo=np.array([r['mean_ssim_valid_lo95'] for r in vals]);hi=np.array([r['mean_ssim_valid_hi95'] for r in vals])
        a[0,0].errorbar(xx,y,xerr=[xx-lo,hi-xx],fmt='o',capsize=2,color=color,label='Seed 42' if j==0 else 'Seed 1001, $\\lambda=0.2$',markersize=4)
        prior=vals[0]['mean_ssim_valid'];values=vals[2:4];xx=1000*np.array([r['mean_ssim_valid']-prior for r in values]);lo=1000*np.array([r['ssim_valid_minus_prior_lo95'] for r in values]);hi=1000*np.array([r['ssim_valid_minus_prior_hi95'] for r in values])
        a[0,1].errorbar(xx,np.arange(2)+(j-.5)*.16,xerr=[xx-lo,hi-xx],fmt='o',capsize=2,color=color,markersize=4)
        for model,style in [('bob','-'),(selected['context'],'--')]:
            f=sorted(choose(frames,case=case,model=model),key=lambda r:r['frame_index']);a[1,0].plot([r['frame_index']+1 for r in f],[r['ssim_valid'] for r in f],color=color,linestyle=style,linewidth=1,label=('Seed 42' if j==0 else 'Seed 1001')+(' Bob' if model=='bob' else ' context'))
        values=[choose(sels,case=case,model=m)[0]['validation_ssim_valid'] for m in models[2:]]
        a[1,1].plot(np.arange(3)+(j-.5)*.12,values,'o',color=color,markersize=4)
    a[0,0].set(yticks=range(5),yticklabels=labels,xlabel='Whole-frame valid-window SSIM');a[0,0].invert_yaxis();a[0,0].legend(fontsize=7,loc='best')
    a[0,1].set(yticks=[0,1],yticklabels=['Tile attack','Context attack'],xlabel='SSIM minus prior ($\\times10^{-3}$)');a[0,1].invert_yaxis();a[0,1].axvline(0,color=COLORS[5],linestyle=':')
    a[1,0].set(xlabel='Kodak frame',ylabel='Valid-window SSIM',xticks=[1,6,12,18,24]);a[1,0].legend(fontsize=6.8,ncol=2)
    a[1,1].set(xticks=range(3),xticklabels=['Tile','Context','Identity'],ylabel='DIV2K validation SSIM',ylim=(.35,.95))
    for ax,l,t in zip(a.flat,'abcd',['Native-frame reconstruction','Paired differences from prior','Frame-level measurements','Validation-selected models']):title(ax,l,t)
    return save(fig,output,'evaluation_natural','Intervals in the top row are recorded 95% whole-frame bootstrap intervals on 24 Kodak frames. Bottom-left points preserve frame identity. Bottom-right points are the validation-selected family members; validation and test quantities are not pooled. Both attack families train on DIV2K; joint context spans four adjacent tiles.')


def channels(data, output):
    fig,a=layout(2,3,(9.0,5.7));rows=data['channel_outcomes.csv'];reuse=data['reuse_recovery.csv'];snrs=[0,5,10,15,20,30];trained=[c for c in CASES if c.endswith('lambda0.2')]
    for ax,prefix in [(a[0,0],'flat_rayleigh'),(a[0,1],'flat_rician'),(a[0,2],'multipath_cfo')]:
        for receiver,color in [('perfect',COLORS[0]),('estimated',COLORS[1])]:
            for interface,style,marker in [('identity','-',None),('keyed','--','o')]:
                groups=[[r for r in rows if r['case'] in trained and r['scenario']==prefix+'_'+receiver and r['snr_db']==s and r['interface']==interface] for s in snrs]
                yy=[mean(g,'mean_ssim_valid') for g in groups];ax.plot(snrs,yy,color=color,linestyle=style,marker=marker,markersize=3,label=receiver.title()+': '+interface)
        ax.set(xlabel='Payload SNR (dB)',ylabel='Bob valid-window SSIM');ax.legend(fontsize=6.5,loc='lower right')
    for scenario,color,label in [('multipath_cfo_estimated',COLORS[0],'Estimated correction'),('multipath_cfo_unsynchronized',COLORS[4],'Uncorrected')]:
        for interface,style in [('identity','-'),('keyed','--')]:
            yy=[mean([r for r in rows if r['case'] in trained and r['scenario']==scenario and r['snr_db']==s and r['interface']==interface],'mean_ssim_valid') for s in snrs]
            a[1,0].plot(snrs,yy,color=color,linestyle=style,marker='o' if interface=='keyed' else None,markersize=3,label=label+': '+interface)
    a[1,0].set(xlabel='Payload SNR (dB)',ylabel='Bob valid-window SSIM');a[1,0].legend(fontsize=6.5,loc='lower right')
    counts=[1,2,4,8,16,32,64,128,192,256,512,4096]
    for arm,color,label in zip(ARMS,COLORS,['$\\lambda=0$','$\\lambda=0.2$']):
        groups=[choose(reuse,case=arm,pairs=m,kind='known_plaintext_linear_inverse') for m in counts]
        range_line(a[1,1],counts,groups,'mean_ssim_valid',color,label)
        range_line(a[1,2],counts,groups,'design_condition_nonzero',color,label)
        a[1,1].axhline(mean(choose(reuse,case=arm,kind='correct_inverse'),'mean_ssim_valid'),color=color,linestyle=':',linewidth=.9)
    for ax in a[1,1:]:ax.set_xscale('log',base=2);ax.set_xticks([1,16,256,4096],['1','16','256','4096']);ax.set_xlabel('Diverse known-image pairs');ax.legend(fontsize=7)
    a[1,1].set_ylabel('Eve valid-window SSIM');a[1,2].set_yscale('log');a[1,2].set_ylabel('Retained-spectrum condition number')
    for ax,l,t in zip(a.flat,'abcdef',['Flat Rayleigh','Flat Rician','Synchronized multipath','Timing/frequency correction','Reused-transform recovery','Inverse-fit conditioning']):title(ax,l,t)
    return save(fig,output,'evaluation_channels','Channel curves average the three defense-trained codecs and three recorded draws, with paired identity/keyed conditions. Reuse curves show means and ranges over three known-image subsets for the two seed-1001 codecs; ranges are not confidence intervals. Dotted reuse references are correct-inverse reconstruction. The channel has packet-constant taps and a frequency-offset proxy.')


def resources(data, output):
    fig,a=layout(3,3,(9.4,8.2));rows=data['protection_operating_points.csv'];case='seed1001_lambda0.2'
    schemes=['unprotected_identity','keyed_h256_g32','aes4','aes8','bfv'];names=['Identity','Keyed auth., B=32','AES / 4 bit','AES / 8 bit','Linear BFV']
    for ax,budget in zip(a[0,:2],[640,1024]):
        for scheme,label,color in zip(schemes,names,COLORS):
            points=sorted(choose(rows,case=case,scheme=scheme,budget_complex_uses=budget),key=lambda r:r['snr_db']);ax.plot([r['snr_db'] for r in points],[r['mean_ssim_valid'] for r in points],marker='o',color=color,markersize=3,label=label)
        ax.set(xlabel='Nominal Bob SNR (dB)',ylabel='Valid-window SSIM',ylim=(0,1))
    a[0,0].legend(fontsize=6.4,loc='lower right')
    vals=[choose(rows,case=case,scheme=s,budget_complex_uses=640,snr_db=20)[0] for s in schemes]
    labels=['Identity','Keyed','AES4','AES8','BFV'];a[0,2].plot(np.arange(5),[r['eve_ssim_valid'] for r in vals],'o',color=COLORS[0]);a[0,2].axhline(mean(choose(data['image_priors.csv'],case=case,model='fitting_mean_image'),'mean_ssim_valid'),color=COLORS[5],linestyle=':');a[0,2].set(xticks=range(5),xticklabels=labels,ylabel='Eve valid-window SSIM',ylim=(0,1))
    active=['keyed_h128_g32','keyed_h256_g32','keyed_h256_g1','aes4','aes8','bfv'];values=[choose(rows,case=case,scheme=s,budget_complex_uses=640,snr_db=20)[0]['active_complex_uses_per_image'] for s in active]
    bars=a[1,0].bar(range(6),values,color=[COLORS[0],COLORS[0],COLORS[3],COLORS[2],COLORS[1],COLORS[4]])
    a[1,0].set(xticks=range(6),xticklabels=['CRC\nB32','Auth\nB32','Auth\nB1','AES4','AES8','BFV'],ylabel='Active complex uses / image',ylim=(0,650))
    for bar,v in zip(bars,values):a[1,0].text(bar.get_x()+bar.get_width()/2,v+10,f'{v:.0f}',ha='center',fontsize=7)
    timings=data['protection_timings.csv']
    for j,op in enumerate(['encrypt','decrypt']):
        y=[choose(timings,case=case,component=s+'_'+op)[0]['median_ms_per_image'] for s in ('aes4','bfv')]
        a[1,1].bar(np.arange(2)+(j-.5)*.32,y,width=.3,color=COLORS[j],label=op.title())
    a[1,1].set(xticks=[0,1],xticklabels=['AES4','Linear BFV'],ylabel='CPU median ms / image');a[1,1].set_yscale('log');a[1,1].legend(fontsize=7)
    header=data['header_reliability.csv']
    for scheme,color,label in [('hamming_nonce96_count16_crc16',COLORS[0],'CRC'),('hamming_authenticated_256bit_information',COLORS[1],'Authenticated')]:
        points=sorted(choose(header,scheme=scheme),key=lambda r:r['snr_db_complex_symbol']);nz=[r for r in points if r['failure_rate']>0];zero=[r for r in points if r['failure_rate']==0]
        a[1,2].plot([r['snr_db_complex_symbol'] for r in nz],[r['failure_rate'] for r in nz],'-o',color=color,label=label,markersize=3)
        a[1,2].fill_between([r['snr_db_complex_symbol'] for r in nz],[r['failure_lo95'] for r in nz],[r['failure_hi95'] for r in nz],color=color,alpha=.12)
        a[1,2].plot([r['snr_db_complex_symbol'] for r in zero],[r['zero_error_one_sided95_upper'] for r in zero],'v',color=color,markersize=4)
    a[1,2].set(xlabel='Header SNR (dB)',ylabel='Failure fraction',yscale='log');a[1,2].legend(fontsize=7);a[1,2].text(.04,.1,'20,000 trials / point\nTriangles: 95% upper bounds',transform=a[1,2].transAxes,fontsize=6.7)
    header=data['header_resources.csv']
    for authenticated,color,label in [(False,COLORS[0],'CRC'),(True,COLORS[1],'Authenticated')]:
        points=sorted(choose(header,authenticated_metadata=authenticated),key=lambda r:r['group_size']);a[2,0].plot([r['group_size'] for r in points],[100*r['header_overhead_fraction'] for r in points],'-o',color=color,label=label,markersize=3)
    a[2,0].set(xlabel='Tiles per header, B',ylabel='Header uses / payload uses (%)',xscale='log',yscale='log',xticks=[1,8,32,128],xticklabels=['1','8','32','128']);a[2,0].legend(fontsize=7)
    timings=data['protocol_timings.csv'];components=['parameter_generation_sender','forward_transform','durable_nonce_reservation','receiver_auth_and_durable_replay'];labels=['Parameters (tile)','Transform (tile)','Reservation (counter)','Auth.+replay (header)']
    for j,batch in enumerate((1,128)):
        yy=[choose(timings,batch=batch,component=c)[0]['median_ms_per_loop_item'] for c in components];a[2,1].barh(np.arange(4)+(j-.5)*.3,yy,height=.28,color=COLORS[j],label='Batch '+str(batch))
    a[2,1].set(yticks=range(4),yticklabels=labels,xlabel='CPU median ms / stated item',xscale='log');a[2,1].invert_yaxis();a[2,1].tick_params(axis='y',labelsize=6.4);a[2,1].legend(fontsize=6.8)
    for batch,color in [(1,COLORS[0]),(128,COLORS[1])]:
        points=sorted(choose(data['protocol_processing.csv'],measurement_batch=batch),key=lambda r:r['group_size']);a[2,2].plot([r['group_size'] for r in points],[r['derived_extra_processing_ms_for_384_tiles'] for r in points],'-o',color=color,label='Batch '+str(batch),markersize=3)
    a[2,2].set(xlabel='Tiles per authenticated header',ylabel='Derived ms / 384 tiles',xscale='log',yscale='log',xticks=[1,8,32,128],xticklabels=['1','8','32','128']);a[2,2].legend(fontsize=7)
    titles=['Bob: 640 reserved uses','Bob: 1024 reserved uses','Compact Eve reconstruction','Active transmission resources','Cryptographic operations','Header reliability','Header amortization','Measured component costs','Derived processing sum']
    for ax,l,t in zip(a.flat,'abcdefghi',titles):title(ax,l,t)
    return save(fig,output,'evaluation_resources','Protection profiles use a fixed common codec and equal reserved uses/nominal energy, uncoded digital modulation and hard decisions. Digital compact attacks are secondary evidence. Header shading is a recorded binomial 95% interval; zero-failure triangles are one-sided upper bounds. Component timing units are shown per stated item; reservation cost is batch-amortized. Derived totals sum component medians and exclude codec/radio/queueing delays.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=ROOT/'data/evaluation')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/evaluation')
    args=parser.parse_args();args.output_dir.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.7,'lines.linewidth':1.2,'svg.fonttype':'none','svg.hashsalt':'keyed-jscc-evaluation','savefig.bbox':'tight'})
    names=['attack_selected.csv','attack_controls.csv','image_priors.csv','attack_budget.csv','attack_snr.csv','attack_packet_sets.csv','attack_binary.csv','attack_attributes.csv','natural_summary.csv','natural_selections.csv','natural_frames.csv','channel_outcomes.csv','reuse_recovery.csv','protection_operating_points.csv','protection_timings.csv','header_reliability.csv','header_resources.csv','protocol_timings.csv','protocol_processing.csv']
    data={name:read_table(args.input_dir,name) for name in names}
    records=[function(data,args.output_dir) for function in (primary,attacks,natural,channels,resources)]
    (args.output_dir/'figure_notes.json').write_text(json.dumps({'input_scope':'Recorded aggregate tables; no model fitting or raw-bootstrap recomputation','figures':records},indent=2)+'\n')
    print(json.dumps({'figures':len(records),'files':2*len(records)+1,'output':str(args.output_dir)},indent=2))


if __name__=='__main__':
    main()
