from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / 'outputs' / 't20wc_visualization.csv').exists():
    ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
PLOTS = ROOT / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(OUT / 't20wc_visualization.csv')
perf = pd.read_csv(OUT / 't20wc_performance.csv')
assert len(df) == len(perf), 'Visualization/performance datasets have different row counts'
for c in ['R','B','4s','6s','SR']:
    df[c] = pd.to_numeric(df[c], errors='raise')

NAVY='#142B49'; BLUE='#3175B9'; TEAL='#138B88'; GOLD='#E3A13D'; CORAL='#D66D60'; PURPLE='#8065AB'
CAT=[TEAL,GOLD,CORAL]
sns.set_theme(style='whitegrid', context='paper', font_scale=1.25)
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white','axes.facecolor':'white',
 'axes.titlesize':15,'axes.titleweight':'bold','axes.titlecolor':NAVY,'axes.labelsize':11,
 'axes.labelcolor':NAVY,'xtick.labelsize':10,'ytick.labelsize':10,'grid.alpha':.17,
 'grid.color':'#8293A7','axes.spines.top':False,'axes.spines.right':False,
 'savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})

def canvas(size=(10,6)):
    fig,ax=plt.subplots(figsize=size,layout='constrained')
    return fig,ax

def finish(fig,ax,name,note=None):
    if note:
        fig.text(.01,.005,note,fontsize=8,color='#596779')
    ax.grid(axis='y',alpha=.16) if name not in ('07_heatmap','04_pie_chart','12_hexbin_plot') else None
    for ext in ('png','pdf'):
        fig.savefig(PLOTS / f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.22)
    plt.close(fig)
    print(f'Saved: {name}.png and {name}.pdf')

# 1: Bar chart
s=df.groupby('Batsman')['R'].sum().nlargest(10).sort_values()
fig,ax=canvas((11,6.2))
bars=ax.barh(s.index,s.values,color=[BLUE]*9+[TEAL],height=.7)
ax.bar_label(bars,padding=5,fontsize=10,fmt='%d')
ax.set_xlim(0,s.max()*1.15)
ax.set(title='Top 10 Run Scorers | T20 World Cups',xlabel='Total runs',ylabel='')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
finish(fig,ax,'01_bar_chart','Aggregated across 2007–2016 tournaments in the cleaned dataset.')

# 2: Histogram
fig,ax=canvas()
sns.histplot(df,x='R',bins=np.arange(-.5,df.R.max()+5,5),color=BLUE,edgecolor='white',ax=ax)
ax.axvline(df.R.median(),color=CORAL,ls='--',lw=2,label=f'Median: {df.R.median():.0f}')
ax.set(title='Runs Scored per Batting Innings',xlabel='Runs',ylabel='Number of innings')
ax.legend(frameon=False)
finish(fig,ax,'02_histogram')

# 3: Boxplot
fig,ax=canvas()
sns.boxplot(data=df,x='Season',y='R',color='#80C9C2',width=.58,linewidth=1.2,
 flierprops={'marker':'o','markersize':3,'alpha':.42,'markerfacecolor':CORAL,'markeredgecolor':'none'},ax=ax)
ax.set(title='Distribution of Runs by Tournament Year',xlabel='Tournament year',ylabel='Runs per innings')
finish(fig,ax,'03_box_plot','Box = interquartile range; central line = median; dots = potential outliers.')

# 4: Donut pie
counts=perf.Performance.value_counts().reindex(['Low','Medium','High'],fill_value=0)
fig,ax=canvas((8.5,6.5))
wedges,texts,autotexts=ax.pie(counts,labels=[f'{k} ({v:,})' for k,v in counts.items()],
 colors=CAT,startangle=90,counterclock=False,autopct='%1.1f%%',pctdistance=.79,
 wedgeprops={'width':.42,'edgecolor':'white','linewidth':2},textprops={'fontsize':11,'color':NAVY})
ax.text(0,0,f'{counts.sum():,}\ninnings',ha='center',va='center',fontsize=15,fontweight='bold',color=NAVY)
ax.set_title('Batting Performance Categories')
finish(fig,ax,'04_pie_chart','Low: 0–19 runs  |  Medium: 20–49 runs  |  High: 50+ runs (assumed thresholds).')

# 5: Line
s=df.groupby('Season').R.mean().sort_index()
fig,ax=canvas()
ax.plot(s.index,s.values,color=TEAL,marker='o',markersize=8,lw=2.5)
for x,y in s.items(): ax.annotate(f'{y:.2f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9,color=NAVY)
ax.set_ylim(s.min()-1,s.max()+1)
ax.set_xticks(s.index)
ax.set(title='Average Runs per Batting Innings by Season',xlabel='Tournament year',ylabel='Mean runs per innings')
finish(fig,ax,'05_line_chart','Descriptive averages; not adjusted for number of matches or playing conditions.')

# 6: Scatter with fitted line
fig,ax=canvas()
sns.regplot(data=df,x='B',y='R',scatter_kws={'alpha':.24,'s':17,'color':BLUE,'edgecolors':'none'},
 line_kws={'color':CORAL,'lw':2},ci=None,ax=ax)
r=df.B.corr(df.R)
ax.text(.03,.95,f'Pearson r = {r:.2f}',transform=ax.transAxes,va='top',color=NAVY,
 bbox={'facecolor':'white','edgecolor':'#DDE4EB','boxstyle':'round,pad=.4'})
ax.set(title='Balls Faced vs Runs Scored',xlabel='Balls faced',ylabel='Runs')
finish(fig,ax,'06_scatter_plot','Trend line is descriptive and does not imply causation.')

# 7: Correlation heatmap
cols=['R','B','4s','6s','SR']
corr=df[cols].corr()
fig,ax=canvas((8,6.7))
sns.heatmap(corr,annot=True,fmt='.2f',cmap='RdBu_r',vmin=-1,vmax=1,center=0,square=True,
 linewidths=1,linecolor='white',cbar_kws={'label':'Pearson correlation','shrink':.85},ax=ax)
ax.set_title('Correlation Between Batting Metrics',pad=15)
ax.tick_params(axis='x',rotation=0);ax.tick_params(axis='y',rotation=0)
finish(fig,ax,'07_heatmap','R = runs; B = balls; 4s = fours; 6s = sixes; SR = strike rate.')

# 8: Violin
fig,ax=canvas()
sns.violinplot(data=df,x='Innings',y='SR',color='#B5A0D9',inner='quartile',cut=0,linewidth=1,ax=ax)
ax.set(title='Strike Rate Distribution by Innings',xlabel='Innings number',ylabel='Strike rate')
finish(fig,ax,'08_violin_plot','Width reflects estimated density; internal lines show quartiles.')

# 9: Countplot
fig,ax=canvas()
s=df.Season.value_counts().sort_index()
bars=ax.bar([str(x) for x in s.index],s.values,color=BLUE,width=.64)
ax.bar_label(bars,padding=4,fontsize=10)
ax.set_ylim(0,s.max()*1.14)
ax.set(title='Number of Batting Records by Season',xlabel='Tournament year',ylabel='Batting records')
finish(fig,ax,'09_count_plot','Raw record counts are not normalized by the number of matches.')

# 10: KDE
fig,ax=canvas()
sns.kdeplot(data=df,x='SR',fill=True,color=TEAL,alpha=.23,bw_adjust=1,cut=0,ax=ax)
ax.axvline(df.SR.mean(),color=CORAL,lw=2,ls='--',label=f'Mean: {df.SR.mean():.1f}')
ax.axvline(df.SR.median(),color=NAVY,lw=2,ls=':',label=f'Median: {df.SR.median():.1f}')
ax.set_xlim(left=0)
ax.set(title='Strike Rate Density',xlabel='Strike rate',ylabel='Estimated density')
ax.legend(frameon=False)
finish(fig,ax,'10_kde_plot','Smoothed density estimate; exact frequencies are shown by a histogram.')

# 11: Stacked bars
ct=pd.crosstab(perf.Season,perf.Performance).reindex(columns=['Low','Medium','High'],fill_value=0)
fig,ax=canvas((10,6))
bottom=np.zeros(len(ct))
for label,color in zip(ct.columns,CAT):
    vals=ct[label].to_numpy()
    ax.bar([str(x) for x in ct.index],vals,bottom=bottom,color=color,label=label,width=.64)
    bottom+=vals
for i,total in enumerate(bottom):ax.text(i,total+7,f'{int(total):,}',ha='center',fontsize=9,color=NAVY)
ax.set_ylim(0,max(bottom)*1.14)
ax.set(title='Batting Performance Mix by Season',xlabel='Tournament year',ylabel='Batting records')
ax.legend(title='Runs category',frameon=False,ncol=3,loc='upper left')
finish(fig,ax,'11_stacked_bar_chart','Low: 0–19  |  Medium: 20–49  |  High: 50+ runs. Counts, not within-season percentages.')

# 12: Hexbin
fig,ax=canvas()
hb=ax.hexbin(df.R,df.SR,gridsize=29,cmap='viridis',mincnt=1,linewidths=.15)
fig.colorbar(hb,ax=ax,label='Batting records per hexagon',shrink=.88)
ax.set(title='Density of Runs and Strike Rate',xlabel='Runs',ylabel='Strike rate')
finish(fig,ax,'12_hexbin_plot','Each hexagon represents the count of batting records in that region.')

print('\nALL 12 PLOTS EXPORTED IN PNG AND PDF FORMAT')
print('Location:',PLOTS)
print('Total files:',len(list(PLOTS.glob('*.png')))+len(list(PLOTS.glob('*.pdf'))))
