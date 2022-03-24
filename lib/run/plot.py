


from ..base import *


def set_theme(size=(3.5,3.5)):
    sns.set(
        palette='deep',
        style='white',
        rc={
            'figure.figsize': size,
            'axes.linewidth': 0.3,
            'axes.grid': True,
            'grid.color': 'white',
            'grid.linewidth': 0.3,
                'text.color': 'white',
                "figure.facecolor":  '#161A20',
                "axes.facecolor":  '#161A20',
                "savefig.facecolor":  '#161A20',
            "ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"
        }
    )
    

class Plot:
    def __init__(s, aa, start, stop):
        s.aa = aa
        s.start = start
        s.stop = stop        
        
    def __call__(s, col):
        df = pd.DataFrame(index=range(1000))
        for a in s.aa:
            p = a.log_dir
            p /= a.exp_name
            p /= 'version_0'
            p /= 'metrics.csv'
            try:
                x = load(p)
                df[a.exp_name] = x[col]
            except:
                print(f'{s.aa.index(a)+1}/{len(s.aa)} not found')
        sns.scatterplot(
            data=df.iloc[s.start:s.stop]).legend(
                bbox_to_anchor=(1.01,1),
                loc='upper left',
        );
        sns.scatterplot(
            x=[s.stop],
            y=[df.iloc[s.start:s.stop][a.exp_name].max()]
        ).legend(
                bbox_to_anchor=(1.01,1),
                loc='upper left',
        );