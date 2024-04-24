# System and Core Libraries
import pandas as pd
import numpy as np
from experiments_util import cols , pickle_paths_to_df


def build_pivot_df(df, dbname, dfcols):
    model_pfx_label = f'Model ({dbname.replace("_", " ")})'
    
    dfcols = [model_pfx_label] + dfcols
    
    df = df.rename(columns={'model_pfx' : model_pfx_label})
    
    rounding = {cols['fid']: 2, cols['target_epsilon']: 1, cols['sigma']: 3, cols['clipscore_consistency']: 2 , cols['clipscore']: 3 , cols['cmmd']: 3 , cols['density']: 2, cols['coverage']: 2, 'subsample_rate': 4}
    latex_sort = [model_pfx_label, cols['n_queries'], cols['target_epsilon'], cols['fid']]
    latex_groupby = [model_pfx_label, cols['n_queries'], cols['target_epsilon']]
    df.loc[df[cols['eps']] > df[cols['target_epsilon']], cols['n_queries']] = 0
    df.loc[df[cols['eps']] > df[cols['target_epsilon']], cols['target_epsilon']] = 'inf'
    
    df.loc[df[cols['sigma']] == 0, cols['n_queries']] = 0
    df.loc[df[cols['sigma']] == 0, cols['target_epsilon']] = 'inf'
    df.loc[(df['aggregate'] & (df[cols['target_epsilon']] == 'inf')), model_pfx_label] = 'RDM-adapt'
    df.loc[((df['aggregate'] == False) & (df[cols['target_epsilon']] == 'inf')), model_pfx_label] = 'RDM-fb'
    A = df[(df['retrieval_db_name'] == dbname)]#.round(rounding)
    
    
    idx = A.groupby(latex_groupby)[cols['fid']].idxmin().values
    A = df.loc[idx][dfcols].groupby(latex_groupby).min().sort_values(latex_sort).round(rounding)
    A = A.rename(columns={'subsample_rate': '$q$', 
                            cols['fid']: 'FID',
                            cols['clipscore_consistency']: 'CLIPScore',
                            cols['sigma']: '$\sigma$'})
    return A.round({cols['target_epsilon']: 2}).astype(str)

def process_latex(latex_str):
    replaces = [
        (".000000 ", " "),
        (cols['target_epsilon'], "$\epsilon$"),
        (cols['clipscore_consistency'], "CLIPScore"),
        ("& 150000 ", "& 150,000 "),
        ("& 100000 ", "& 100,000 "),
        ("& 10000 ", "& 10,000 "),
        ("& 1000 ", "& 1,000 "),
        ("& inf &", "& $\infty$ &"),
        ("No. Neighbors ($k$)", "$k$"),
        ("& Q. Interpolation ($\lambda$) &", "& $\lambda$ &"),
        ("Shutterstock 239M", " "),
        ("MS-COCO FB", " "),
        ("ImageNet FB", " "),
        ("CIFAR10", " "),
        ("DP-RDM-adapt-0.05", "DP-RDM-adapt"),
        ("NaN", "-"),
        ("(DP-RDM-adapt)", "DP-RDM-adapt"),
        ("(RDM-fb)", "RDM-fb"),
        ("(DP-RDM-concat)", "RDM-fb"),
        ("(DP-RDM-0.05.n)", "DP-RDM-adapt"),
        ("retrieval_db_name", "Retrieval"),
        ("validation_distribution", "Validation"),
        ("model_pfx", "Model"),
        ("shutterstock", "Shutterstock"),
        ("mscoco", "MS-COCO FB"),
        ("cifar10", "CIFAR-10"),
        ("Quality (FID)", "FID"),
        ("No. Queries ($T$)", "$T$"),
        ("Model (MS-COCO FB faceblurred)", "MS-COCO FB"),
        ("Model (Shutterstock)", "Shutterstock"),
        ("Model (cifar10)", "CIFAR-10")
    ]

    for r in replaces:
        latex_str = latex_str.replace(*r)
    return latex_str