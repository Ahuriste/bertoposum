import pandas as pd
import os
import numpy as np
from math import ceil

metrics = ["rouge1", "rouge2", "rougeL"]

metric_models = {}
metric_baselines = {}

for metric in metrics:
    metric_models[metric] = pd.read_csv(f"{metric}.csv",index_col=0)
    metric_baselines[metric] = pd.read_csv(f"baselines_{metric}.csv",index_col=0)
coeff = ceil(2 * metric_models["rouge1"].loc["bluetooth","P"])
for domain in metric_models ["rouge1"].index:
    print(domain.replace("_","\\_"), end=" & ")
    gout = []
    for metric in metrics:
        m = (metric_models[metric].loc[domain].values.tolist())
        b = (metric_baselines[metric].loc[domain].values.tolist())
        m = [m_/coeff for m_ in m]
        v = m+b
        mm = np.argmax(v)
        outl = [f"{100*val:0.1f}" for val in v]
        for i,o in enumerate(outl):
            if o == outl[mm]:
                outl[i] = "\\textbf{"+outl[mm]+"}"
        gout = gout + outl
    print(" & ".join(gout),end="")
    print ("\\\\")
