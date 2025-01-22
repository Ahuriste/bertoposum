import pandas as pd
import os
import numpy as np

metrics = ["rouge1", "rouge2", "rougeL"]

metric_models = {}
metric_baselines = {}

for metric in metrics:
    metric_models[metric] = pd.read_csv(f"{metric}.csv",index_col=0)
    metric_baselines[metric] = pd.read_csv(f"baselines_{metric}.csv",index_col=0)

for domain in metric_models ["rouge1"].index:
    print(domain.replace("_","\\_"), end=" & ")
    gout = []
    for metric in metrics:
        m = (metric_models[metric].loc[domain].values.tolist())
        b = (metric_baselines[metric].loc[domain].values.tolist())
        v = m+b
        mm = np.argmax(v)
        outl = [f"{val:0.2f}" for val in v]
        for i,o in enumerate(outl):
            if o == outl[mm]:
                outl[i] = "\\textbf{"+outl[mm]+"}"
        gout = gout + outl
    print(" & ".join(gout),end="")
    print ("\\\\")
