"""Checks that analysis/qc.py's ported filters give identical flags to OpenOA's own
openoa.utils.filters on the same synthetic series. Run once under each venv and compare:

    .venv-openoa/bin/python -m operational_assessment.check_filter_parity oa   > /tmp/oa.json
    .venv/bin/python        -m operational_assessment.check_filter_parity port > /tmp/port.json
    diff /tmp/oa.json /tmp/port.json && echo identical
"""
import numpy as np, pandas as pd, json, sys
rng=np.random.default_rng(0)
v=np.round(rng.normal(7,2,3000),1); v[100:106]=5.5; v[2000:2003]=9.9
p=np.clip(rng.normal(1000,400,3000),0,2050); w=7+p/400+rng.normal(0,0.5,3000); w[::97]+=6
df=pd.DataFrame({"ws":v,"p":p,"w":w})
if sys.argv[1] == "oa":
    from openoa.utils.filters import unresponsive_flag, bin_filter
    a=unresponsive_flag(df,3,col=["ws"])["ws"]
    b=bin_filter(df["p"],df["w"],bin_width=0.06*2050,threshold=2,center_type="median",bin_min=0.01*2050,bin_max=0.9*2050,threshold_type="std",direction="all")
else:
    from analysis.qc import unresponsive_flag, bin_filter
    a=unresponsive_flag(df["ws"],3)
    b=bin_filter(df["p"],df["w"],bin_width=0.06*2050,threshold=2,bin_min=0.01*2050,bin_max=0.9*2050)
print(json.dumps({"unresp":a.astype(int).tolist(),"bin":b.astype(int).tolist()}))
