
# PD-EXPLAIN

PD-EXPLAIN is a Python library that wraps Pandas, allowing users to obtain multiple type of query explanations over Pandas DataFrames. 
PD-EXPLAIN is under active development, currently featuring interestingness based explanations, those are deviation-based explanations (for filter, join, and set operations) and explanations for high-variance group-by-and-aggregate operations. Both explainers utilizes the [FEDEX](https://www.vldb.org/pvldb/vol15/p3854-gilad.pdf) system.

The system also supports aggregate outlier explanations, based on the [SCORPION](https://sirrice.github.io/files/papers/scorpion-vldb13.pdf) systems
, and will soon fully support Boolean-query explanations based on [this paper](https://arxiv.org/abs/2112.08874).

PD-EXPLAIN was demonstrated at VLDB '24.




## Installation

Install pd-explain with pip or by git ssh

```bash
  pip install pd-explain
  
  pip install git+ssh://git@github.com/analysis-bots/pd-explain.git
```

For cloning this project use
```bash
git clone git@github.com:analysis-bots/pd-explain.git

cd pd_explain

pip install -r requirements.txt
```

## Demo

[Demo Spotify example](https://github.com/analysis-bots/pd-explain/blob/main/src/demo.ipynb)


![Demo Spotify example notebook - click to view](./assets/pdexplain_demo.gif)

## Documentation

[Documentation](https://pd-explain-documentation.readthedocs.io/en/main/index.html)


## Citation Information
TBD



## Authors

- [@Eden Isakov](https://github.com/edenIsakov)
- [@Itay Elyashiv](https://github.com/ItayELY)
- [@analysis-bots](https://github.com/analysis-bots)

