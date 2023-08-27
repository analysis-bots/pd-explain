
# PD-EXPLAIN

PD-EXPLAIN is a Python library that wraps Pandas, allowing users to obtain explanations and additional insights on their analytical operations.
PD-EXPLAIN is under active development, currently featuring the [FEDEX](https://www.vldb.org/pvldb/vol15/p3854-gilad.pdf) system, which detects interesting segments in dataframes resulted from filter, join, union and group-by operations. 





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

[Demo Spotify example](https://github.com/analysis-bots/pd-explain/blob/main/Examples/Notebooks/PD-explain%20DEMO.ipynb)


![Demo Spotify example notebook - click to view](./assets/pdexplain_demo.gif)

## Documentation

[Documentation](https://stirring-medovik-ba9b36.netlify.app/src/pd_explain.html)


## Citation Information


- [ExplainED: Explanations for EDA Notebooks](http://www.vldb.org/pvldb/vol13/p2917-deutch.pdf)

## Authors

- [@Eden Isakov](https://github.com/edenIsakov)
- [@analysis-bots](https://github.com/analysis-bots)

