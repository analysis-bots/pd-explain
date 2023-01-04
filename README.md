
# Explain pd

A library  that create explation for dataframes manipulations, using pandas dataframes and explations based on [ExplainED: Explanations for EDA Notebooks
](https://github.com/TAU-DB/ExplainED)



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
## Usage
You can use explain dataframe like this

```python
import pandas as pd
import pd_explain

pd.read_...() # All read functions create explainable dataframe

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df = pd_explain.to_explainable(df)
```


## Demo

[Full example Notebook](https://github.com/analysis-bots/pd-explain/blob/main/Examples/Notebooks/Bank%20Churners%20Pitch.ipynb)

[Demo Spotify example](https://github.com/analysis-bots/pd-explain/blob/main/Examples/Notebooks/Demo.ipynb)

![Demo Spotify example](./assets/explain_demo.gif)

## Documentation

[Documentation](https://stirring-medovik-ba9b36.netlify.app/src/pd_explain.html)


## Articles

- [ExplainED: Explanations for EDA Notebooks](http://www.vldb.org/pvldb/vol13/p2917-deutch.pdf)

## Authors

- [@Eden Isakov](https://github.com/edenIsakov)
- [@analysis-bots](https://github.com/analysis-bots)

