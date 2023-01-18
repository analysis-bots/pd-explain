Usage
=====

.. _installation:

Installation
------------

To use pd_explain, first install it using pip:

.. code-block:: console

   pip install pd_explain


import pd_explain
----------------
you have to import pd_explain after importing pandas to use explanations

.. code-block:: python

    import pandas as pd
    import pd_explain

Create dataframe
----------------

To read dataframe from csv (or any file use pd read functions),
you can use the ``pd.red_csv()`` function:

This pandas read function already returns an explainable dataframe
For example:

>>> import pandas as pd
>>> import pd_explain
>>> pd.read_csv(r'Dataset/spotify_all')

Alternatively you can create your own dataframe
and convert it using ``pd_explain.to_explainable``

.. autoclass:: pd_explain.ExpDataFrame
     .. method:: explain(self, schema: dict = None, attributes: List = None, top_k: int = 1,
                figs_in_row: int = 2, show_scores: bool = False, title: str = None)

The ``df`` parameter should be a dataframe

For example:

>>> import pandas as pd
>>> import pd_explain
>>> d = {'col1': [1, 2], 'col2': [3, 4]}
>>> df = pd.DataFrame(data=d)
>>> df = pd_explain.to_explainable(df)


