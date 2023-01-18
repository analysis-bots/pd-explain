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

.. autofunction:: utils.to_explainable

The ``df`` parameter should be a dataframe

For example:

>>> import pandas as pd
>>> import pd_explain
>>> d = {'col1': [1, 2], 'col2': [3, 4]}
>>> df = pd.DataFrame(data=d)
>>> df = pd_explain.to_explainable(df)


