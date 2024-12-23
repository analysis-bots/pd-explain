The recommender engine uses dynamic loading to load the various recommenders.\
So long as a recommender satisfies three conditions, it can be loaded automatically by the engine:
1. The recommender must be a subclass of the `RecommenderBase` class and implements its abstract methods.
2. The recommender must be in the `recommenders` package.
3. The recommender must be in a file that ends in `_recommender.py`, and its class name must end in `Recommender`.

So long as the above conditions are satisfied, the recommender engine will automatically load the recommender and make it available for use, without
any additional configuration.\
If you do not abide by these conditions, you will need to write code that loads the recommender manually.\