# EECS_731-Project_3
In this project, I implemented three clustering model to cluster movies by genres and tags, and determine the top rating similiar movies. I used K-means, DBSCAN, and Agglomoerative clustering models, and each models will return different results. 

# Project Instruction
Blockbuster or art film?
1. Set up a data science project structure in a new git repository in your GitHub account
2. Download the one of the MovieLens datasets from https://grouplens.org/datasets/movielens/
3. Load the data set into panda data frames
4. Formulate one or two ideas on how the combination of ratings and tags by users helps the data set to establish additional value using exploratory data analysis
5. Build one or more clustering models to determine similar movies to recommend using the other ratings and tags of movies by other users as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

# Datasets
https://grouplens.org/datasets/movielens/ 

# Results
All three cluster model will successfully return similiar movies in same cluster, but the return result of three model are different, K-mean is the fastest model, and agglomoerative is the slowest. 

# References
https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas 

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.groupby.html 

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.combine.html 

https://numpy.org/doc/stable/reference/generated/numpy.fmax.html#numpy.fmax 

https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#concatenating-using-append 

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 

https://realpython.com/k-means-clustering-python/ 

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html 

https://www.thetopsites.net/article/50575374.shtml 

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html 

