import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.cluster import DBSCAN

link = pd.read_csv('Data/links.csv') # this data is different ID of movies, will not use this data
movie = pd.read_csv('Data/movies.csv')
rating = pd.read_csv('Data/ratings.csv')
tag = pd.read_csv('Data/tags.csv')




# remove unused features
rating = rating.drop(['userId', 'timestamp'], axis=1)
tag = tag.drop(['userId', 'timestamp'], axis=1)

avg_rating = rating.groupby('movieId',as_index=False)['rating'].mean()

movie_genres = pd.merge(movie, avg_rating, how='outer',on='movieId').fillna(0)
splitGenres = movie_genres['genres'].astype('str').str.split(pat='|',expand=True)
genres = pd.get_dummies(splitGenres[0])
for i in range(1,len(splitGenres.columns)):
    temp = pd.get_dummies(splitGenres[i])
    genres = genres.combine(temp,np.fmax)

movie_genres = pd.concat([movie_genres, genres], axis=1, sort=False).drop('genres', axis=1)
movie_genres.to_csv('Data/movie_by_genres.csv', index=False)

tags = pd.get_dummies(tag['tag'])
withMovieID = pd.concat([tag, tags], axis=1, sort=False).drop('tag', axis=1)
movie_tags = withMovieID.groupby('movieId',as_index=False)[tags.columns].sum()
movie_tags = pd.merge(movie_tags, avg_rating, how='outer', on='movieId').fillna(0)
movie_tags = pd.merge(movie_tags, movie.drop('genres',axis=1), on='movieId')
movie_tags.to_csv('Data/movie_by_tags.csv', index=False)

feature_genres = movie_genres.drop(['movieId', 'title', 'rating'],axis=1)
feature_tags = movie_tags.drop(['movieId','title', 'rating'],axis=1)

def get_numClusters(features):
    sse = []
    for k in range(1,50,5):
        kmeans = KMeans(n_clusters=k).fit(features)
        sse.append(kmeans.inertia_)
        
    plt.plot(range(1,50,5), sse,'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    
    kl = KneeLocator(range(1,50,5), sse, curve="convex", direction="decreasing")
    return kl.elbow


def similiarMoviesGenres(Title, number_movies,cluster):
    if cluster == 'KMeans':
        C = 'Cluster'
        numClusters = get_numClusters(feature_genres)
        cluster_label = KMeans(n_clusters=numClusters).fit_predict(feature_genres)
    elif cluster == 'DBSCAN':
        C = 'Cluster_DBSCAN'
        cluster_label = DBSCAN(eps=0.5).fit_predict(feature_genres)
    elif cluster == 'agg':
        C = 'Cluster_agg'
        numClusters = get_numClusters(feature_genres)
        cluster_label = AgglomerativeClustering(n_clusters=numClusters).fit_predict(feature_genres)

    movie_genres[C] = cluster_label
    row_index = movie_genres.index[movie_genres['title'] == Title]
    cluster_number = movie_genres.loc[row_index][C].values[0]
    moviesByGenres = movie_genres.loc[movie_genres[C] == cluster_number]
    moviesByGenres = moviesByGenres.sort_values(by=['rating'],ascending=False)  # sort the movies by rating
    top_similiar_movies = pd.DataFrame(moviesByGenres['title'].values).head(number_movies)
    return top_similiar_movies

def similiarMoviesTags(Title, number_movies,cluster):
    if cluster == 'KMeans':
        C = 'Cluster'
        numClusters = get_numClusters(feature_tags)
        cluster_label = KMeans(n_clusters=numClusters).fit_predict(feature_tags)
    elif cluster == 'DBSCAN':
        C = 'Cluster_DBSCAN'
        cluster_label = DBSCAN(eps=0.5).fit_predict(feature_tags)
    elif cluster == 'agg':
        C = 'Cluster_agg'
        numClusters = get_numClusters(feature_tags)
        cluster_label = AgglomerativeClustering(n_clusters=numClusters).fit_predict(feature_tags)

    movie_tags[C] = cluster_label
    row_index = movie_tags.index[movie_tags['title'] == Title]
    cluster_number = movie_tags.loc[row_index][C].values[0]
    moviesByTags = movie_tags.loc[movie_tags[C] == cluster_number]
    moviesByTags = moviesByTags.sort_values(by=['rating'],ascending=False)  # sort the movies by rating
    top_similiar_movies = pd.DataFrame(moviesByTags['title'].values).head(number_movies)
    return top_similiar_movies

similiarMoviesGenres('Toy Story (1995)',10,'KMeans')
similiarMoviesGenres('Toy Story (1995)',10,'DBSCAN')
similiarMoviesGenres('Toy Story (1995)',10,'agg')

similiarMoviesTags('Toy Story (1995)',10,'KMeans')
similiarMoviesTags('Toy Story (1995)',10,'DBSCAN')
similiarMoviesTags('Toy Story (1995)',10,'agg')



















