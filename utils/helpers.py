import numpy as np 
import pandas as pd 
import joblib 
from config.paths_config import *

############## 1. GET_ANIME_FRAME ##############
def getAnimeFrame(anime,DF_PATH):
    df = pd.read_csv(DF_PATH,low_memory=True)
    if isinstance(anime,int):
        return df[df['MAL_ID']==anime]
    elif isinstance(anime,str):
        return df[df['eng_version'].str.contains(anime,case=False,na=False)]
    else:
        return pd.DataFrame()
    
############## 2. GET_SYNOPSIS ##############
def getSynopsis(anime,SYNOPSIS_DF):
    cols = ['MAL_ID','Name','Genres','sypnopsis']
    df = pd.read_csv(SYNOPSIS_DF,usecols=cols,low_memory=True)
    if isinstance(anime,int):
        return df[df['MAL_ID']==anime]['sypnopsis'].values[0]
    elif isinstance(anime,str):
        return df[df['Name'].str.contains(anime,case=False,na=False)]['sypnopsis'].values[0]
    else:
        return pd.DataFrame()
    
############## 3. FIND_SIMILAR_ANIME ##############
def find_similar_animes(name, ANIME_WEIGHTS_PATH, ANIME2ANIME_ENCODED, ANIME2ANIME_DECODED, DF_PATH, SYNOPSIS_DF, n=10, return_dist=False, neg=False):
    anime2anime_encoded = joblib.load(ANIME2ANIME_ENCODED)
    anime2anime_decoded = joblib.load(ANIME2ANIME_DECODED)
    index = getAnimeFrame(name, DF_PATH)['MAL_ID'].values[0]
    encoded_index = anime2anime_encoded.get(index)

    weights = joblib.load(ANIME_WEIGHTS_PATH)

    # Compute the similarity distance
    dists = np.dot(weights, weights[encoded_index])
    sorted_dists = np.argsort(dists)

    n = n + 1  # Add 1 to include the input anime (we will remove it later)

    if neg:
        closest = sorted_dists[:n]
    else:
        closest = sorted_dists[-n:]

    if return_dist:
        return dists, closest

    similarityArr = []
    for close in closest:
        decoded_id = anime2anime_decoded.get(close)
        synopsis = getSynopsis(decoded_id, SYNOPSIS_DF)
        anime_frame = getAnimeFrame(decoded_id, DF_PATH)
        anime_name = anime_frame['eng_version'].values[0]
        genre = anime_frame['Genres'].values[0]
        similarity = dists[close]
        similarityArr.append({
            'anime_id': decoded_id,
            'name': anime_name,
            'similarity': similarity,
            'genre': genre,
            'synopsis': synopsis,
        })

    Frame = pd.DataFrame(similarityArr).sort_values(by='similarity', ascending=False).reset_index(drop=True)

    # Remove the original anime and drop 'anime_id' column
    return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)


############## 4. FIND_SIMILAR_USERS ##############
def find_similar_user(user, USER_WEIGHTS_PATH, USER2USER_ENCODED, USER2USER_DECODED, n=10, return_dist=False, neg=False):
    user2user_encoded = joblib.load(USER2USER_ENCODED)
    user2user_decoded = joblib.load(USER2USER_DECODED)
    index = user 
    encoded_index = user2user_encoded.get(index)
    weights = joblib.load(USER_WEIGHTS_PATH) 

    dists = np.dot(weights,weights[encoded_index])
    sorted_dists = np.argsort(dists)

    n = n+1 # Add 1 to include the input user (we will remove it later)

    if neg:
        closest = sorted_dists[:n]
    else:
        closest = sorted_dists[-n:]
    
    if return_dist:
        return dists, closest
    
    similarityArr = []
    for close in closest:
        decoded_id = user2user_decoded.get(close)
        similarity = dists[close]
        similarityArr.append({
            'user_id': decoded_id,
            'similarity': similarity
        })
    Frame = pd.DataFrame(similarityArr).sort_values(by='similarity', ascending=False).reset_index(drop=True)
    return Frame[Frame['user_id'] != index]


############## 5. GET_USER_PREFERENCES ##############
def get_user_preferences(user_id, RATING_DF, DF_PATH):
    rating_df = pd.read_csv(RATING_DF,low_memory=True)
    df = pd.read_csv(DF_PATH,low_memory=True)
    animes_watched_by_user = rating_df[rating_df['user_id'] == user_id]
    user_rating_percentile = np.percentile(animes_watched_by_user['rating'], 75)
    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user['rating'] >= user_rating_percentile]
    top_animes_user = (
        animes_watched_by_user.sort_values(by='rating', ascending=False)['anime_id'].values
    )

    top_animes_df = df[df['MAL_ID'].isin(top_animes_user)]
    top_animes_df = top_animes_df[['eng_version', 'Genres']]
    return top_animes_df



############## 6. GET_USER_RECOMMENDATIONS ##############
def get_user_recommendations(similar_users , user_pref ,DF_PATH , SYNOPSIS_DF, RATING_DF, n=10):
    recommended_animes = []
    anime_list = []

    for user_id in similar_users.user_id.values:
        pref_list = get_user_preferences(int(user_id) , RATING_DF, DF_PATH)

        pref_list = pref_list[~pref_list.eng_version.isin(user_pref.eng_version.values)]

        if not pref_list.empty:
            anime_list.append(pref_list.eng_version.values)

    if anime_list:
            anime_list = pd.DataFrame(anime_list)

            sorted_list = pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)

            for i,anime_name in enumerate(sorted_list.index):
                n_user_pref = sorted_list[sorted_list.index == anime_name].values[0][0]

                if isinstance(anime_name,str):
                    frame = getAnimeFrame(anime_name,DF_PATH)
                    anime_id = frame.MAL_ID.values[0]
                    genre = frame.Genres.values[0]
                    synopsis = getSynopsis(int(anime_id),SYNOPSIS_DF)

                    recommended_animes.append({
                        "n" : n_user_pref,
                        "anime_name" : anime_name,
                        "Genres" : genre,
                        "Synopsis": synopsis
                    })
    return pd.DataFrame(recommended_animes).head(n)