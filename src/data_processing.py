import os 
import numpy as np 
import pandas as pd 
import joblib 
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.preprocessing import MinMaxScaler
from config.paths_config import *
import sys 

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, input_file,output_dir):
        self.input = input_file
        self.output_dir = output_dir
        self.rating_df = None
        self.anime_df = None 
        self.synopsis_df = None
        self.x_train_array = None
        self.x_test_array = None
        self.y_train = None 
        self.y_test = None 

        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Data Processing Initialized")
    
    def load_data(self,usecols):
        try:
            self.rating_df = pd.read_csv(self.input, low_memory=True, usecols=usecols)
            logger.info(f"Data loaded successfully from {self.input}")
        except Exception as e:
            raise CustomException("Failed to load data",e, sys)
    
    def filter_users(self, min_ratings=100):
        try:
            n_ratings = self.rating_df['user_id'].value_counts()
            self.rating_df = self.rating_df[self.rating_df['user_id'].isin(n_ratings[n_ratings >= min_ratings].index)]
            self.rating_df.drop_duplicates(inplace=True)
            logger.info(f"Filtered users with at least {min_ratings} ratings")
        except Exception as e:
            raise CustomException("Failed to filter data",sys)
    
    def scale_ratings(self):
        try:
            scaler = MinMaxScaler(feature_range=(0,1))
            self.rating_df['rating'] = scaler.fit_transform(self.rating_df[['rating']])
            logger.info("Ratings scaled to range [0, 1]")
        except Exception as e:
            raise CustomException("Failed to scale the data",sys)
    
    def encode_decode_data(self):
        try:
            ### For users 
            user_ids = self.rating_df['user_id'].unique().tolist()
            self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            self.user2user_decoded = {i: x for i, x in enumerate(user_ids)}
            self.rating_df['user'] = self.rating_df['user_id'].map(self.user2user_encoded)
            logger.info("User IDs encoded and decoded successfully")
            ### For anime 
            anime_ids = self.rating_df['anime_id'].unique().tolist()
            self.anime2anime_encoded = {x : i for i, x in enumerate(anime_ids)}
            self.anime2anime_decoded = {i : x for i, x in enumerate(anime_ids)}
            self.rating_df['anime'] = self.rating_df['anime_id'].map(self.anime2anime_encoded)
            logger.info("Anime IDs encoded and Decoded successfully")
        except Exception as e:
            raise CustomException("Failed to scale the data",sys)
        
    def split_data(self, test_size=0.01):
        try:
            x = self.rating_df[['user', 'anime']]
            y = self.rating_df['rating']
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=42)
            x_train = x_train.to_numpy()
            x_test = x_test.to_numpy()
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            self.x_train_array = [x_train[:,0],x_train[:,1]]
            self.x_test_array = [x_test[:,0],x_test[:,1]]
            self.y_train = y_train
            self.y_test = y_test
            logger.info("Data split into training and testing sets successfully")
        except Exception as e:
            raise CustomException("Failed to split the data",sys)
        
    def save_artifacts(self):
        try:
            artifacts = {
                "user2user_encoded" : self.user2user_encoded,
                "user2user_decoded" : self.user2user_decoded,
                "anime2anime_encoded" : self.anime2anime_encoded,
                "anime2anime_decoded" : self.anime2anime_decoded,
                "x_train_array" : self.x_train_array,
                "x_test_array" : self.x_test_array,
                "y_train" : self.y_train,
                "y_test" : self.y_test
            }
            for name, data in artifacts.items():
                joblib.dump(data, os.path.join(self.output_dir, f"{name}.pkl"))
                logger.info(f"{name} - Artifacts saved successfully")
            self.rating_df.to_csv(RATING_DF, index=False)
            logger.info("Rating DataFrame saved successfully")
        except Exception as e:
            raise CustomException("Failed to save artifacts", sys)
        
    def process_anime_data(self):
        try:
            self.anime_df = pd.read_csv(ANIME_CSV, low_memory=True)
            self.anime_df.replace('Unknown',np.nan,inplace=True)
            cols = ['MAL_ID','Name','Genres','sypnopsis']
            self.synopsis_df = pd.read_csv(SYNOPSIS_CSV, usecols = cols,low_memory=True)

            def getAnimeName(anime_id):
                try:
                    anime_row = self.anime_df[self.anime_df['MAL_ID'] == anime_id]
                    if not anime_row.empty:
                        english_name = anime_row['English name'].values[0]
                        original_name = anime_row['Name'].values[0]
                        
                        if pd.notna(english_name):
                            return english_name
                        else:
                            return original_name
                    else:
                        return "Anime ID not found"
                except Exception as e:
                    print("Error in getting anime name for ID:", anime_id, "|", e)
                    return None
            
            self.anime_df['eng_version'] = self.anime_df['MAL_ID'].apply(getAnimeName)
            self.anime_df.sort_values(by='Score',inplace=True, ascending=False, kind='quicksort', na_position='last')
            self.anime_df = self.anime_df[['MAL_ID','eng_version','Score','Genres','Episodes','Type','Premiered','Members']]

            self.anime_df.to_csv(DF_PATH,index=False)
            self.synopsis_df.to_csv(SYNOPSIS_DF, index=False)
            logger.info("Anime data processed and saved successfully")
        except Exception as e:
            raise CustomException("Failed to process anime data", sys)
    
    def run(self):
        try:
            self.load_data(usecols=['user_id', 'anime_id', 'rating'])
            self.filter_users()
            self.scale_ratings()
            self.encode_decode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()
            logger.info("Data processing completed successfully")
        except Exception as e:
            raise CustomException("Data processing failed", sys)
if __name__ == "__main__":
    data_processor = DataProcessor(input_file=ANIMELIST_CSV, output_dir=PROCESSED_DIR)
    data_processor.run()


