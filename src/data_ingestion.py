import os 
import pandas as pd 
from google.cloud import storage 
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIR, CONFIG_PATH
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_names = self.config['bucket_file_names']

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data Ingestion started with {self.bucket_name} and {self.file_names}")
    
    def download_csv_from_gcp(self):
        """
        Dowload csv files from GCP buckets
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)
                logger.info(f"Downloaded smaller file {file_name} successfully")
        except Exception as e:
            logger.error(f"Error in downloading files from GCP: {e}")
            raise CustomException("Failed to download the csv files from GCP", e)
        
    def run(self):
        """
        Run the data ingestion process
        """
        try:
            logger.info('Data ingestion process started')
            self.download_csv_from_gcp()
            logger.info(f'Data Ingestion completed successfully')
        except Exception as e:
            logger.error(f"Error in data ingestion")
            raise CustomException("Data ingestion failed", e)
        finally:
            logger.info('Data ingestion process finished')

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
