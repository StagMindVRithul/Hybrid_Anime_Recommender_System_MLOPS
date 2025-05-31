import comet_ml
import joblib 
import numpy as np 
import os 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src.logger import get_logger 
from src.custom_exception import CustomException
from src.base_model import BaseModel
from config.paths_config import * 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("COMET_API_KEY")

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, data_path):
        self.data_path = data_path
        self.experiement = comet_ml.Experiment(
            api_key=api_key,
            project_name="anime-recommender-system",
            workspace="stagmindvrithul"
        )
        logger.info("Model Training & Comet ML Experimental Tracking Initialized!!!!")
    
    def load_data(self):
        try:
            x_train_array = joblib.load(X_TRAIN_ARRAY)
            x_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)
            logger.info("Data loaded successfully for model training")
            return x_train_array, x_test_array, y_train, y_test
        except Exception as e:
            logger.error("Error loading data for model training")
            raise CustomException("Failed to load data for model training", e)
    
    def train_model(self):
        try:
            x_train_array, x_test_array, y_train, y_test = self.load_data()
            n_users = len(joblib.load(USER2USER_ENCODED))
            n_animes = len(joblib.load(ANIME2ANIME_ENCODED))
            base_model = BaseModel(config_path = CONFIG_PATH)
            model = base_model.RecommenderNet(n_users = n_users, n_animes = n_animes)
            logger.info("Model created successfully")

            start_lr = 1e-6
            max_lr = 1e-4
            min_lr = 5e-6
            batch_size = 100000
            ramup_epochs = 3
            sustain_epochs = 2
            exp_decay = 0.05

            def lrfn(epoch):
                if epoch < ramup_epochs:
                    lr = (max_lr - start_lr) / ramup_epochs * epoch + start_lr
                elif epoch < ramup_epochs + sustain_epochs:
                    lr = max_lr
                else:
                    lr = (max_lr - min_lr) * exp_decay ** (epoch - ramup_epochs - sustain_epochs) + min_lr
                return lr
            
            lr_callback = LearningRateScheduler(lambda epoch:lrfn(epoch), verbose=0)
            checkpoint_callback = ModelCheckpoint(
                filepath=CHECKPOINT_FILE_PATH,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min'
            )

            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                restore_best_weights=True
            )

            my_callbacks = [lr_callback, checkpoint_callback, early_stopping_callback]

            os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH),exist_ok=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            try:
                history = model.fit(
                    x = x_train_array,
                    y = y_train,
                    batch_size = batch_size,
                    epochs = 20, 
                    validation_data = (x_test_array, y_test),
                    callbacks = my_callbacks,
                    verbose = 1
                )
                model.load_weights(CHECKPOINT_FILE_PATH)
                logger.info("Model trained completed.....")

                for epoch in range(len(history.history['loss'])):
                    train_loss = history.history['loss'][epoch]
                    val_loss = history.history['val_loss'][epoch]
                    self.experiement.log_metric('train_loss',train_loss,step=epoch)
                    self.experiement.log_metric('val_loss',val_loss,step=epoch)

            except Exception as e:
                logger.error("Error during model training")
                raise CustomException("Failed to train the model", e)
            self.save_model_weights(model)
            logger.info("Model weights saved successfully")
        except Exception as e:
            logger.error(str(e))
            raise CustomException("Failed to train the model", e)
    
    def extract_weights(self,layer_name,model):
        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1).reshape(-1,1)
            logger.info(f"Extracting weights for {layer_name} layer")
            return weights
        except Exception as e:
            logger.error(f"Error extracting weights for {layer_name} layer")
            raise CustomException(f"Failed to extract weights for {layer_name} layer", e)

    def save_model_weights(self,model):
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved at {MODEL_PATH}")
            anime_weights = self.extract_weights('anime_embedding', model)
            user_weights = self.extract_weights('user_embedding', model)
            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)
            self.experiement.log_asset(MODEL_PATH)
            self.experiement.log_asset(ANIME_WEIGHTS_PATH)
            self.experiement.log_asset(USER_WEIGHTS_PATH)

            logger.info(f"User and Anime weights saved at {USER_WEIGHTS_PATH} and {ANIME_WEIGHTS_PATH}")
        except Exception as e:
            logger.error(str(e))
            raise CustomException("Failed to save model weights", e)
        
if __name__ == "__main__":
    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()
    
            

