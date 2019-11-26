import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import os, sys, gc, warnings
import pickle
warnings.filterwarnings("ignore")


class MakePredictions():
    def __init__(self):
        self.all_row_ids = []
        self.all_row_order_no = []
        self.all_predictions = []
        self.all_platforms = []
        self.category_cols = ['Personal or Business', 'Pickup - Day of Month']
        self.model_preds = pd.DataFrame({})
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file = 'df_test.csv'
        self.read_data()
        
    def read_data(self):
        data_path = os.path.join(self.repo_path, 'data', 'selected_features', self.file)
        self.df = pd.read_csv(data_path)
        for i in self.category_cols:
            self.df[i] = self.df[i].astype('category')
        pkl_path = os.path.join(self.repo_path, 'data', 'models', 'lgb_models.pkl')
        self.lgb_models = pickle.load(open(pkl_path, 'rb'))[0]
        self.predict_test()

    def fetch_test_data_batch(self, test_data, platform_type):
        to_test_data = test_data.loc[test_data['Platform Type'] == platform_type]
        return to_test_data

    def make_preds(self, X_pred, models):
        X_pred_idx = X_pred['Unnamed: 0'] # returning the row_id in order to preserve order
        X_pred_order_no = X_pred['Order No']
        X_pred_platform_type = X_pred['Platform Type']
        X_pred = X_pred.drop(['Order No',
                              'Unnamed: 0',
                              'Platform Type',
                              'Time from Pickup to Arrival'], axis='columns')
        final_predictions_total = np.zeros(X_pred.shape[0])
        for i in range(len(self.lgb_models)):
            estimator = self.lgb_models[i]
            predictions = estimator.predict(X_pred, num_iteration=estimator.best_iteration)
            final_predictions_total += predictions
        # Now we are done with the predictions, we'll take the average of the predictions
        final_predictions_total /= len(self.lgb_models)
        return X_pred_idx, X_pred_order_no, X_pred_platform_type, final_predictions_total

    def predict_test(self):
        for i in [1, 2, 3, 4]:
            platform_type = i
            use_model = self.lgb_models[i - 1]

            X_valid = self.fetch_test_data_batch(self.df, platform_type)
            print("Starting predictions for Platform type " + str(platform_type))
            y_valid_id, y_valid_order_no, y_platform, y_valid = self.make_preds(X_valid, use_model)
            
            print("Appending predictions...")
            self.all_row_ids.append(y_valid_id)
            self.all_row_order_no.append(y_valid_order_no)
            self.all_platforms.append(y_platform)
            self.all_predictions.append(y_valid)
            print("Finished predictions for platform " + str(platform_type))
        self.prepare_to_save()

        
    def prepare_to_save(self):
        self.all_row_ids_flat = [ids for sublist in self.all_row_ids for ids in sublist]
        self.all_row_order_no_flat = [orderno for ordersublist in self.all_row_order_no for orderno in ordersublist]
        self.all_platforms_flat = [platformtype for platformsublist in self.all_platforms for platformtype in platformsublist]
        self.all_predictions_flat = [preds for predsublist in self.all_predictions for preds in predsublist]
        self.model_preds['Index col'] = self.all_row_ids_flat
        self.model_preds['Order No'] = self.all_row_order_no_flat
        self.model_preds['Platform Type'] = self.all_platforms_flat
        self.model_preds['Time from Pickup to Arrival'] = self.all_predictions_flat
        self.model_preds = self.model_preds.sort_values(by=['Index col'])
        self.savepredictions()

    def savepredictions(self):
        data_path = os.path.join(self.repo_path, 'data', 'model_predictions')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, 'lgb_predictions.csv')
        self.model_preds.to_csv(combined_path_test, index=False)


if __name__ == "__main__":
    MakePredictions()