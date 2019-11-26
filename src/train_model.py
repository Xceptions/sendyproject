import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")


class Train_LGB_Model():
    """
    Class trains models per meter type
    """
    def __init__(self):
        self.category_cols = ['Personal or Business', 'Pickup - Day of Month']
        self.models1 = []
        self.models2 = []
        self.models3 = []
        self.models4 = []
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file = 'df_train.csv'
        self.read_data()
        
    def read_data(self):
        data_path = os.path.join(self.repo_path, 'data', 'selected_features', self.file)
        self.df = pd.read_csv(data_path)
        for i in self.category_cols:
            self.df[i] = self.df[i].astype('category')
        self.train_plat1()

    def fetch_train_data_batch(self, data, platform_type):
        to_train = data.loc[data['Platform Type'] == platform_type]
        to_train_target = to_train['Time from Pickup to Arrival'].values
        to_train_data = to_train.drop(['Order No',
                                       'Time from Pickup to Arrival',
                                       'Unnamed: 0',
                                       'Platform Type'], axis='columns')
        del to_train
        gc.collect()
        return to_train_data, to_train_target

    def fit_lgbm(self, train, val, devices=(-1), seed=None, cat_features=None, num_rounds=1500, lr=0.05):
        """Function to train the Light GBM model"""
        X_tt, y_tt = train
        X_vl, y_vl = val
        params = {
            'objective':'regression',
            'boosting_type':'gbdt',
            'learning_rate':lr,
            'num_leaves': 2**8,
            'max_depth':20,
            'n_estimators':5000,
            'max_bin':255,
            'num_leaves': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.3,
            'verbose':-1,
            'seed': 42,
            "metric": 'rmse',
            'early_stopping_rounds':100
        }
        device = devices
        if device == -1:
            pass # use cpu
        else:
            params.update({'device': 'gpu', 'gpu_device_id': device})
        params["seed"] = seed
        d_train = lgb.Dataset(X_tt, label=y_tt, categorical_feature=cat_features)
        d_valid = lgb.Dataset(X_vl, label=y_vl, categorical_feature=cat_features)
        watchlist = [d_train, d_valid]
        print("training LGB: ")
        model = lgb.train(params,
                        train_set=d_train,
                        num_boost_round=num_rounds,
                        valid_sets=watchlist,
                        verbose_eval=100)
        print("best score", model.best_score)
        return model

    def train_plat1(self):
        print("Training for platform 1")
        platform_type = 1

        X_train, y_train = self.fetch_train_data_batch(self.df, platform_type)
        gc.collect()

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in self.category_cols]
        kfcv = KFold(n_splits=4)

        for train_idx, val_idx in kfcv.split(X_train):
            train_data = X_train.iloc[train_idx], y_train[train_idx]
            val_data = X_train.iloc[val_idx], y_train[val_idx]
            print("Training: ", len(train_idx), " and validating: ", len(val_idx))
            model = self.fit_lgbm(train_data,val_data,cat_features=cat_features,num_rounds=1000,lr=0.05)
            self.models1.append(model)
            gc.collect()
        print("training platform 1 has ended")
        self.train_plat2()

    def train_plat2(self):
        print("Training for platform 2")
        platform_type = 2

        X_train, y_train = self.fetch_train_data_batch(self.df, platform_type)
        gc.collect()

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in self.category_cols]
        kfcv = KFold(n_splits=4)

        for train_idx, val_idx in kfcv.split(X_train):
            train_data = X_train.iloc[train_idx], y_train[train_idx]
            val_data = X_train.iloc[val_idx], y_train[val_idx]
            print("Training: ", len(train_idx), " and validating: ", len(val_idx))
            model = self.fit_lgbm(train_data,val_data,cat_features=cat_features,num_rounds=1000,lr=0.05)
            self.models2.append(model)
            gc.collect()
        print("training platform 2 has ended")
        self.train_plat3()
    
    def train_plat3(self):
        print("Training for platform 3")
        platform_type = 3

        X_train, y_train = self.fetch_train_data_batch(self.df, platform_type)
        gc.collect()

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in self.category_cols]
        kfcv = KFold(n_splits=4)

        for train_idx, val_idx in kfcv.split(X_train):
            train_data = X_train.iloc[train_idx], y_train[train_idx]
            val_data = X_train.iloc[val_idx], y_train[val_idx]
            print("Training: ", len(train_idx), " and validating: ", len(val_idx))
            model = self.fit_lgbm(train_data,val_data,cat_features=cat_features,num_rounds=1000,lr=0.05)
            self.models3.append(model)
            gc.collect()
        print("training platform 3 has ended")
        self.train_plat4()

    def train_plat4(self):
        print("Training for platform 4")
        platform_type = 4

        X_train, y_train = self.fetch_train_data_batch(self.df, platform_type)
        gc.collect()

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in self.category_cols]
        kfcv = KFold(n_splits=4)

        for train_idx, val_idx in kfcv.split(X_train):
            train_data = X_train.iloc[train_idx], y_train[train_idx]
            val_data = X_train.iloc[val_idx], y_train[val_idx]
            print("Training: ", len(train_idx), " and validating: ", len(val_idx))
            model = self.fit_lgbm(train_data,val_data,cat_features=cat_features,num_rounds=1000,lr=0.05)
            self.models4.append(model)
            gc.collect()
        print("training platform 4 has ended")
        self.savemodels()

    def savemodels(self):
        self.lgb_models = [self.models1, self.models2, self.models3, self.models4]
        data_path = os.path.join(self.repo_path, 'data', 'models')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, 'lgb_models.pkl')
        pickle.dump(self.lgb_models, open(combined_path_test, 'wb'))



if __name__ == "__main__":
    Train_LGB_Model()