import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pickle


class Generate_base_features():
    """
    Class generates features that are dataset implicitly dependent.
    These features are generated for both the train and test set.
    """
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for self.file in ['df_train.csv', 'df_test.csv']:
            self.read_data()
        
    def read_data(self):
        to_parse = ['Placement - Time', 'Confirmation - Time', 'Pickup - Time']
        data_path = os.path.join(self.repo_path, 'data', 'raw', self.file)
        self.df = pd.read_csv(data_path, parse_dates=to_parse)
        self.split_time()
        
    def split_time(self):
        self.df['placement_hour'] = self.df['Placement - Time'].dt.hour
        self.df['confirmation_hour'] = self.df['Confirmation - Time'].dt.hour
        self.convert_to_24hr()

    def convert_to_24hr(self):
        self.df['Placement - Time'] = self.df['Placement - Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        self.df['Confirmation - Time'] = self.df['Confirmation - Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        self.df['Pickup - Time'] = self.df['Pickup - Time'].apply(lambda x: x.strftime('%H:%M:%S'))
        self.fill_weather_nan()
        
    def fill_weather_nan(self):
        self.df['Temperature'] = self.df['Temperature'].fillna(22.4)
        self.calc_displacement_fromLatLonInKm()

    def calc_displacement_fromLatLonInKm(self):
        lat1 = self.df['Pickup Lat']
        lon1 = self.df['Pickup Long']
        lat2 = self.df['Destination Lat']
        lon2 = self.df['Destination Long']
        R = 6371 # radius of the earth in km
        dLat = np.radians(lat2-lat1)
        dLon = np.radians(lon2-lon1)
        
        a = (np.sin(dLat/2))** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * (np.sin(dLon/2))**2
        c = 2 * np.arcsin(np.sqrt(a))
        self.df['Displacement'] = R * c
        self.ratings_weight()

    def ratings_weight(self):
        self.df['ratings_weight'] = self.df['Average_Rating'] * self.df['No_of_Ratings']
        self.add_dist_feats()

    def add_dist_feats(self):
        self.df['dist_diff'] = (self.df['Distance (KM)']**2) - (self.df['Displacement']**2)
        self.add_angle_feats()

    def add_angle_feats(self):
        self.df['sin_angle'] = np.sin(self.df['Displacement'] / self.df['Distance (KM)'])
        self.df['cos_angle'] = np.cos(self.df['Displacement'] / self.df['Distance (KM)'])
        self.df['tan_angle'] = np.tan(self.df['Displacement'] / self.df['Distance (KM)'])
        self.savefile()

    def savefile(self):
        data_path = os.path.join(self.repo_path, 'data', 'base_features')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, self.file)
        self.df.to_csv(combined_path_test, index=False)


class Generate_historical_features():
    """
    Historical features are based on data contained in the train.
    I generate features such as statistical mean, median, std.
    These features will be applied to both train and test data in
    a way that ensures no leakage.
    """
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file = 'df_train.csv'
        self.read_data()
        
    def read_data(self):
        to_parse = ['Placement - Time', 'Confirmation - Time', 'Pickup - Time']
        data_path = os.path.join(self.repo_path, 'data', 'base_features', self.file)
        self.df = pd.read_csv(data_path, parse_dates=to_parse)
        self.create_rider_stats()

    def create_rider_stats(self):
        self.df['Speed'] = self.df['Distance (KM)'] / self.df['Time from Pickup to Arrival']
        rider_group = self.df.groupby('Rider Id')
        rider_stat_mean = rider_group['Time from Pickup to Arrival'].mean().astype(np.float16)
        rider_stat_median= rider_group['Time from Pickup to Arrival'].median().astype(np.float16)
        rider_stat_min = rider_group['Time from Pickup to Arrival'].min().astype(np.float16)
        rider_stat_max = rider_group['Time from Pickup to Arrival'].max().astype(np.float16)
        rider_stat_std = rider_group['Time from Pickup to Arrival'].std().astype(np.float16)
        rider_stat_count = rider_group['Time from Pickup to Arrival'].count().astype(np.float16)
        rider_speed_mean = rider_group['Speed'].mean().astype(np.float16)
        rider_speed_median= rider_group['Speed'].median().astype(np.float16)
        rider_speed_min = rider_group['Speed'].min().astype(np.float16)
        rider_speed_max = rider_group['Speed'].max().astype(np.float16)
        rider_speed_std = rider_group['Speed'].std().astype(np.float16)
        rider_speed_count = rider_group['Speed'].count().astype(np.float16)
        self.rider_stats = [rider_stat_mean,
                       rider_stat_median,
                       rider_stat_min,
                       rider_stat_max,
                       rider_stat_std,
                       rider_stat_count,
                       rider_speed_mean,
                       rider_speed_median,
                       rider_speed_min,
                       rider_speed_max,
                       rider_speed_std,
                       rider_speed_count]
        self.savefile()

    def savefile(self):
        data_path = os.path.join(self.repo_path, 'data', 'rider_statistics')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, 'rider_stats.pkl')
        pickle.dump(self.rider_stats, open(combined_path_test, 'wb'))


class Generate_advanced_features():
    """
    Features generated in the historical class are applied to both train
    and test dataframes
    """
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for self.file in ['df_train.csv', 'df_test.csv']:
            self.read_data()
        
    def read_data(self):
        to_parse = ['Placement - Time', 'Confirmation - Time', 'Pickup - Time']
        data_path = os.path.join(self.repo_path, 'data', 'base_features', self.file)
        pkl_path = os.path.join(self.repo_path, 'data', 'rider_statistics', 'rider_stats.pkl')
        self.df = pd.read_csv(data_path, parse_dates=to_parse)
        self.pkl = pickle.load(open(pkl_path, 'rb'))
        self.map_features()

    def map_features(self):
        rider_stat_mean = self.pkl[0]
        rider_stat_median = self.pkl[1]
        rider_stat_min = self.pkl[2]
        rider_stat_max = self.pkl[3]
        rider_stat_std = self.pkl[4]
        rider_stat_count = self.pkl[5]
        rider_speed_mean = self.pkl[6]
        rider_speed_median = self.pkl[7]
        rider_speed_min = self.pkl[8]
        rider_speed_max = self.pkl[9]
        rider_speed_std = self.pkl[10]
        rider_speed_count = self.pkl[11]
        self.df['rider_stat_mean'] = self.df['Rider Id'].map(rider_stat_mean)
        self.df['rider_stat_median'] = self.df['Rider Id'].map(rider_stat_median)
        self.df['rider_stat_min'] = self.df['Rider Id'].map(rider_stat_min)
        self.df['rider_stat_max'] = self.df['Rider Id'].map(rider_stat_max)
        self.df['rider_stat_std'] = self.df['Rider Id'].map(rider_stat_std)
        self.df['rider_stat_count'] = self.df['Rider Id'].map(rider_stat_count)
        self.df['rider_speed_mean'] = self.df['Rider Id'].map(rider_speed_mean)
        self.df['rider_speed_median'] = self.df['Rider Id'].map(rider_speed_median)
        self.df['rider_speed_min'] = self.df['Rider Id'].map(rider_speed_min)
        self.df['rider_speed_max'] = self.df['Rider Id'].map(rider_speed_max)
        self.df['rider_speed_std'] = self.df['Rider Id'].map(rider_speed_std)
        self.df['rider_speed_count'] = self.df['Rider Id'].map(rider_speed_count)
        self.fill_stat_nan()

    def fill_stat_nan(self):
        self.df['rider_stat_mean'] = self.df['rider_stat_mean'].fillna(-100)
        self.df['rider_stat_median'] = self.df['rider_stat_median'].fillna(-200)
        self.df['rider_stat_min'] = self.df['rider_stat_min'].fillna(-300)
        self.df['rider_stat_max'] = self.df['rider_stat_max'].fillna(-400)
        self.df['rider_stat_std'] = self.df['rider_stat_std'].fillna(-500)
        self.df['rider_stat_count'] = self.df['rider_stat_count'].fillna(-600)
        self.df['rider_speed_mean'] = self.df['rider_stat_mean'].fillna(-150)
        self.df['rider_speed_median'] = self.df['rider_stat_median'].fillna(-250)
        self.df['rider_speed_min'] = self.df['rider_stat_min'].fillna(-350)
        self.df['rider_speed_max'] = self.df['rider_stat_max'].fillna(-450)
        self.df['rider_speed_std'] = self.df['rider_stat_std'].fillna(-550)
        self.df['rider_speed_count'] = self.df['rider_stat_count'].fillna(-650)
        self.savefile()

    def savefile(self):
        data_path = os.path.join(self.repo_path, 'data', 'advanced_features')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, self.file)
        self.df.to_csv(combined_path_test, index=False)


class Feature_selection():
    """
    Class drops features that add negatively to the model.
    Confirmation of which features add positively or negatively was
    done using LOFOImportance
    """
    def __init__(self):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for self.file in ['df_train.csv', 'df_test.csv']:
            self.read_data()
        
    def read_data(self):
        to_parse = ['Placement - Time', 'Confirmation - Time', 'Pickup - Time']
        data_path = os.path.join(self.repo_path, 'data', 'advanced_features', self.file)
        self.df = pd.read_csv(data_path, parse_dates=to_parse)
        self.drop_features()

    def drop_features(self):
        to_drop = ['Vehicle Type',
                   'User Id',
                   'Arrival at Pickup - Day of Month',
                   'Arrival at Pickup - Weekday (Mo = 1)',
                   'Arrival at Pickup - Time',
                   'Precipitation in millimeters',
                   'Rider Id',
                   'Arrival at Destination - Day of Month',
                   'Arrival at Destination - Time',
                   'Arrival at Destination - Weekday (Mo = 1)',
                   'Placement - Day of Month',
                   'Placement - Weekday (Mo = 1)',
                   'Placement - Time',
                   'Pickup - Time',
                   'Confirmation - Day of Month',
                   'Confirmation - Weekday (Mo = 1)',
                   'Confirmation - Time']
        self.df = self.df.drop(to_drop, axis='columns')
        self.savefile()

    def savefile(self):
        data_path = os.path.join(self.repo_path, 'data', 'selected_features')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        combined_path_test = os.path.join(data_path, self.file)
        self.df.to_csv(combined_path_test, index=False)

if __name__== "__main__":
    Generate_base_features()
    Generate_historical_features()
    Generate_advanced_features()
    Feature_selection()