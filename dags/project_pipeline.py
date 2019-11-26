""" 
    Dag has a goal of fetching the data from the data/raw folder,
    generating features, training, and predicting on test data
"""
from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
# from airflow.utils.dates import days_ago
import sys

sys.path.insert(0,"./src/")
from feature_engineering import Generate_base_features, \
    Generate_historical_features, Generate_advanced_features, \
        Feature_selection
from train_model import Train_LGB_Model
from predict_tests import MakePredictions

default_args = {"start_date": "2019-11-24",
                "email": ["agbokenechukwu.k@gmail.com"],
                "email_on_failure": True,
                "eamil_on_retry": True,
                "retries": 0,
                "retry_delay": timedelta(minutes=5)}

dag = DAG("project_pipeline",
          description="Building the entire project",
          # train every first day of the month
          schedule_interval="@monthly",
          default_args=default_args,
          catchup=False)

with dag:
    task_1_create_base_features = PythonOperator(task_id="generate_base_features",
                                                 python_callable=Generate_base_features)

    task_2_create_historical_features = PythonOperator(task_id="generate_historic_features",
                                                     python_callable=Generate_historical_features)

    task_3_create_advanced_features = PythonOperator(task_id="generate_advanced_features",
                                                     python_callable=Generate_advanced_features)

    task_4_select_features = PythonOperator(task_id="feature_selection",
                                                     python_callable=Feature_selection)

    task_5_train_lgb_model = PythonOperator(task_id="train_lgb_model",
                                                     python_callable=Train_LGB_Model)

    task_6_make_predictions = PythonOperator(task_id="make_predictions",
                                                     python_callable=MakePredictions)


    task_1_create_base_features >> task_2_create_historical_features >> task_3_create_advanced_features \
        >> task_4_select_features >> task_5_train_lgb_model >> task_6_make_predictions

