## Sendy project hosted on zindi. For predicting time from pickup to delivery
## How Everything Fits Together

### Project Structure
* adhoc - contains the data science and machine learning engineering process used in building the project
* dags - for airflow consumption to specify workflow
* data - contains the data used for our model. The **raw** folder contains the raw data as I was fed.
* logs - for logging
* output - where the output of our engineering would get save. Each task in the workflow
            would produce atleast one output
* src - the source file containing the same methods as the adhoc that is run with the airflow

#### Inside the adhoc folder
**adhoc/lgb_model.ipynb**
This is where the adhoc analysis was done. Feature selection, engineering, model training,
and predictions were done here. The model was built with light gbm.
The output models were pickled and other prediction data were saved as csv files, ready to
be used by our stacking model file

**adhoc/index.ipynb**
The initial ipynb file that was used to merge all seperate files that were given to me.
It makes use of pandas in this merging operation

**adhoc/EDA.ipynb**
Contains a bit of the Exploratory Data Analysis done for this project.

**src/feature_engineering.py**
Contains 4 classes. Three generate new features to be used for our project and the
last one selects the appropriate features to be used.

**src/train_model.py**
Contains the class used to train our model*s*. It saves these models to a pickle file.
Ready for use for the next project

**src/predict_tests.py**
Contains class that makes use of our saved model to make ouput predictions.


#### Inside the data folder
**The output files are generated in the following order**
* raw *(already present, contains the data to train and predict on)*
* base_features
* rider_statistics
* advanced_features
* selected features
* models
* model_predictions

**reports/model_report.html**
Contains the report on the model, showing the SHAP values of each feature and how
they contribute to model predictions.