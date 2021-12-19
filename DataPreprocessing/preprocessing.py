import numpy as np
import json
import pandas as pd
from sklearn.impute import KNNImputer
from imblearn.combine import SMOTETomek
from ApplicationLogging.logger import App_Logger

class Preprocessor:
    def __init__(self):
        self.log_writer = App_Logger()

    def label_encoding(self, data, feature, map_dict):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the label_encoding method of Preprocessor Class!")
            data[feature] = data[feature].map(map_dict)
            self.log_writer.log("PreprocessingLog.txt", "Label Mapping of {} feature is completed!!".format(feature))
            return data

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "Label Mapping of {} feature is failed!! :{}".format(feature, str(e)))
            raise e

    def one_hot_encoding(self, data, columns_to_encode):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the one_hot_encoding method of Preprocessor Class!")
            data = pd.get_dummies(data=data, columns=columns_to_encode)
            cols = data.columns
            self.log_writer.log("PreprocessingLog.txt", ")ne Hot Encoding of Preprocessor Class is completed!:The final cols is"+str(cols))
            return data

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "one_hot_encoding is failed!!"+str(e))
            raise e

    def remove_column(self, data, feature):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the remove_columns method of Preprocessor Class!")
            data.drop(feature, axis="columns", inplace=True)
            self.log_writer.log("PreprocessingLog.txt", "remove_columns method of Preprocessor Class is completed!")
            return data

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "remove_columns method of Preprocessor Class is failed! "+str(e))
            raise e

    def knn_imputer(self,data):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the knn_imputer method of Preprocessor Class!")
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            new_array = imputer.fit_transform(data)  # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            new_data = pd.DataFrame(data=new_array, columns=data.columns)
            self.log_writer.log("PreprocessingLog.txt", "knn_imputer method of Preprocessor Class is completed!!!")
            return new_data

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "knn_imputer method of Preprocessor Class is failed!!! "+str(e))
            raise e

    def seperate_features_as_xy(self, data, label):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the seperate_features method of Preprocessor Class!")
            X = data.drop(label, axis="columns")
            y = data[label]
            return X,y

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "seperate_features method of Preprocessor Class is failed "+str(e))
            raise e


    def over_smapling(self, X, y):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the over_smapling method of Preprocessor Class!")
            smk = SMOTETomek(random_state=42)
            sampled_X, sampled_y = smk.fit_resample(X, y)
            self.log_writer.log("PreprocessingLog.txt", "over_smapling method of Preprocessor Class is completed")
            return sampled_X, sampled_y

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "over_smapling method of Preprocessor Class is failed "+str(e))
            raise e

    def change_dtype_to_int(self, data, feature_not_to_change):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the change_dtype_to_int method of Preprocessor Class!")
            for feature in data.columns:
                if feature != feature_not_to_change:
                    data[feature] = data[feature].astype(int)
            return data

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "change_dtype_to_int method of Preprocessor Class Failed! "+str(e))
            raise e

    def save_cols(self,df):
        try:
            self.log_writer.log("PreprocessingLog.txt", "Entered the save_cols method of Preprocessor Class!")
            cols = {"data_columns": [col for col in df.columns]
                    }
            with open("columns.json", "w") as f:
                f.write(json.dumps(cols))

        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt", "save_cols method of Preprocessor Class Failed! "+str(e))
            raise e

    def is_null_present(self,data):

        self.log_writer.log("PreprocessingLog.txt", 'Entered the is_null_present method of the Preprocessor class')
        self.data=data
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = self.data.columns
        try:
            self.null_counts = self.data.isna().sum()  # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if (self.null_present):  # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = self.data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(self.data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.log_writer.log("PreprocessingLog.txt",'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.log_writer.log("PreprocessingLog.txt",'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log("PreprocessingLog.txt",'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()









