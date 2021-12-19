import pandas as pd
from sklearn.model_selection import train_test_split

from ApplicationLogging.logger import App_Logger
from ModelFinder.tuning import Model_Finder
from FileOperations.file_methods import File_Operation
from DataPreprocessing.preprocessing import Preprocessor

class TrainModel:
    def __init__(self):
        self.log_writer = App_Logger()
        self.log_file = "TrainModel.txt"
        self.preprocessor = Preprocessor()
        self.model_finder = Model_Finder()
        self.model_saver = File_Operation()
        self.data_path = "Final_InputDataset/InputFile.csv"

    def train(self):
        try:
            self.log_writer.log(self.log_file, "Training of the model is started")
            # Importing the dataset
            df = pd.read_csv(self.data_path)

            # Label Encoding
            education_map = {"Below Secondary": 0, "Bachelor's": 1, "Master's & above": 2}
            gender_map = {"m": 1, "f": 0}
            recruitment_channel_map = {"sourcing": 0, "referred": 1, "other": 2}
            df = self.preprocessor.label_encoding(df, "education", education_map)
            df = self.preprocessor.label_encoding(df, "gender", gender_map)
            df = self.preprocessor.label_encoding(df, "recruitment_channel", recruitment_channel_map)

            # Removing unwanted columns
            df = self.preprocessor.remove_column(df, 'region')
            df = self.preprocessor.remove_column(df, 'employee_id')

            # One hot encoding
            df = self.preprocessor.one_hot_encoding(df, ['department'])

            # Handling the missing values
            df = self.preprocessor.knn_imputer(df)

            # changing the data types of the features
            df = self.preprocessor.change_dtype_to_int(df, 'previous_year_rating')

            # Seperate features as X and Y
            X, y = self.preprocessor.seperate_features_as_xy(df, 'is_promoted')
            # print(X.shape)
            # print(X.head())

            # Change the column names
            dt = {'awards_won?': 'awards_won', 'KPIs_met >80%': 'KPIs_met_above_80_percent',
                  'department_Analytics': 'analytics',
                  'department_Finance': 'finance', 'department_HR': 'hr', 'department_Legal': 'legal',
                  'department_Operations': 'operations',
                  'department_Procurement': 'procurement', 'department_R&D': 'research_and_development',
                  'department_Sales & Marketing': 'sales_and_marketing', 'department_Technology': 'technology'
                  }
            X = X.rename(columns={'edu': 'education'}, inplace=False)

            # Oversampling
            sampled_X, sampled_y = self.preprocessor.over_smapling(X, y)

            # splitting the data using train_test_split
            X_train, X_test, y_train, y_test = train_test_split(sampled_X, sampled_y, test_size=0.3, random_state=20)

            # Saving the column names for future use
            self.preprocessor.save_cols(X_train)
            # X_train.to_csv("eg.csv", index=False)
            # print(X_train.dtypes)
            # print(y_train.dtypes)

            best_model, model_name = self.model_finder.get_best_model(X_train, y_train, X_test, y_test)

            self.log_writer.log(self.log_file, "The best model is " + str(best_model))

            # saving the best model
            saved_status = self.model_saver.save_model(best_model, str(model_name) + '.sav')

            self.log_writer.log(self.log_file, str(saved_status))

        except Exception as e:
            self.log_writer.log(self.log_file, "Training Model Failed "+str(e))
            raise e