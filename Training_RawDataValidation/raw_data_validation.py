import os
import shutil
import json
import pandas as pd
from ApplicationLogging.logger import App_Logger


class RawDataValidation:

    """
        This class is used for Validating the Raw Batches of input data.
        The following can be validated using this class
        1. The given file has the all the required columns
        2. The total number of columns are correct
        3. Filling the NAN values with NULL
    """

    def __init__(self, path):
        self.logging_file_name = 'RawDataValidation.txt'
        self.log_writer = App_Logger()
        self.GoodDataFolderPath = 'Training_ValidatedRawData/GoodDataFolder/'
        self.BadDataFolderPath = 'Training_ValidatedRawData/BadDataFolder/'
        self.batch_path = path
        self.schema_path = 'schema_training.json'

    def values_from_schema(self):
        try:
            with open(self.schema_path, 'r') as f:
                dict = json.load(f)
                f.close()


            no_of_columns = dict["NumberofColumns"]
            column_names = dict["ColName"]

            self.log_writer.log(self.logging_file_name,
                                "Got the values from scehema JSON file."
                                                        "The number of columns are " + str(no_of_columns))
            return no_of_columns, column_names

        except Exception as e:
            self.log_writer.log(self.logging_file_name,
                                "Error occured in getting values from schema."
                                                        "The exception is "+str(e))
            raise e



    def createDirectoryForGoodBadDataFolder(self):
        try:
            good_folder_path = os.path.join('Training_ValidatedRawData/', 'GoodDataFolder/')
            if os.path.isdir(good_folder_path) == False:
                os.makedirs(good_folder_path)

            bad_folder_path = os.path.join('Training_ValidatedRawData/', 'BadDataFolder/')
            if os.path.isdir(bad_folder_path) == False:
                os.makedirs(bad_folder_path)

            self.log_writer.log(self.logging_file_name,
                                "Successfully created Directory for good and bad data folder")



        except Exception as e:
            self.log_writer.log(self.logging_file_name,
                                "Something went wrong in creating good and bad data folders.The exception is "+str(e))
            raise e

    def deleteExistingGoodDataRawFolder(self):
        try:
            path = 'Training_ValidatedRawData/'
            if os.path.isdir(path+'GoodDataFolder/'):
                shutil.rmtree(path+'GoodDataFolder/')
            self.log_writer.log(self.logging_file_name,
                                "Good Data Folder Deleted Successfilly!")


        except OSError as e:
            self.log_writer.log(self.logging_file_name,
                                "Something went wrong in deleting existing good data folder.The exception is "+str(e))
            raise e

    def deleteExistingBadDataTrainingFolder(self):
        try:
            path = 'Training_ValidatedRawData/'
            if os.path.isdir(path + 'BadDataFolder/'):
                shutil.rmtree(path + 'BadDataFolder/')
            self.log_writer.log(self.logging_file_name,
                                "Bad Data Folder Deleted Successfilly!")

        except OSError as e:
                self.log_writer.log(self.logging_file_name,
                                    "Something went wrong in deleting existing Bad data folder.The exception is "+str(e))
                raise e

    def moveBadFilesToArchiveBad(self):
        try:
            source = 'Training_ValidatedRawData/BadDataFolder/'
            path = 'Training_ArchivedBadData/'

            if os.path.isdir(source):
                if os.path.isdir(path) == False:
                    os.mkdir(path)

                dest = path + 'BadDataFolder/'
                if os.path.isdir(dest) == False:
                    os.mkdir(dest)

                files = os.listdir(source)
                for file in files:
                    if file not in os.listdir(dest):
                        shutil.move(source+file, dest)
                self.log_writer.log(self.logging_file_name,
                                    "Bad raw files successfully move to archive folder")


        except OSError as e:
            self.log_writer.log(self.logging_file_name,
                                "Something went wrong in deleting existing Bad data folder.The exception is " + str(e))

        except Exception as e:
            self.log_writer.log(self.logging_file_name,
                                "Something went wrong in deleting existing Bad data folder.The exception is " + str(e))


    def validate_column_length(self, NumberOfColumns):
        try:
            self.deleteExistingGoodDataRawFolder()
            self.deleteExistingBadDataTrainingFolder()

            self.createDirectoryForGoodBadDataFolder()

            GoodDataFolderPath = 'Training_ValidatedRawData/GoodDataFolder/'
            BadDataFolderPath = 'Training_ValidatedRawData/BadDataFolder/'

            files = [file for file in os.listdir(self.batch_path)]
            for file in files:
                file_path = self.batch_path + file
                df = pd.read_csv(file_path,encoding= 'unicode_escape')
                no_of_columns = df.shape[1]
                if no_of_columns == NumberOfColumns:
                    shutil.copy(file_path, GoodDataFolderPath)
                    self.log_writer.log(self.logging_file_name,
                                        "Valid file Name!! File successfully moved to Good Data Folder")
                else:
                    shutil.copy(file_path, BadDataFolderPath)
                    self.log_writer.log(self.logging_file_name,
                                        "Invalid File!! File successfully moved to Bad Data Folder")

                self.log_writer.log(self.logging_file_name,
                                    "File column length validation completed!!")


        except OSError as e:
            self.log_writer.log(self.logging_file_name,
                                "Failed in validating column length "+ str(e))

        except Exception as e:
            self.log_writer.log(self.logging_file_name,
                                "Failed in validating column length " + str(e))

    def merge_files_from_path(self):
        try:
            self.log_writer.log(self.logging_file_name,
                                "Entered the merge_files_from_path method")
            final_df = pd.DataFrame()
            for file in os.listdir(self.GoodDataFolderPath):
                if file.endswith('.csv'):
                    #print(file)
                    final_df = final_df.append(pd.read_csv(self.GoodDataFolderPath+file))
            self.log_writer.log(self.logging_file_name, "files has been merged successfully")
            final_df.to_csv("Final_InputDataset/InputFile.csv", index=False)
            # print(final_df.head())
            # print(final_df.shape)
            # print(final_df.columns)

        except Exception as e:
            self.log_writer.log(self.logging_file_name, "Failed in merging files: "+str(e))
            raise e
