from ApplicationLogging.logger import App_Logger
from Training_RawDataValidation.raw_data_validation import RawDataValidation


class TrainingFilesValidation:


    def __init__(self, path):
        self.file_name = 'TrainingFilesValidation.txt'
        self.log_writer = App_Logger()
        self.raw_data = RawDataValidation(path)


    def train_validation(self):
        try:
            self.log_writer.log(self.file_name, "Entered into TrainingFilesValidation for training")
            # extracting values from prediction schema
            noofcolumns, column_names= self.raw_data.values_from_schema()

            # validating column length in the file
            self.log_writer.log(self.file_name, "Started Validating the columns length")
            self.raw_data.validate_column_length(noofcolumns)

            self.raw_data.merge_files_from_path()

            self.raw_data.deleteExistingBadDataTrainingFolder()
            self.raw_data.deleteExistingGoodDataRawFolder()

        except Exception as e:
            self.log_writer.log(self.file_name, "Something went wrong in TrainingFiles Validation.The exception is "+str(e))
            raise e

