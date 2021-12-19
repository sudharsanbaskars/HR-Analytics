import json
import numpy as np
from FileOperations.file_methods import File_Operation

class Prediction:
    def __init__(self):
        self.model_name = "Models/random_forest.sav"
        self.col_names = "columns.json"
        self.education_map = {"Below Secondary": 0, "Bachelors": 1, "Master and Above": 2}
        self.gender_map = {"Male": 1, "Female": 0}
        self.recruitment_channel_map = {"Sourcing": 0, "Referred": 1, "Others": 2}
        self.awards_won_map = {"Yes" : 1, "No" : 0}
        self.KPIs_met_above_80_percent_map = {"Yes" : 1, "No": 0}

    def predict(self, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating,
                length_of_service, KPIs_met_above_80_percent, awards_won, avg_training_score,
                department):
        try:
            file_loader = File_Operation()
            model = file_loader.load_model(self.model_name)

            with open(self.col_names, "r") as f:
                data_columns = json.load(f)['data_columns']

            try:
                loc_index = data_columns.index(department)
            except:
                loc_index = -1

            education = self.education_map[education]
            gender = self.gender_map[gender]
            recruitment_channel = self.recruitment_channel_map[recruitment_channel]
            awards_won = self.awards_won_map[awards_won]
            KPIs_met_above_80_percent = self.KPIs_met_above_80_percent_map[KPIs_met_above_80_percent]


            x = np.zeros(len(data_columns))
            x[0] = education
            x[1] = gender
            x[2] = recruitment_channel
            x[3] = no_of_trainings
            x[4] = age
            x[5] = previous_year_rating
            x[6] = length_of_service
            x[7] = KPIs_met_above_80_percent
            x[8] = awards_won
            x[9] = avg_training_score

            if loc_index >= 0:
                x[loc_index] = 1

            return model.predict([x])

        except Exception as e:
            raise e