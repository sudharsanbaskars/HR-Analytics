from flask import Flask, render_template, request, Response
from flask_cors import cross_origin
from training_validation_insertion import TrainingFilesValidation as train_validation
from train_model import TrainModel as trainModel
from prediction_utils import Prediction


app = Flask(__name__)

# home page
@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

# url for training the data
@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRouteClient():
    try:
        folder_path = "dataset/"
        if folder_path is not None:
            path = folder_path

            train_valObj = train_validation(path) #object initialization
            train_valObj.train_validation()#calling the training_validation function

            trainModelObj = trainModel() #object initialization
            trainModelObj.train() #training the model

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)

    except KeyError:
        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")


# Prediction url
@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def predict():
    try:
        if request.method == "POST":
            education = request.form['education']
            gender = request.form['gender']
            recruitment_channel = request.form['recruitment_channel']
            no_of_trainings = request.form['no_of_trainings']
            age = request.form['age']
            previous_year_rating = float(request.form['previous_year_rating'])
            length_of_service = request.form['length_of_service']
            KPIs_met_above_80_percent = request.form['KPIs_met_above_80_percent']
            awards_won = request.form['awards_won']
            avg_training_score = request.form['avg_training_score']
            department = request.form['department']
            # print(education)
            # print(gender)
            # print(recruitment_channel)
            # print(no_of_trainings)
            # print(age)
            # print(previous_year_rating)
            # print(length_of_service)
            # print(KPIs_met_above_80_percent)
            # print(department)

            pred = Prediction()
            result = pred.predict(education,gender, recruitment_channel, no_of_trainings, age, previous_year_rating,
                         length_of_service, KPIs_met_above_80_percent, awards_won, avg_training_score,
                         department)

            if result == 1:
                result = "This Employer is eligible for Promotion"
            else:
                result = "This Employer is not eligible for Promotion"

            return render_template('index.html', result=result)
        else:
            return Response("Please Enter a Valid Input")

    except Exception as e:
        return Response("Error Occured: "+ str(e))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

