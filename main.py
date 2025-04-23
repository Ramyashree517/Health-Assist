# # importing the dependices

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score

# # Data Collection and analysis
# # PIMA Diabates Dataset
# # loading the diabetes dataset to a pandas Dataframe

# diabetes_dataset=pd.read_csv("diabetes.csv")
# # pd.read_csv?
# pd.read_csv

# # printing the first 5 rows of the dataset
# print(diabetes_dataset.head())

# # number of rows and columns
# diabetes_dataset.shape

# # Getting the statistical mesaures of the data
# diabetes_dataset.describe()
# diabetes_dataset['Outcome'].value_counts()

# diabetes_dataset.groupby('Outcome').mean()

# #separting the data and labels
# X=diabetes_dataset.drop(columns='Outcome',axis=1)
# Y=diabetes_dataset['Outcome']
# print(X)
# print(Y)

# # data standarized

# scaler=StandardScaler()
# scaler.fit(X)
# standardized_data=scaler.transform(X)
# print(standardized_data)

# X=standardized_data
# Y=diabetes_dataset['Outcome']
# print(X)
# print(Y)

# # Train Test Spilt

# X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)

# # training the model
# classifier=svm.SVC(kernel='linear')

# # training the support vector function classifier
# classifier.fit(X_train, Y_train)

# # Model Evalution
# #  accuracy_score

# #accuracy score on the training data
# X_train_prediction =classifier.predict(X_train)
# training_data_accuracy=accuracy_score(X_train_prediction, Y_train)
# print("Accuracy score of the training data:",training_data_accuracy)

# X_test_prediction =classifier.predict(X_test)
# test_data_accuracy=accuracy_score(X_test_prediction, Y_test)
# print("Accuracy score of the test data:",test_data_accuracy)

# # making a predction system

# input_data=(1,89,66,23,94,28.1,0.167,21)

# #changeing the input data to numpy array
# input_data_as_numpy_array=np.asarray(input_data)

# #reshape the array as we are predicting for one instance
# input_data_reshaped=input_data_as_numpy_array.reshape(1, -1)

# #standarised the input data
# std_data=scaler.transform(input_data_reshaped)
# print(std_data)

# prediction=classifier.predict(std_data)
# print(prediction)

# if(prediction[0]==0):
#     print("the person is not diabetic")
# else:
#     print("the person is diabetic")


from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import psycopg2
from psycopg2 import Error

app = Flask(__name__)


# Neon.tech database connection string
NEON_CONNECTION_STRING = "postgresql://neondb_owner_owner:npg_ocwhasli2V7d@ep-shy-meadow-a54ni1fm-pooler.us-east-2.aws.neon.tech/neondb_owner?sslmode=require"

# Load and train diabetes model
diabetes_dataset = pd.read_csv("diabetes.csv")
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])

        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, 
                             insulin, bmi, diabetes_pedigree_function, age])
        input_data = input_data.reshape(1, -1)
        std_data = scaler.transform(input_data)

        prediction = classifier.predict(std_data)
        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
        
        return render_template('index.html', result=result)
    
    except Exception as e:
        error_message = f"Error in prediction: {str(e)}"
        return render_template('index.html', result=error_message)

@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    try:
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        date = request.form['date']
        time = request.form['time']

        if not all([name, email, phone, date, time]):
            return jsonify({'status': 'error', 'message': 'All fields are required'})

        # Connect to Neon.tech database
        connection = psycopg2.connect(NEON_CONNECTION_STRING)
        cursor = connection.cursor()

        insert_query = """
            INSERT INTO appointments (name, email, phone, appointment_date, appointment_time)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (name, email, phone, date, time))
        
        connection.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Appointment booked successfully!'
        })

    except Error as e:
        return jsonify({
            'status': 'error',
            'message': f'Error booking appointment: {str(e)}'
        })
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

if __name__ == '__main__':
    app.run(debug=True)