from module.data_processing import *
from tensorflow import keras
from tensorflow.keras import layers


"""
Created at : 28/03/2020
Created by : Angga Pur
Description : 
Using the saved model to predict data
"""

columns = ['user_id','gender','age','estimated_salary','output']
data = extract_data('dataset/Social_Network_Ads.csv',columns)

"""
Make numeric data to be categorical data
the range is (a,b,c) => a is bottom_value, b is top_value+1, c is the step
example : range(18,61,6) => bottom age is 18, toppest age is 60, the step is 6, so it will make 7 class
"""
data["age"] = pd.cut(data["age"],range(18,61,6),include_lowest=True) # will be 7 class
data["estimated_salary"] = pd.cut(data["estimated_salary"],range(15000,150001,22500),include_lowest=True) # will be 6 class

"""
See the distribution Data after convert numerical data to categorical
data = data.groupby("age")["age"].count()
data = data.groupby("estimated_salary")["estimated_salary"].count()
"""

# Make Dataset
gender = pd.get_dummies(data.gender,prefix='gender')
age = pd.get_dummies(data.age,prefix='age')
estimated_salary = pd.get_dummies(data.estimated_salary,prefix='estimared_salary')
labels = pd.get_dummies(data.output,prefix='condition')

X,y = create_dataset([gender, age, estimated_salary],labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=np.random) #0 = not random

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('saved_model/model.h5')

# Evaluate
score = new_model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])