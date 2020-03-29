from module.data_processing import *
import datetime
import os
"""
Created at : 28/03/2020
Created by : Angga Pur, Henrico Aldy Ferdian, & Juli Andika
Description : 
Process from get data, splitting data, feature scaling , training , evaluate, and logging
You can choose to using 1.A or 1.B
1.A => NOT convert numerical feature to categorical feature, creating dataset wiith dimension 400 x 152
1.B => convert  numerical feature to categorical feature, creating dataset wiith dimension 400 x 15
2. Training model with neural network
"""



"""
1.A.
This code will not make the numerical column like age and estimated_salary to be categorical (aka stay numerical)
The dataset that you create will be 400 x 152
"""
# # Extract Data
# columns = ['user_id','gender','age','estimated_salary','output']
# data = extract_data('dataset/Social_Network_Ads.csv',columns)
#
# # Make Dataset
# gender = pd.get_dummies(data.gender,prefix='gender')
# age = pd.get_dummies(data.age,prefix='age')
# estimated_salary = pd.get_dummies(data.estimated_salary,prefix='estimared_salary')
# labels = pd.get_dummies(data.output,prefix='condition')
#
# X,y = create_dataset([gender, age, estimated_salary],labels)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)





"""
1.B.
This code will make the numerical column like age and estimated_salary to be categorical
The dataset that you create will be 400 x 15
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



"""
2.
Training Model
Using neural network:
input layer : adjust based on how many feature the dataset have 
hidden layer (1) : 15 node
hidden layer (2) : 10 node
output layer : adjust based on how many label the dataset have  
"""


input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(15, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(10, activation='relu')(dense_layer_2)
dense_layer_4 = Dense(10, activation='relu')(dense_layer_3)
dense_layer_5 = Dense(10, activation='relu')(dense_layer_4)
dense_layer_6 = Dense(10, activation='relu')(dense_layer_5)
dense_layer_7 = Dense(10, activation='relu')(dense_layer_6)
dense_layer_8 = Dense(10, activation='relu')(dense_layer_7)
dense_layer_9 = Dense(10, activation='relu')(dense_layer_8)
dense_layer_10 = Dense(10, activation='relu')(dense_layer_9)
output = Dense(y.shape[1], activation='softmax')(dense_layer_10)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
log_dir= os.path.join('logs','fit',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),'')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(X_train, y_train, batch_size=10, epochs=50, verbose=1, validation_split=0.2, callbacks=[tensorboard_callback])

# Save the model
model.save('saved_model/model.h5')

"""
3. Evaluate Model
"""
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

