from module.data_processing import *
import datetime
import os
"""
Created at : 29/03/2020
Created by : Henrico Aldy
Description : Process from get data, splitting data, feature scaling , training , evaluate
"""



"""
1.A.
This code will not make the numerical column like age and estimated_salary to be categorical (aka stay numerical)
The dataset that you create will be 400 x 152
"""
# Extract Data
columns = ['user_id','gender','age','estimated_salary','output']
data = extract_data('dataset/Social_Network_Ads.csv',columns)

# Make Dataset
gender = pd.get_dummies(data.gender,prefix='gender')
age = pd.get_dummies(data.age,prefix='age')
estimated_salary = pd.get_dummies(data.estimated_salary,prefix='estimared_salary')
labels = pd.get_dummies(data.output,prefix='condition')

X,y = create_dataset([gender, age, estimated_salary],labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)




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
output = Dense(y.shape[1], activation='softmax')(dense_layer_2)

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