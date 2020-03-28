import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model


#Check Version Tensorflow
def check_version_tensorflow():
  print(tf.__version__)

#Extract Data
def extract_data(csv_url,columns_name,header=0):
  cols = columns_name
  cars = pd.read_csv(r''+csv_url, names=cols, header=header).iloc[:, 1:]
  return cars

def create_dataset(data,labels):
  X = pd.concat(data, axis=1)
  y = labels.values
  return X,y


