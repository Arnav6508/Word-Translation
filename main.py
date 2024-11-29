import pickle
from load_data import load_data
from utils import train_model, test

import os
from dotenv import load_dotenv
load_dotenv()

def build_model():
    X_train, Y_train, X_test, Y_test = load_data()

    train_steps = int(os.getenv('train_steps'))
    lr = float(os.getenv('lr'))

    R = train_model(X_train, Y_train, train_steps, lr)
    with open('./weights/rotation_matrix.p', 'wb') as file:
        pickle.dump(R, file)

    accuracy = test(X_test, R, Y_test)
    return accuracy
    
def test_model():
    X_train, Y_train, X_test, Y_test = load_data()
    R = pickle.load(open('./weights/rotation_matrix.p', 'rb'))
    accuracy = test(X_test, R, Y_test)
    return accuracy

build_model()