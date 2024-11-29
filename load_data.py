import pickle
from utils import get_dict, get_matrices

def load_data():
    en_embeddings = pickle.load(open('./data/en_embeddings.p', 'rb'))
    fr_embeddings = pickle.load(open('./data/fr_embeddings.p', 'rb'))
    en_fr_train = get_dict('./data/en-fr.train.txt')
    en_fr_test = get_dict('./data/en-fr.test.txt')

    X_train, Y_train = get_matrices(en_embeddings, fr_embeddings, en_fr_train)
    X_test, Y_test = get_matrices(en_embeddings, fr_embeddings, en_fr_test)

    return X_train, Y_train, X_test, Y_test