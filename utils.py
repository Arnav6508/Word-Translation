import numpy as np
import pandas as pd

def get_dict(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    df = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned

    for row in df.itertuples(index = False, name = None):
        etof[row[0]] = row[1] # <- PROBLEMATIC
        '''
        The above line introduces bias in the mapping as if one en value 
        is mapped to multiple fr values then it will only consider the last en value
        '''

    return etof

def get_matrices(en_embeddings, fr_embeddings, en_fr):
    """
    This function returns the matrix of X(eng embeddings) and Y(french embeddings)
    from english-french word mappings and the respective word embeddings of those words
    """
    en_words, fr_words = set(en_embeddings.keys()), set(fr_embeddings.keys())
    en_vecs, fr_vecs = [], []

    for en, fr in en_fr.items():
        if en in en_words and fr in fr_words:
            en_vecs.append(en_embeddings[en])
            fr_vecs.append(fr_embeddings[fr])

    X, Y = np.vstack(en_vecs), np.vstack(fr_vecs)
    return X, Y

def compute_loss(X, Y, R):
    m = X.shape[0]
    return (1/m)*(np.sum(np.square(np.dot(X,R)-Y)))

def compute_gradient(X, Y ,R):
    m = X.shape[0]
    return (2/m)*(np.dot(X.transpose(),np.dot(X, R)-Y))

def train_model(X, Y, train_steps, lr):
    n = X.shape[1]
    R = np.random.rand(n,n)

    for i in range(train_steps):
        if(i%25 == 0): print(f'Loss at {i}:', compute_loss(X, Y, R))
        grad = compute_gradient(X, Y ,R)
        R -= lr*grad
    return R

def cosine_similarity(u, v):
    return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def knn(vec_Y, Y, k = 1):
    similarity = []
    for Y_ele in Y: similarity.append(cosine_similarity(vec_Y, Y_ele))

    sorted_ids = np.argsort(np.array(similarity))[::-1]
    return sorted_ids[:k]

def test(X, R, Y):

    m = X.shape[0]
    pred_Y = np.dot(X,R)
    tot_correct = 0
    
    for i in range(m):
        actual_Y_idx = knn(pred_Y[i], Y)
        if (i == actual_Y_idx): tot_correct += 1
    
    accuracy = tot_correct/m
    print("Accuracy: ", accuracy)
    return accuracy