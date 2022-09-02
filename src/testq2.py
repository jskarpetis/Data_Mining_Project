import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def tokenize_sentence(sentence):

    sentence_strip_numbers = re.sub(r"[0-9]+", "", sentence)
    final_sentence = re.sub(
        r"!|@|#|$|%|^|&|:|;|'|<|>|/|-|=|(|)|", "", sentence_strip_numbers)
    final_sentence = re.sub('[\W\_]',' ', final_sentence)

    split_sentence = re.split('\s+', final_sentence)
    return split_sentence

def pre_model_processing(summary,vocab_size):
    tokenized_sentence = str(tokenize_sentence(summary))
    encoded_data = [one_hot(tokenized_sentence, vocab_size)]
    return np.array(encoded_data[0])

if __name__=='__main__':
    vocab_size = 20000
    amazon_data = pd.read_csv(os.getcwd() + '/dataset/amazon.csv')
    # amazon_data.hist('Score')
    plt.plot(amazon_data['Score'],'-')
    # plt.title("Complete set")
    # plt.show()
    
    summaries = amazon_data['Text']
    summaries_list = summaries.to_numpy()   
    
    max_len = 0
    processed_data = []
    for summary in summaries_list:
        processed_data.append(pre_model_processing(summary,vocab_size))

    for summary in processed_data:
        if len(summary) > max_len:
            max_len = len(summary)


    processed_data = pad_sequences(processed_data, maxlen=max_len, padding='post')
    processed_data = np.array(processed_data)


    scaled_data = StandardScaler().fit_transform(processed_data)

    #RANDOM FOREST ALGORITHM
    train_X = processed_data[:int(len(processed_data) * 0.8)]
    train_Y = amazon_data['Score'].to_numpy()[:int(len(amazon_data)*0.8)]
    Test_X = processed_data[int(len(processed_data) * 0.8):]
    
    Test_Y = amazon_data['Score'].to_numpy()[int(len(amazon_data)*0.8):]
    print('Train_X shape --> {}\tTest_X shape --> {}'.format(train_X.shape, Test_X.shape))
    print('Train_Y shape --> {}\tTest_Y --> {}'.format(train_Y.shape, Test_Y.shape))
    
    random_forest = RandomForestClassifier(n_estimators=20,verbose=2)
    random_forest.fit(train_X, train_Y)
    
    
    #TESTING PREDICTIONS
    validation_prediction = random_forest.predict(Test_X)
    print("SCORE: ",random_forest.score(Test_X,Test_Y))
    errors = abs(validation_prediction - Test_Y)

    print('Testing Mean Absolute errors:', round(np.mean(errors), 2), 'degrees.')
    
    # Calculate mean absolute percentage error (MAPE)
    mape = np.mean(100 * (errors / Test_Y))
    accuracy = 100 - mape
    print('Testing accuracy: ',accuracy)

    dataframe_testing = pd.DataFrame(validation_prediction)
    # dataframe.hist()

    #TRAINING PREDICTIONS
    train_prediction = random_forest.predict(train_X)
    errors = abs(train_prediction - train_Y)

    print('Training Mean Absolute errors:', round(np.mean(errors), 2), 'degrees.')
    mape = np.mean(100 * (errors / train_Y))
    accuracy = 100 - mape
    print('Training accuracy: ',accuracy)
    
    dataframe_training = pd.DataFrame(train_prediction)
    # dataframe.hist()
    
    #shift test for plotting
    testPlot = np.empty_like(amazon_data)
    testPlot[:, :] = np.nan
    testPlot[len(train_prediction):len(amazon_data),:]=dataframe_testing
    
   
    plt.plot(dataframe_training)
    plt.plot(testPlot)
    plt.legend(['Complete set','training','testing'])
    
    
    plt.show()