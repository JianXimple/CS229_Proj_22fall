import collections

import numpy as np

import util

from sklearn import datasets, preprocessing, model_selection, decomposition
from sklearn import neighbors, naive_bayes, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = []
    for word in message.split(' '):
        #normalize each word
        norm_word = word.lower()
        words.append(norm_word)
    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    word_count = collections.defaultdict(int)
    for message in messages:
        for word in set(get_words(message)):
            #increment the word count
            word_count[word] = word_count[word] + 1
    
    final_dic = {}
    for word_dic in word_count.items():
        word, count = word_dic
        #only add words to the dictionary if they occur in at least five messages
        if count >= 5:
            ind = len(final_dic)
            final_dic[word] = ind
    return final_dic
    # *** END CODE HERE ***

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    words = np.zeros((len(messages), len(word_dictionary)))
    idx = 0
    for message in messages:
        for word in get_words(message):
            if word in word_dictionary:
                #if word occurrs >= 5 times
                words[idx, word_dictionary[word]] += 1
        idx = idx + 1
    return words
    # *** END CODE HERE ***

def main():
    x, labels = util.load_review_dataset('test.tsv')

    #split the X data and y classifications
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, labels, test_size = 0.3, shuffle = True)
    #shuffle = True
    #convert to numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # print("The shape of X_training: ",X_train.shape)
    # print("The shape of X_testing: ",X_test.shape)
    # print("The shape of Y_training: ",y_train.shape)
    # print("The shape of Y_testing: ",y_test.shape)
    # # Tokenization of reviews
    # dictionary = create_dictionary(X_train)
    # print('Size of dictionary: ', len(dictionary))
    # util.write_json('review_dictionary', dictionary)
    # train_matrix = transform_text(X_train, dictionary)
    # print(train_matrix.shape)
    # # print(X_train[:10])


    # #Run Naive Bayes
    # test_matrix = transform_text(X_test, dictionary)
    # clf = MultinomialNB()
    # clf.fit(train_matrix, y_train)
    # mn_accuracy = clf.score(test_matrix, y_test)
    # print("Multinomial Naive bayes Classifier Accuracy: ", mn_accuracy)

    # #Support Vector Machine
    # clf = svm.SVC()
    # clf.fit(train_matrix, y_train)
    # SVM_accuracy = clf.score(test_matrix, y_test)
    # print("SVM Classifier Accuracy: ", SVM_accuracy)

    #knn
    

    #print (stopwords)
    #tf=TfidfVectorizer()
    exp={}
    model_type=["Naive bayes","SVM","KNN","MLP"]
    models=[MultinomialNB(),svm.SVC(),KNeighborsClassifier(), MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)]
    tfs=[CountVectorizer(stop_words="english"),TfidfVectorizer(stop_words="english"),CountVectorizer(stop_words="english",ngram_range=(2,4))]
    tfs_type=["BOW","Tfid","n-gram"]
    dict_m={}
    dict_tfs={}
    for i in range(len(model_type)):
        dict_m[model_type[i]]=models[i]
    for i in range(len(tfs_type)):
        dict_tfs[tfs_type[i]]=tfs[i]

    
    for clf_n,clf in dict_m.items():
        for tf_n,tf in dict_tfs.items():
            train_matrix = tf.fit_transform(X_train)
            test_matrix=tf.transform(X_test)
            clf.fit(train_matrix, y_train)
            accuracy = clf.score(test_matrix, y_test)
            exp[clf_n+tf_n]=accuracy

    
    print(exp)
#%%
    fig, ax = plt.subplots()
#%%
    l=[i for i,j in exp.items()]
#%%
    acc = [j for i,j in exp.items() ]

    ax.bar(l, acc, label=l)

    ax.set_ylabel('acc')
    ax.set_title('model_names')
    ax.legend(title='test acc')
    plt.plot()
    plt.show()
    plt.savefig("1.png")
    


            



    # k=KNeighborsClassifier()
    # k.fit(train_matrix,y_train)
    # knn_acc=k.score(test_matrix,y_test)
    # print("knn Classifier Accuracy: ", knn_acc)

    # clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(train_matrix,y_train)
    # mlp_acc=clf.score(test_matrix,y_test)
    # print("mlp Classifier Accuracy: ", mlp_acc)


    # val_matrix = transform_text(val_messages, dictionary)
    # test_matrix = transform_text(test_messages, dictionary)

    # naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    # naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    # np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    # naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    # print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    # top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

if __name__ == "__main__":
    main()

###exp={'Naive bayesBOW': 0.8509852216748769, 
# 'Naive bayesTfid': 0.7955665024630542, 
# 'Naive bayesn-gram': 0.8004926108374384, 
# 'SVMBOW': 0.7992610837438424, 
# 'SVMTfid': 0.8017241379310345, 
# 'SVMn-gram': 0.7955665024630542, 
# 'KNNBOW': 0.7992610837438424, 
# 'KNNTfid': 0.7955665024630542, 
# 'KNNn-gram': 0.7943349753694581, 
# 'MLPBOW': 0.7955665024630542, 
# 'MLPTfid': 0.7955665024630542, 
# 'MLPn-gram': 0.7955665024630542}