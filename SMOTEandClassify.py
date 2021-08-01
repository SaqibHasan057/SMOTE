import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.model_selection import KFold
import xgboost
from datasetPreprocessing import getPimaDataset,getIonosphereDataset
import smote_variants as sv


np.random.seed(10)



def evaluate_model(predict_fun, X_train, y_train, X_test, y_test):
    '''
    evaluate the model, both training and testing errors are reported
    '''
    # training error
    y_predict_train = predict_fun(X_train)
    train_acc = accuracy_score(y_train,y_predict_train)

    # testing error
    y_predict_test = predict_fun(X_test)
    test_acc = accuracy_score(y_test,y_predict_test)

    #precision,recall,f1score
    precision,recall,f1Score,dud = precision_recall_fscore_support(y_test,y_predict_test,average='binary')

    return train_acc, test_acc,precision,recall,f1Score




def classify(x,y):
    model = xgboost.XGBClassifier()
    kf = KFold(10,True,1)
    score = []

    for train_index,test_index in kf.split(x,y):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model.fit(x_train,y_train)
        train_acc, test_acc, precision, recall, f1Score = evaluate_model(model.predict, x_train, y_train, x_test,y_test)
        score.append([test_acc,precision,recall,f1Score])

    final_score = np.mean(score, axis=0)
    #print(score)
    #print(final_score)

    print("Testing Accuracy: {:.2f}%".format(final_score[0] * 100))
    print("Precision:{:.4f}".format(final_score[1]))
    print("Recall:{:.4f}".format(final_score[2]))
    print("F1_score:{:.4f}".format(final_score[3]))
    print("\n\n")


if __name__=="__main__":
    x,y = getIonosphereDataset()

    print("Without oversampling:")
    classify(x,y)


    SMOTE = sv.SMOTE()
    distance_SMOTE = sv.distance_SMOTE()

    print("Using SMOTE:")
    x_samp_1, y_samp_1 = SMOTE.sample(x, y)
    classify(x_samp_1,y_samp_1)


    print("Using Distance SMOTE:")
    x_samp_2, y_samp_2 = distance_SMOTE.sample(x, y)
    classify(x_samp_2, y_samp_2)

    print(x.shape)
    print(x_samp_1.shape)
    print(x_samp_2.shape)