#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer


# !!!!! A Executer la premier fois !!!!!
#nltk.download('punkt')
#nltk.download('stopwords')

path_csv = "Tobacco3482.csv"

def etude_des_classes(path_csv):
    df =pd.read_csv(path_csv)
    print("Nombres de classes : ", len(df['label'].value_counts()))
    print("Nombres d'elements : ", len(df['label']))
    print("Echantillons de la structure : \n", df.iloc[:3])
    sns.countplot(data=df,y='label', order = df['label'].value_counts().index)
    plt.show()
    
def recup_chemin(df):
    chemin_img = df["img_path"]
    chemin_txt = []
    for elem in chemin_img:
        chemin_txt.append(elem.replace('jpg', 'txt'))
        
    return chemin_txt

def recup_fichier(path_csv,stopmot=False):
    df =pd.read_csv(path_csv)
    chemin_txt = recup_chemin(df)
    list_fichiers = []
    
    if (stopmot==True):
        print("Nombre de chemins : ", len(chemin_txt))
        for chemin in chemin_txt:
            with open("Tobacco3482-OCR/"+chemin, 'r') as fichier:
                list_fichiers.append(fichier.read().replace('\n', ''))
        print(" doc recup (3482) : ", len(list_fichiers))

        
        stop_words = set(stopwords.words('english'))
        list_preprocess = []
        for fichier in list_fichiers:
            preprocess = []
            words = word_tokenize(fichier)
            for mot in words:
                if mot not in stop_words:
    
                    preprocess.append(mot)
            str_preprocess = ' '.join(preprocess)# Pour retourner sur une phrase et non une liste
            list_preprocess.append(str_preprocess)
        list_fichiers=list_preprocess
        
    else:
        print("Nombre de chemins : ", len(chemin_txt))
        for chemin in chemin_txt:
            with open("Tobacco3482-OCR/"+chemin, 'r') as fichier:
                list_fichiers.append(fichier.read().replace('\n', ''))
        print("doc recup (3482) : ", len(list_fichiers))
   
    #Conversion sous format df
    for i, content in enumerate(list_fichiers):  
        df.loc[i, 'img_path'] = content


    return df

def split_dataset(X,y):

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

    
    return X_train, X_test, y_train, y_test

def vectorize(X_train, X_test):
    vectorizer = CountVectorizer(max_features=3000)
    vectorizer.fit(X_train)
    X_train_counts = vectorizer.transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    return X_train_counts, X_test_counts



 
etude_des_classes(path_csv)


#Import des fichiers
df_fichiers = recup_fichier(path_csv,stopmot=False)
#print("\nAffichage d'un fichier aléatoire : \n\n", list_fichiers[int(np.random.rand(1) * len(list_fichiers))])
X_train, X_test, y_train, y_test = split_dataset(X=df_fichiers.img_path,y=df_fichiers.label)
X_train_counts, X_test_counts = vectorize(X_train, X_test)

# Naives Bayes

print("\n NAIVES BAYES \n")
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

print("vectorize Naive Bayes Score durant la phase de test : ", clf.score(X_test_counts,y_test))
print("vectorize Naive Bayes Score durant la phase d'entrainement : ", clf.score(X_train_counts,y_train))

y_pred_test = clf.predict(X_test_counts)
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

#TF-IDF REPRESENTATION

print("\n TF-IDF REPRESENTATION \n")
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_Tfid = tf_transformer.transform(X_train_counts)
X_test_Tfid = tf_transformer.transform(X_test_counts)


clf = MultinomialNB()
clf.fit(X_train_Tfid, y_train)

print("TF-IDF Naives Bayes Score durant la phase de test : ", clf.score(X_test_Tfid,y_test))
print("TF-IDF Naives Bayes Score durant la phase d'entrainement : ", clf.score(X_train_Tfid,y_train))
print(X_test_Tfid.shape)
y_pred_test = clf.predict(X_test_Tfid)

print("Classe non prédit(s) avec TF-IDF Naive Bayes: ", set(y_test)-set(y_pred_test))
print(classification_report(y_test, y_pred_test))

print(confusion_matrix(y_test, y_pred_test))

# Random forest with vectorize frequency representation

#!!!!! La phase de test des Hyperparamètres est sur le Notebook !!!!!

print("\n Random Forest \n")

clf_rf = RandomForestClassifier(n_estimators=800, min_samples_split = 6, max_features = 10)

clf_rf.fit(X_train_counts, y_train)

y_pred_test = clf_rf.predict(X_test_counts)

print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

