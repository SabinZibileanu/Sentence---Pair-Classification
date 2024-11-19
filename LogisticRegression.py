import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df_antrenare = pd.read_json('/content/date/train.json')

df_testare = pd.read_json('/content/date/test.json')

df_validare = pd.read_json('/content/date/validation.json')

# etapele de preprocesare

def sterge_punctuatie(text):
    for punctuatie in string.punctuation:
        text = text.replace(punctuatie, '')
    return text


def sterge_diacritice(text):
    text_nou = text.replace("ț","t").replace("ș","s").replace("â","a").replace("ă","a").replace("î","i")
    return text_nou

def sterge_spatii_extra(text):
    return text.strip()


def sterge_caractere_speciale(text):
    text_final = re.sub(r'[^ nA-Za-z0-9/]+', '', text)
    return text_final


def lemming_text(text):
    tok = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    rez_final = []

    for char in tok:
      tok_lem = lemmatizer.lemmatize(char)
      rez_final.append(tok_lem)
      rez_final.append(" ")

    return "".join(rez_final)


def sterge_latex(text):
    return re.sub(r'\\[a-zA-Z]+(?:\{[^\}]*\})?', '', text)

def sterge_html(text):
    html_tag_text = BeautifulSoup(text, 'html.parser')
    return html_tag_text.get_text(separator=' ', strip=True)

def aplica_preprocesari(df):
    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].astype(str).apply(lambda col: col.str.lower())
    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].applymap(sterge_diacritice)
    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].applymap(sterge_latex)
    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].applymap(sterge_html)
    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].applymap(sterge_punctuatie)
    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].applymap(sterge_spatii_extra)
    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].applymap(sterge_caractere_speciale)

    df[['sentence1', 'sentence2']] = df[['sentence1','sentence2']].applymap(lemming_text)

    return df

from sklearn.utils import class_weight


def get_weights(label_train):

    weights = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(label_train),
    y = label_train
)
    # obtinerea weights pentru a verifica balansul setului de date
    return weights

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def get_parametrii_optimi(dict_parametrii, feat_train, label_train):
    # gasirea parametrilor optimi
    grd = GridSearchCV(LogisticRegression(), dict_parametrii, cv = 3, n_jobs = -1)
    grd.fit(feat_train, label_train)
    return grd.best_params_

def predict_test(vectorizer1, vectorizer2, classif, df):
    # obtinerea fiecarui label pentru setul de testare

    s1 = df['sentence1']
    s2 = df['sentence2']

    feat_text1 = vectorizer1.transform(s1)
    feat_text2 = vectorizer2.transform(s2)

    matrice_feat = hstack([feat_text1, feat_text2])
    predictii = classif.predict(matrice_feat)

    return predictii

def afis_mat_confuzie(y_true, y_pred, classif):
    # crearea si afisarea matricei de confuzie
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classif.classes_)
    disp.plot()
    plt.show()

def get_csv(pred_test, df_test, nume_fisier):
    df_submissions = df_test.copy()
    df_submissions['label'] = pred_test
    csvv = df_submissions[['guid', 'label']].to_csv(nume_fisier + '.csv', index = False)

df_antrenare_preprocesat = aplica_preprocesari(df_antrenare)
df_validare_preprocesat = aplica_preprocesari(df_validare)
df_testare_preprocesat = aplica_preprocesari(df_testare)

vectorizer_text1 = TfidfVectorizer(ngram_range = (1,2))
feat_train_text1 = vectorizer_text1.fit_transform(df_antrenare_preprocesat['sentence1'])
feat_test_text1 = vectorizer_text1.transform(df_validare_preprocesat['sentence1']) # aplicarea tf - idf pe fiecare propozitie din setul de antrenare si validare

vectorizer_text2 = TfidfVectorizer(ngram_range = (1,2))
feat_train_text2 = vectorizer_text2.fit_transform(df_antrenare_preprocesat['sentence2'])
feat_test_text2 = vectorizer_text2.transform(df_validare_preprocesat['sentence2'])

features_train = hstack([feat_train_text1, feat_train_text2])
features_test = hstack([feat_test_text1, feat_test_text2])

parameters_log = {
      'max_iter':[500,100],
      'class_weight':[None, 'balanced'],
      'C':[1.0,0.1,10],
      'penalty':['l1','l2']
}

dict = get_parametrii_optimi(parameters_log, features_train, df_antrenare_preprocesat['label'])

clasificator = LogisticRegression(max_iter = 1000, class_weight = 'balanced', n_jobs = -1)
clasificator.fit(features_train, df_antrenare_preprocesat['label'])

predict = clasificator.predict(features_test)
clf_report = classification_report(df_validare_preprocesat['label'], predict)

print(clf_report)
afis_mat_confuzie(df_validare_preprocesat['label'], predict, clasificator)
predictii_test = predict_test(vectorizer_text1, vectorizer_text2, clasificator, df_testare_preprocesat)

get_csv(predictii_test, df_testare_preprocesat, 'submisie_exemplu')