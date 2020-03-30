# Imports
import nltk
import numpy as np
import random
import string
import re
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import FrenchStemmer

# Instance Stemmer
stemmer = FrenchStemmer()

# Fonction nettoyage texte/input
def cleaner(text):
    text = re.sub(r"(covid-19)|(covid 19)","coronavirus",text)
    text = re.sub("coronavirus coronavirus","coronavirus",text)
    text = re.sub(r"(morts?)|(victimes?)","décès",text)
    text = re.sub(r"(contraventions?|amendes?)","amende",text)
    text = re.sub(r"(chez soi|chez moi|chez vous)","à domicile",text)
    text = re.sub(r"(docteurs?|médecins?|professionn?el de santé)","médecin",text)
    text = re.sub("n.c.a.","NCA",text)
    text = re.sub(r'[éèê]','e',text)
    text = re.sub(r'[ù]','u',text)
    text = re.sub(r'[àâ]','a',text)
    text = re.sub(r'[ç]','c',text)
    text = re.sub(r'[ô]','o',text)
    return text

# fonction qui "stemmatize"
def stem(sentence):
    stem = ''
    for word in nltk.word_tokenize(sentence) :
        stem += ' ' + stemmer.stem(word)
    #print(stem) # Pour débug
    return stem

# Fonction obtention du résultat
def get_answer(user_question):
    question = []
    question.append(user_question)
    tfidf_a = tf_idf_chat.transform(phrases_clean)
    tfidf_q = tf_idf_chat.transform(question)
    vals = cosine_similarity(tfidf_q, tfidf_a)  
    idx=vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    if(user_question!='au revoir'):
    	if (req_tfidf == 0):
    		return "COROBOT: Je n'ai pas bien compris votre question."
    	else:
        	return f"COROBOT : {phrases[idx]} (Taux de confiance: {round(vals[0][idx]*100, 2)}%)"
    else:
        return "COROBOT: Au revoir !"

# On stocke le texte dans une variable
f=open('./static/infos_corona.txt','r',errors = 'ignore',encoding="utf-8-sig")
texte = f.read()

# Transformation du texte en liste de phrases
nltk.download('punkt')
nltk.download('wordnet')
phrase_token = nltk.sent_tokenize(texte) 

# On ne garde que les phrases qui ne sont pas des questions
phrases = []
for phrase in phrase_token:
    if ('?' not in phrase):
        phrases.append(phrase)

# On récupère les stop words fr
from stop_words import get_stop_words
stop_words = get_stop_words('fr')

# Phase de nettoyage du texte
phrases_clean = []
for phrase in phrases:
    phrases_clean.append(cleaner(phrase))

# Entrainement du modèle
TfidfVec = TfidfVectorizer(stop_words = stop_words)
#TfidfVec = TfidfVectorizer(tokenizer=stem,stop_words = stop_words) # Avec stem
tf_idf_chat = TfidfVec.fit(phrases_clean)

# On déclare l'app
app = Flask(__name__)

# Routage
@app.route('/')
def question():
    return render_template('corobot.html',question="Vous posez une question...", answer="Je vous répondrai dans la limite de mes compétences !")

@app.route('/', methods=['POST'])
def answer():
    user_question = request.form['question']
    question = f"Vous: {user_question}"

    answer = get_answer(cleaner(user_question))
    #answer = get_answer(stem(cleaner(user_question))) # Si j'active stem
 
    return render_template('corobot.html', question=question, answer=answer)

# On lance l'app
if __name__ == "__main__":
    app.run()