import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import random

nltk.download("stopwords")
nltk.download("wordnet")

stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()
tokenizer=RegexpTokenizer(r'\w+')

def preprocess(text):
    tokens=tokenizer.tokenize(text.lower())
    tokens=[t for t in tokens if t not in stop_words]
    tokens=[lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

custom_data=[]
with open("movie_lines.csv",newline="",encoding="utf-8") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        if len(row)!=2:
            continue
        custom_data.append((row[0],row[1]))
random.shuffle(custom_data)

texts=[preprocess(text) for text,label in custom_data]
labels=[label for text,label in custom_data]

pipeline=Pipeline([
    ("tfidf",TfidfVectorizer(ngram_range=(1,2),max_features=3000)),
     ("clf",MultinomialNB())
     ])
print("model eğitiliyor...")
pipeline.fit(texts,labels)
print("eğitim tamamlandı!!!")

emoji_map={
    "pos":"(pozitif)",
    "neg":"(negatif)",
    "neu":"(nötr)"
    }

print("\nEnter the movie scene line...(type 'exit' to exit)")
while True:
    user_input=input(">")
    if user_input.strip().lower in ["exit","quit"]:
        print("the program is ending")
        break
    user_cleaned=preprocess(user_input)
    prediction=pipeline.predict([user_cleaned])[0]
    print(f"predicted emotion: {prediction.upper()}{emoji_map[prediction]}\n")
