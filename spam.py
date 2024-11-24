import pandas as pd
import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Farklı encoding ile dosyayı yüklemeyi deneyin
data = pd.read_csv("C:/Users/hp/OneDrive/Masaüstü/Anlaşılır ekonomi/spam.csv", encoding='latin1')

veri = data.copy()
print(veri)

veri =veri.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
veri = veri.rename(columns={"v1":"ETİKET","v2":"SMS"})

#print(veri.groupby("ETİKET").count())
veri = veri.drop_duplicates()
#print(veri.isnull().sum())

veri["Karakter sayısı"]= veri["SMS"].apply(len)

#veri.hist(column="Karakter sayısı", by="ETİKET",bins=50)
#plt.show()

veri.ETİKET =[1 if kod=="spam" else 0 for kod in veri.ETİKET]
#print(veri)

#mesaj = re.sub("[^a-zA-Z]"," ",veri["SMS"]) # alfabe içeriisndeki harflerin dışındakileri boşlukla değiştir diyorum

def harfler(cumle):
    yer = re.compile("[^a-zA-Z]")
    return re.sub(yer," ",cumle)

durdurma = stopwords.words("english")
#print(durdurma)

spam =[]
ham =[]
tumcumleler =[]

for i in range(len(veri["SMS"].values)):
    r1 =veri["SMS"].values[i]
    r2 =veri["ETİKET"].values[i]     

    temizcumle=[]
    cumleler =harfler(r1)
    cumleler= cumleler.lower()

    for kelimeler in cumleler.split():
        temizcumle.append(kelimeler)

        if r2==1:
            spam.append(cumleler)
        else:
            ham.append(cumleler)    

    tumcumleler.append(" ".join(temizcumle))       

veri["Yeni sms"]= tumcumleler

veri = veri.drop(columns= ["SMS","Karakter sayısı"],axis=1)
print(veri)

cv = CountVectorizer()
x= cv.fit_transform(veri["Yeni sms"]).toarray()

y = veri["ETİKET"]
X=x

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



model = MultinomialNB()
model.fit(X_train,y_train)
tahmin = model.predict(X_test)

acs = accuracy_score(y_test,tahmin)
print(acs*100)