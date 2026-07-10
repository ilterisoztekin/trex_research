
!pip install transformers datasets accelerate -q

import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

#Veri Setini Hazırlama ASşaması

print("--- 1. Adım: Veri Seti Yükleniyor ve Düzenleniyor ---")


train_path, test_path = "train.csv", "test.csv"

for root, dirs, files in os.walk("/"):
    if "train.csv" in files:
        train_path = os.path.join(root, "train.csv")
    if "test.csv" in files:
        test_path = os.path.join(root, "test.csv")

print(f"Bulunan Doğru Yol - Train: {train_path}")
print(f"Bulunan Doğru Yol - Test: {test_path}")


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


sample_size_train = 1000  
sample_size_test = 250    




train_df = train_df.groupby("Class Index", group_keys=False).apply(lambda x: x.sample(sample_size_train, random_state=42))
test_df = test_df.groupby("Class Index", group_keys=False).apply(lambda x: x.sample(sample_size_test, random_state=42))

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df["label"] = train_df["Class Index"] - 1
test_df["label"] = test_df["Class Index"] - 1

target_names = ["World", "Sports", "Business", "Sci-Tech"]

#Text ön işleme aşaması

def clean_text(text):
    """Metni küçültür ve noktalama işaretlerini temizler."""
    text = text.lower()  
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()  
    return text



print("Metin ön işleme adımları uygulanıyor...")
train_df["clean_text"] = train_df["Description"].apply(clean_text)
test_df["clean_text"] = test_df["Description"].apply(clean_text)

#TF-IDF AşamaSı




print("\n--- 2. Adım: TF-IDF Vektörleştirme Yapılıyor ---")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(train_df["clean_text"])
X_test_tfidf = vectorizer.transform(test_df["clean_text"])

y_train = train_df["label"]
y_test = test_df["label"]


def evaluate_model(y_true, y_pred, model_name):
    print(f"\n===== {model_name} Sonuç Raporu =====")
    print(classification_report(y_true, y_pred, target_names=target_names))
    

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Gerçek Sınıf")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.show()

#Naıve Bayes AşŞaması

print("\n--- 3. Adım: Naive Bayes Eğitiliyor ---")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)
evaluate_model(y_test, nb_preds, "Naive Bayes")





#Model 2: Logistic Regression

print("\n--- 4. Adım: Logistic Regression Eğitiliyor ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
evaluate_model(y_test, lr_preds, "Logistic Regression")


#MODEL 3: TRANSFER LEARNING Aşaması(DISTILBERT FINE-TUNING)

print("\n--- 5. Adım: DistilBERT İnce Ayar (Fine-Tuning) Başlıyor ---")


class AGNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)






tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(list(train_df["clean_text"]), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_df["clean_text"]), truncation=True, padding=True, max_length=128)




train_dataset = AGNewsDataset(train_encodings, list(train_df["label"]))
test_dataset = AGNewsDataset(test_encodings, list(test_df["label"]))




model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

#Gpu devrede mi kontrol !
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model şu an eğitilecek donanım: {device.upper()}")
model.to(device)


training_args = TrainingArguments(
    
                                  

    output_dir="./results",          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=100,                
    weight_decay=0.01,               
    logging_dir="./logs",            

    logging_steps=50,
    eval_strategy="epoch",     
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"                 
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


print("İnce ayar yapılıyor ...")
trainer.train()


raw_preds = trainer.predict(test_dataset)
bert_preds = np.argmax(raw_preds.predictions, axis=1)


evaluate_model(y_test, bert_preds, "DistilBERT (Fine-Tuned)")
print("\nTüm deneysel süreçlerr başarıyla tamamlandı!")