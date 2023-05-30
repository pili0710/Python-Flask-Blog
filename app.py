import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from google_play_scraper import app
import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews
import re

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/scrapping-data")
def scrap_input():
    return render_template("scrapping.html")

@app.route("/proses-scrap", methods=["POST"])
def scrapping_proses():

    id_google = request.form.get('id_google')
    jumlah_data = request.form.get('jumlah_data')
    jumlah_data = int(jumlah_data)
    
    # Scrape the reviews
    result, continuation_token = reviews(
        id_google,
        lang='id',
        country='id',
        sort=Sort.NEWEST,
        count=jumlah_data, 
        filter_score_with=None 
    )
    
    # Convert the scraped reviews to a DataFrame
    df_busu = pd.DataFrame(np.array(result), columns=['review'])
    df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist()))
    
    # Sort the reviews by date and select the relevant columns
    new_df = df_busu[['userName', 'score', 'at', 'content']]
    sorted_df = new_df.sort_values(by='at', ascending=False) 
    my_df = sorted_df[['userName', 'score', 'at', 'content']] 
    my_df = my_df[['content', 'score']]
    
    # Label the reviews as positive or negative
    def pelabelan(score):
        if score < 3:
            return 'Negatif'
        elif score == 4:
            return 'Positif'
        elif score == 5:
            return 'Positif'

    my_df['label'] = my_df['score'].apply(pelabelan)
    
    # Save the DataFrame to a CSV file
    # my_df.to_csv("scrapped-data.csv", index=False)

    pd.set_option('display.max_column', None)
    # data = pd.read_csv('scrapped-data.csv')
    my_df.dropna(subset=['label'],inplace = True)
    # my_df.to_csv("scrapped-data-labelling.csv", index = False) 
    # df = pd.read_csv('scrapped-data-labelling.csv')

    # case folding versi lain

    def  clean_text(df, text_field, new_text_field_name):
        my_df[new_text_field_name] = my_df[text_field].str.lower()
        my_df[new_text_field_name] = my_df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
        # remove numbers
        my_df[new_text_field_name] = my_df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem)) 
        return my_df
    
    my_df['text_clean'] = my_df['content'].str.lower()
    my_df['text_clean']
    data_clean = clean_text(my_df, 'content', 'text_clean')
    data_clean.head(10)

    # stopword removal versi lain

    import nltk.corpus
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words('indonesian')
    data_clean['text_StopWord'] = data_clean['text_clean'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
    data_clean.head(50)


    # tokenizing versi lain

    import nltk 
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize, word_tokenize
    data_clean['text_tokens'] = data_clean['text_StopWord'].apply(lambda x: word_tokenize(x))
    data_clean.head()

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()


    #-----------------STEMMING -----------------
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    #import swifter


    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}
    hitung=0

    for document in data_clean['text_tokens']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '
                
    print(len(term_dict))
    print("------------------------")
    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        hitung+=1
        print(hitung,":",term,":" ,term_dict[term])

    print(term_dict)
    print("------------------------")

    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]


    #script ini bisa dipisah dari eksekusinya setelah pembacaaan term selesai
    data_clean['text_steamindo'] = data_clean['text_tokens'].apply(lambda x:' '.join(get_stemmed_term(x)))
    data_clean.head(20)

    data_clean.to_csv(f"hasil-text-processing-{id_google}.csv", index=False)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_clean['content'], data_clean['label'], test_size = 0.40, random_state = 0)

    
    X_train_df = pd.DataFrame(X_train, columns=['content'])
    y_train_df = pd.DataFrame(y_train, columns=['label'])
    train_df = pd.concat([X_train_df, y_train_df], axis=1)

    X_test_df = pd.DataFrame(X_test, columns=['content'])
    y_test_df = pd.DataFrame(y_test, columns=['label'])
    test_df = pd.concat([X_test_df, y_test_df], axis=1)

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)



    return redirect(url_for('scrap_input'))

@app.route("/proses-naive-bayes")
def nb_hasil():
    return render_template("hasil-nb.html")

import os

@app.route("/train-data")
def train_data():
    csv_path = 'train.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path) # membaca file CSV
        return render_template('train.html', data=df)
    else:
        return render_template('error.html')

@app.route("/test-data")
def test_data():
    csv_path = 'test.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path) # membaca file CSV
        return render_template('test.html', data=df)
    else:
        return render_template('error.html')


@app.route("/proses-naive-bayes",methods=["POST"])
def nb_hasil_proses():
    import pandas as pd
    file = request.files['file']
    data_clean = pd.read_csv(file)
    #membagi data menjadi data training dan testing dengan test_size = 0.20 dan random state nya 0
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_clean['content'], data_clean['label'], test_size = 0.40, random_state = 0)

    # Pembobotan TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test) 



    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    from sklearn.naive_bayes import MultinomialNB

    nb = MultinomialNB()
    nb.fit(tfidf_train, y_train)

    X_train.toarray()

    y_pred = nb.predict(tfidf_test)

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    print("MultinomialNB Accuracy:", accuracy_score(y_test,predicted))
    print("MultinomialNB Precision:", precision_score(y_test,predicted, average="binary", pos_label="Negatif"))
    print("MultinomialNB Recall:", recall_score(y_test,predicted, average="binary", pos_label="Negatif"))
    print("MultinomialNB f1_score:", f1_score(y_test,predicted, average="binary", pos_label="Negatif"))

    print(f'confusion_matrix:\n {confusion_matrix(y_test, predicted)}')
    print('====================================================\n')
    print(classification_report(y_test, predicted, zero_division=0))


    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import KFold
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define empty lists to store the evaluation metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(data_clean):
        X_train, X_test = data_clean.iloc[train_index]['content'], data_clean.iloc[test_index]['content']
        y_train, y_test = data_clean.iloc[train_index]['label'], data_clean.iloc[test_index]['label']

        # Vectorize the text data using TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test = tfidf_vectorizer.transform(X_test)

        # Train the model
        clf = MultinomialNB()
        clf.fit(tfidf_train, y_train)

        # Make predictions on the test set
        predicted = clf.predict(tfidf_test)

        # Evaluate the performance of the model
        accuracy_scores.append(accuracy_score(y_test,predicted))
        precision_scores.append(precision_score(y_test,predicted, average="macro"))
        recall_scores.append(recall_score(y_test,predicted, average="macro"))
        f1_scores.append(f1_score(y_test,predicted, average="macro"))

        print(f'confusion_matrix:\n {confusion_matrix(y_test, predicted)}')
        print('====================================================\n')
        print(classification_report(y_test, predicted, zero_division=0))

    # Print the average evaluation metrics across all folds
    print("Average Accuracy:", sum(accuracy_scores)/n_splits)
    print("Average Precision:", sum(precision_scores)/n_splits)
    print("Average Recall:", sum(recall_scores)/n_splits)
    print("Average F1-score:", sum(f1_scores)/n_splits)


    count_label = data_clean['label'].value_counts()
    print(count_label)

    count_label = data_clean['label'].value_counts()
    total = count_label.sum()

    percent_pos = (count_label['Positif'] / total) * 100
    percent_neg = (count_label['Negatif'] / total) * 100

    print(f"Persentase Positif: {percent_pos:.2f}%")
    print(f"Persentase Negatif: {percent_neg:.2f}%")


    positif = format(percent_pos, '.2f')
    negatif = format(percent_neg, '.2f')
    akurasi = round((sum(accuracy_scores)/n_splits) * 100, 2)

    return redirect(url_for('nb_hasil_akhir', positif=positif, negatif=negatif, akurasi=akurasi))


@app.route("/hasil-naive-bayes")
def nb_hasil_akhir():
    positif = request.args.get('positif')
    negatif = request.args.get('negatif')
    akurasi = request.args.get('akurasi')
    return render_template("hasil-akhir-nb.html", positif=positif, negatif=negatif, akurasi=akurasi)