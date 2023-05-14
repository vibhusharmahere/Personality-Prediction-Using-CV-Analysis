import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re
from nltk.tokenize import word_tokenize
import pandas as pd
import tweepy

import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def get_tweets(username):


    # Enter your Twitter API credentials
    consumer_key = "KjydR0zAaJDnIlYQxhZfJYYlp"
    consumer_secret = "dnOZ0qanlecgd9JGT5KyNH2i8TwwH8fmmM3GWgCuW143aMohuR"
    access_token = '1533426490078941184-IeDstb75aMo1EhLnFFAZZLbI6OOflA'
    access_token_secret = 'LMOBrFlrJwETXqvsM4HekZoX762d1yJtJWScXndXytdaT'
    bearer_token='AAAAAAAAAAAAAAAAAAAAANbGnQEAAAAAAbaa5%2BpdWfxIszGLrhy%2FnL%2B5pPU%3DdDHetcnN502kKj2iWW2BHHcB8Nh1VGiY09lXXjk8A49XV6JydB'
# consumer_key = "gWREkSJyHnZ0EX2TbDf6ls6hr"
#     consumer_secret = "xdV9tuU5OHqHlGb2kqsSsqcVlnrVNHegaQoyy2eUMkfeLcLIdY"
#     access_token = '1401910044547944448-IYTjEOmLy8D8nGDPeLYAdfoNuLgvL1'
#     access_token_secret = 'qlNZlcj9PuzL2J0hfBs4vpNFafQNwXxsr4YpKfHIei4cN'
#     bearer_token='AAAAAAAAAAAAAAAAAAAAAAbwnQEAAAAAJ39J1ByFAtsJIsVXqL7k3s0kwDo%3D9hRgN7N5gAfy4nsVTsmLTGFdmUtLrqEsda3tcA31k8vjK4VGzX'

    # Authenticate to Twitter API
    # auth = tweepy.OAuth1UserHandler(
    #     consumer_key, consumer_secret, access_token, access_token_secret, bearer_token
    # )

    # Create the API client object

    client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token,
                           access_token_secret=access_token_secret, bearer_token=bearer_token)
    # for tweet in tweepy.Paginator(client.search_recent_tweets, query=username,max_results= 100):
    #     tweets.append(tweet.text)

    # api = tweepy.API(client)
    # tweetey = api.home_timeline()
    # Get the user's timeline tweets
    user = client.get_user(username=username)

    # Extract the user ID from the user object
    user_id = user.id
    tweetey = client.get_users_tweets(user_id, max_results=20)

    tweets = []
    # Print the tweets
    for tweet in tweetey:
        tweets.append(tweet.text)

    return tweets


class ModelNotFoundError(Exception):
    pass

class TokenizerNotFoundError(Exception):
    pass

def load_files():
    try:
        with open("saved-models/RandomForest_E-I.sav", "rb") as file:
            ei_classifier = pickle.load(file)
        with  open("saved-models/RandomForest_N-S.sav", "rb") as file:
            ns_classifier = pickle.load(file)
        with open("saved-models/SVM_F-T.sav", "rb") as file:
            ft_classifier = pickle.load(file)
        with  open("saved-models/RandomForest_J-P.sav", "rb") as file:
            jp_classifier = pickle.load(file)
    except FileNotFoundError:
        raise ModelNotFoundError("One or more model files not found!")

    try:
        with open("vectorizer/vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)
    except FileNotFoundError:
        raise TokenizerNotFoundError("Tokenizer file not found!")

    return ei_classifier, ns_classifier, ft_classifier, jp_classifier, vectorizer


def preprocessing(text):
    stopword_list = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'@([a-zA-Z0-9_]{1,50})', '', text)
    text = re.sub(r'#([a-zA-Z0-9_]{1,50})', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = " ".join([word for word in text.split() if not len(word) < 3])
    text = word_tokenize(text)
    text = [word for word in text if not word in stopword_list]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    return text


def get_prediction(text):
    try:
        ei_classifier, ns_classifier, ft_classifier, jp_classifier, vectorizer = load_files()
    except ModelNotFoundError:
        print("One or more model files not found!")
        return None
    except TokenizerNotFoundError:
        print("Tokenizer file not found!")
        return None
    # tweets = get_tweets(username)
    # text = " ".join(tweets)

    text = preprocessing(text)
    text = vectorizer.transform([text])

    prediction = ""
    e_or_i = "E" if ei_classifier.predict(text)[0] == 1 else "I"
    n_or_s = "N" if ns_classifier.predict(text)[0] == 1 else "S"
    f_or_t = "F" if ft_classifier.predict(text)[0] == 1 else "T"
    j_or_p = "J" if jp_classifier.predict(text)[0] == 1 else "P"
    prediction = e_or_i + n_or_s + f_or_t + j_or_p

    return prediction


def preprocessingtweets(text):
    stopword_list = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'@([a-zA-Z0-9_]{1,50})', '', text)
    text = re.sub(r'#([a-zA-Z0-9_]{1,50})', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = " ".join([word for word in text.split() if not len(word) < 3])
    text = word_tokenize(text)
    text = [word for word in text if not word in stopword_list]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    return text


def get_prediction_for_tweets(username):
    ei_classifier, ns_classifier, ft_classifier, jp_classifier, vectorizer = load_files()
    tweets = get_tweets(username)
    text = " ".join(tweets)

    text = preprocessingtweets(text)
    text = vectorizer.transform([text])

    prediction = ""
    e_or_i = "E" if ei_classifier.predict(text)[0] == 1 else "I"
    n_or_s = "N" if ns_classifier.predict(text)[0] == 1 else "S"
    f_or_t = "F" if ft_classifier.predict(text)[0] == 1 else "T"
    j_or_p = "J" if jp_classifier.predict(text)[0] == 1 else "P"
    prediction = e_or_i + n_or_s + f_or_t + j_or_p

    return prediction, tweets
