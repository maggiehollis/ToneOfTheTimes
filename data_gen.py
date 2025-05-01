import base64
import csv
import os

import nltk
import requests
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from requests import post

nltk.download("punkt_tab")
nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("stopwords")

os.environ["TRANSFORMERS_NO_TF"] = "1"

# Get api keys from .env
load_dotenv()
spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
genius_token = os.getenv("GENIUS_API_TOKEN")

# Load in emotion model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Get emotional scores of text
def get_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    labels = model.config.id2label
    return {labels[i]: float(probs[i]) for i in range(len(probs))}


# Get Spotify token
def get_token(client_id, client_secret):
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"grant_type": "client_credentials"}

    result = post(url, headers=headers, data=data)
    token = result.json()

    return token


# Check that token is valid
token = get_token(spotify_client_id, spotify_client_secret)
access_token = token["access_token"]


# Retrieve tracks from Spotify playlist
def get_tracks(token):
    playlist_id = "4xHmVFfa1xBKaCBrefgVRv"
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {token}"}

    tracks = []
    offset = 0
    limit = 100

    while True:
        params = {"offset": offset, "limit": limit}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        for item in data["items"]:
            track = item["track"]
            if track:
                track_name = track["name"]
                artist_name = track["artists"][0]["name"]
                tracks.append({"track": track_name, "artist": artist_name})

        if data["next"]:
            offset += limit
        else:
            break

    return tracks


# Access Genius API to get song URL
def get_genius_url(genius_token, artist, title):
    search_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {genius_token}"}
    query = f"{artist} {title}"
    params = {"q": query}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    hits = data["response"]["hits"]
    if hits:
        return hits[0]["result"]["url"]
    return None


# Get lyrics from Genius URL
def get_lyrics(genius_url):
    page = requests.get(genius_url)
    soup = BeautifulSoup(page.text, "html.parser")
    lyrics_containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    lyrics = "\n".join([div.get_text(strip=True) for div in lyrics_containers])
    return lyrics


tracks = get_tracks(access_token)
stop_words = set(stopwords.words("english"))
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiments = {}

# Process each song to get sentiment and emotions
for song in tracks:
    artist = song["artist"]
    title = song["track"]
    print(f"\nProcessing: {artist} @ {title}")
    try:
        genius_url = get_genius_url(genius_token, artist, title)
        if genius_url:
            lyrics = get_lyrics(genius_url)

            # Tokenize lyrics
            tokenized = word_tokenize(lyrics)
            filtered_lyrics = [
                word.lower()
                for word in tokenized
                if word.isalnum() and word.lower() not in stop_words
            ]
            cleaned_lyrics = " ".join(filtered_lyrics)

            # VADER Sentiment
            sentiment = sentiment_analyzer.polarity_scores(cleaned_lyrics)

            # Emotion Sentiment
            emotions = get_emotions(lyrics)

            sentiments[f"{artist} @ {title}"] = {
                "sentiment": sentiment,
                "emotions": emotions,
            }
        else:
            print("Genius URL not found.")
    except Exception as e:
        print(f"Error processing {artist} @ {title}: {e}")

# Determine all possible emotions for CSV header
all_emotions = set()
for data in sentiments.values():
    all_emotions.update(data["emotions"].keys())

# CSV Header
fieldnames = [
    "Artist",
    "Title",
    "Negative",
    "Neutral",
    "Positive",
    "Compound",
] + sorted(all_emotions)

# Export to CSV
with open("song_sentiments_emotions.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for song, data in sentiments.items():
        artist, title = song.split(" @ ")
        row = {
            "Artist": artist,
            "Title": title,
            "Negative": data["sentiment"]["neg"],
            "Neutral": data["sentiment"]["neu"],
            "Positive": data["sentiment"]["pos"],
            "Compound": data["sentiment"]["compound"],
        }
        for emotion in all_emotions:
            row[emotion] = data["emotions"].get(emotion, 0)
        writer.writerow(row)
