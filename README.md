# ToneOfTheTimes

[Medium Post](https://medium.com/@mhollis_2953/tone-of-the-times-b9c8ee438a20)

This repository generates song recommendations for articles based on both sentiment and emotional analysis of song lyrics and article content.

## Data Generation

The data has been generated and stored in `song_sentiments_emotions.csv`. But to generate the data we pulled song titles from a 10,000 song [playlist](https://open.spotify.com/playlist/4xHmVFfa1xBKaCBrefgVRv) on Spotify and then got each song's lyrics from Genius. We then preformed sentiment and emotional analysis on the song lyrics.

## Figure Generation

All the figures generated can be viewed in `makingGraphs.ipynb`. Some graphs required additional analysis; this can also be found in the same file.

## News Matching

To get songs matched to current articles, run `news_match.ipynb`. Simply adjust the second or third block to select the article you want and once you run the final block, a song will be suggested based on how well its sentiment matches.

## API Keys
This repository also requires a `.env` file of the form:

```python
NYT_ID = ...
GUARDIAN_ID = ...
SPOTIFY_CLIENT_ID = ...
SPOTIFY_CLIENT_SECRET = ...
GENIUS_API_TOKEN = ...
```
