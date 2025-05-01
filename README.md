# ToneOfTheTimes

This repository generates song recommendations for articles based on both nltk sentiment and emotional analysis of song lyrics and article content.

## Data Generation

The data has been generated and stored in `song_sentiments_emotions.csv`. But to generate the data we pulled song titles from a 10,000 song playlist on Spotify and then got each song's lyrics from Genius. We then preformed sentiment and emotional analysis on the song lyrics.

## Figure Generation

...

## News Matching

To get songs matched to current articles, run `news_match.ipynb`. Simply adjust the second or third block to select the article you want and once you run the final block, a song will be suggested based on how well its sentiment matches.
