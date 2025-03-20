# Netflix Dataset Analysis

## Overview
This repository contains an **Exploratory Data Analysis (EDA)** and **Machine Learning** approach for analyzing the Netflix dataset. The dataset includes metadata about movies and TV shows available on Netflix

## Dataset Description
The dataset consists of :

- **show_id**: Unique identifier for each movie or show.
- **type**: Identifies whether the entry is a "Movie" or "TV Show".
- **title**: Name of the movie or show.
- **director**: Name(s) of the director(s) (if available).
- **cast**: List of actors featured in the content.
- **country**: Country where the movie or show was produced.
- **date_added**: Date when the content was added to Netflix.
- **release_year**: The year the movie or show was released.
- **rating**: Audience rating (e.g., PG-13, R, TV-MA, etc.).
- **duration**: Duration of the movie (in minutes) or number of seasons for TV shows.
- **listed_in**: Categories or genres assigned to the content.
- **description**: A short summary of the movie or show.

## Objectives
This project aims to:
- Perform **Exploratory Data Analysis (EDA)** to identify trends and insights.
- **Visualize** the most common genres, directors, and ratings.

## Data Preprocessing
The dataset underwent the following preprocessing steps:
- **Handling missing values**: Filling or removing missing director, cast, and country information.
- **Feature engineering**: Extracting relevant features such as decade-based trends and content duration categorization.
- **Encoding categorical variables**: Converting text-based categories into numerical representations for analysis.

## Exploratory Data Analysis (EDA)
Several insights were derived from EDA:
- **Most common movie ratings** to determine the type of content available.
- **Top genres** by frequency.
- **Most frequent directors** and their contributions.
- **Content distribution over time** to analyze how Netflix's catalog evolved.

## Visualizations
Key visualizations include:
- **Bar charts** for most frequent genres, ratings, and directors.
- **Pie charts** for comparing different categories of content.
