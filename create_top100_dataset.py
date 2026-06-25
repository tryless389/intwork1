import pandas as pd
import ast
import os

# ---------------------------------------------------
# File Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

MOVIES_CSV = os.path.join(
    BASE_DIR,
    "data",
    "tmdb_5000_movies.csv"
)

CREDITS_CSV = os.path.join(
    BASE_DIR,
    "data",
    "tmdb_5000_credits.csv"
)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

print("Loading datasets...")

movies = pd.read_csv(MOVIES_CSV)
credits = pd.read_csv(CREDITS_CSV)

print("Movies:", len(movies))
print("Credits:", len(credits))

# ---------------------------------------------------
# Rename and Merge
# ---------------------------------------------------

credits = credits.rename(
    columns={"movie_id": "id"}
)

df = movies.merge(
    credits[["id", "cast", "crew"]],
    on="id",
    how="left"
)

print("Merged:", len(df))

# ---------------------------------------------------
# Director Extraction
# ---------------------------------------------------

def get_director(crew_text):

    try:

        crew = ast.literal_eval(crew_text)

        for person in crew:

            if person.get("job") == "Director":

                return person.get("name")

    except:

        pass

    return ""

# ---------------------------------------------------
# Genre Extraction
# ---------------------------------------------------

def get_genres(genres_text):

    try:

        genres = ast.literal_eval(genres_text)

        return ", ".join(
            [g["name"] for g in genres]
        )

    except:

        return ""

# ---------------------------------------------------
# Apply Transformations
# ---------------------------------------------------

print("Extracting directors...")
df["director"] = df["crew"].apply(get_director)

print("Extracting genres...")
df["genres"] = df["genres"].apply(get_genres)

# ---------------------------------------------------
# Select Top 100 Movies
# ---------------------------------------------------

df = df.sort_values(
    by="popularity",
    ascending=False
)

top100 = df.head(100)

# ---------------------------------------------------
# Keep Required Columns
# ---------------------------------------------------

final_df = top100[
    [
        "id",
        "title",
        "genres",
        "overview",
        "cast",
        "director",
        "vote_average",
        "popularity",
        "release_date"
    ]
]

# ---------------------------------------------------
# Save CSV
# ---------------------------------------------------

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "movies_top100.csv"
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("SUCCESS!")
print("Created:", OUTPUT_FILE)
print("Movies Saved:", len(final_df))