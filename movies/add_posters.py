
import pandas as pd
import requests
import time

# --------------------------------------------------
# Replace with your TMDB API key
# --------------------------------------------------
API_KEY = "3e1bb823e7b4903d98e5962cff72d83b"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("Loading movies_top100.csv...")

df = pd.read_csv("movies_top100.csv")

print(f"Found {len(df)} movies")

# --------------------------------------------------
# Fetch posters
# --------------------------------------------------
poster_urls = []

for count, movie_id in enumerate(df["id"], start=1):

    print(f"Processing {count}/{len(df)} - Movie ID: {movie_id}")

    poster_url = ""

    try:

        url = f"https://api.themoviedb.org/3/movie/{movie_id}"

        response = requests.get(
            url,
            params={"api_key": API_KEY},
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            poster_path = data.get("poster_path")

            if poster_path:

                poster_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

    except Exception as e:

        print("Error:", e)

    poster_urls.append(poster_url)

    time.sleep(0.2)

# --------------------------------------------------
# Add poster column
# --------------------------------------------------
df["poster_url"] = poster_urls

# Keep only movies with posters
df = df[df["poster_url"] != ""]

# --------------------------------------------------
# Save
# --------------------------------------------------
df.to_csv(
    "movies_top100_posters.csv",
    index=False
)

print()
print("SUCCESS")
print("Movies with posters:", len(df))
print("Saved as movies_top100_posters.csv")
