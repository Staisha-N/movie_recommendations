import pandas as pd

def transform_genres(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Define the genre mapping
    genre_mapping = {
        "animation": 1,
        "adventure": 2,
        "comedy": 3,
        "action": 4,
        "fantasy": 5,
        "family": 6,
        "romance": 7,
        "heartwarming": 8,
        "drama": 9,
        "dramas": 9,
        "sci-fi": 10,
        "superhero movies": 11,
        "historical films": 12
    }

    # Function to map genres to their corresponding IDs
    def map_genres(genre_string):
        # Split genres and transform them
        genres = genre_string.lower().split(', ')
        genre_ids = []
        for genre in genres:
            if genre == "heartwarming comedies":
                genre_ids.append(8)  # heartwarming
                genre_ids.append(3)  # comedy
            else:
                genre_ids.append(genre_mapping.get(genre, None))
        # Ensure we have 4 genres; fill with None if needed
        while len(genre_ids) < 4:
            genre_ids.append(None)
        return genre_ids[:4]  # Return only the first 4 genres

    # Apply the mapping function and create new columns
    genre_columns = df['genre'].apply(map_genres)
    
    # Create a DataFrame for the genre columns
    genre_df = pd.DataFrame(genre_columns.tolist(), index=df.index)
    
    # Concatenate the age column with the genre DataFrame
    result_df = pd.concat([df['age'], genre_df], axis=1)
    
    # Rename the genre columns
    result_df.columns = ['age'] + [f'genre{i+1}' for i in range(result_df.shape[1] - 1)]
    
    # Fill any None values with 0 (or you can leave them as is)
    result_df = result_df.fillna(0)

    # Save to CSV
    result_df.to_csv(output_file, index=False)
