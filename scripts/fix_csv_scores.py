import pandas as pd
import os

# Updated genre lists (Matching kaggle_mbti_reprocessor.py)
e_genres = ['pop', 'dance', 'edm', 'electronic', 'hip hop', 'hiphop', 'rap', 'house', 'deep house', 
            'future bass', 'latin', 'trap', 'club', 'party', 'k-pop', 'kpop', 'reggaeton', 'upbeat', 
            'workout', 'disco', 'funky', 'funk', 'rock', 'techno']
i_genres = ['lofi', 'lo-fi', 'lo fi', 'indie', 'indie pop', 'indie rock', 'acoustic', 'jazz', 'classical', 
            'ambient', 'chill', 'chillhop', 'chillwave', 'folk', 'folkish', 'sleep', 'bedroom pop', 
            'alternative r&b', 'quiet', 'slow', 'meditative', 'peaceful', 'shoegaze', 'metal', 'soul']
s_genres = ['pop', 'v-pop', 'vpop', 'country', 'r&b', 'rnb', 'mainstream', 'pop rock', 'adult standards', 
            'schlager', 'bolero', 'easy listening', 'standard', 'singer-songwriter', 'latin', 'upbeat', 'trap']
n_genres = ['indie', 'experimental', 'psychedelic', 'synthwave', 'synthpop', 'shoegaze', 'avant-garde', 
            'cyberpunk', 'post-rock', 'progressive', 'prog', 'electronic experimental', 'glitch',
            'vaporwave', 'future funk', 'art rock', 'complex', 'idm', 'math rock', 'alternative']
f_genres = ['soul', 'blues', 'emo', 'emotional', 'ballad', 'romantic', 'vocal', 'acapella',
            'gospel', 'singer-songwriter', 'love songs', 'sad', 'sad songs', 'heartbreak', 'indie', 'pop', 'r&b']
t_genres = ['metal', 'metalcore', 'deathcore', 'hardcore', 'techno', 'tech house', 'math rock', 
            'idm', 'intelligent dance', 'dubstep', 'bass', 'trance', 'instrumental', 'hardstyle',
            'drum and bass', 'dnb', 'breakcore', 'edm', 'industrial', 'experimental']

def normalize_genre(g):
    if not isinstance(g, str): return ""
    g = g.lower().strip()
    replacements = {
        'alternative': 'indie',
        'electronic dance': 'edm',
        'indie pop': 'indie',
        'indie rock': 'indie',
        'r&b': 'rnb',
        'rhythm and blues': 'rnb',
        'hip-hop': 'hip hop',
        'hip hop/rap': 'hip hop',
        'singer/songwriter': 'singer-songwriter',
        'k-pop': 'kpop',
        'k pop': 'kpop',
        'lo-fi': 'lofi',
        'chill hop': 'chillhop'
    }
    for key, val in replacements.items():
        if key in g:
            g = g.replace(key, val)
    return g

def calculate_genre_mbti_scores(genre_str):
    if not genre_str or not isinstance(genre_str, str):
        return 0.5, 0.5, 0.5
        
    found_genres = [normalize_genre(g.strip()) for g in genre_str.split(',')]
    found_genres = [g for g in found_genres if g]
    
    if not found_genres:
        return 0.5, 0.5, 0.5

    counts = {'e': 0, 'i': 0, 's': 0, 'n': 0, 't': 0, 'f': 0}
    high_weight_genres = ['experimental', 'shoegaze', 'synthwave', 'metal', 'metalcore', 'lofi', 'math rock', 'progressive']

    for genre in found_genres:
        w = 2.0 if genre in high_weight_genres else 1.0
        # Check all dimensions as they can overlap now
        if genre in e_genres: counts['e'] += w
        if genre in i_genres: counts['i'] += w
        if genre in s_genres: counts['s'] += w
        if genre in n_genres: counts['n'] += w
        if genre in f_genres: counts['f'] += w
        if genre in t_genres: counts['t'] += w
    
    # Logic: If counts[pos] + counts[neg] == 0, return 0.5 (Neutral)
    # Alignment: E=1, S=1, T=1 to match training targets
    genre_ei = counts['e'] / (counts['e'] + counts['i']) if (counts['e'] + counts['i']) > 0 else 0.5
    genre_sn = counts['s'] / (counts['s'] + counts['n']) if (counts['s'] + counts['n']) > 0 else 0.5
    genre_tf = counts['t'] / (counts['t'] + counts['f']) if (counts['t'] + counts['f']) > 0 else 0.5
    
    return genre_ei, genre_sn, genre_tf

def main():
    csv_path = 'data/mbti_master_training_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Updating scores based on expanded genre mappings...")
    new_ei = []
    new_sn = []
    new_tf = []
    
    for _, row in df.iterrows():
        ei, sn, tf = calculate_genre_mbti_scores(row['artist_genres'])
        new_ei.append(round(ei, 4))
        new_sn.append(round(sn, 4))
        new_tf.append(round(tf, 4))
        
    df['genre_ei_score'] = new_ei
    df['genre_sn_score'] = new_sn
    df['genre_tf_score'] = new_tf
    
    print(f"Saving updated data to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print("Done! Data cleaned.")

if __name__ == "__main__":
    main()
