"""
Shared MBTI Genre Processing Module
Centralized genre mappings and scoring logic for all crawlers
"""

# ==========================================
# MBTI GENRE MAPPINGS
# ==========================================
E_GENRES = ['pop', 'dance', 'edm', 'electronic', 'hip hop', 'hiphop', 'rap', 'house', 'deep house', 
            'future bass', 'latin', 'trap', 'club', 'party', 'k-pop', 'kpop', 'reggaeton', 'upbeat', 
            'workout', 'disco', 'funky', 'funk', 'rock', 'techno']

I_GENRES = ['lofi', 'lo-fi', 'lo fi', 'indie', 'indie pop', 'indie rock', 'acoustic', 'jazz', 'classical', 
            'ambient', 'chill', 'chillhop', 'chillwave', 'folk', 'folkish', 'sleep', 'bedroom pop', 
            'alternative r&b', 'quiet', 'slow', 'meditative', 'peaceful', 'shoegaze', 'metal', 'soul']

S_GENRES = ['pop', 'v-pop', 'vpop', 'country', 'r&b', 'rnb', 'mainstream', 'pop rock', 'adult standards', 
            'schlager', 'bolero', 'easy listening', 'standard', 'singer-songwriter', 'latin', 'upbeat', 'trap']

N_GENRES = ['indie', 'experimental', 'psychedelic', 'synthwave', 'synthpop', 'shoegaze', 'avant-garde', 
            'cyberpunk', 'post-rock', 'progressive', 'prog', 'electronic experimental', 'glitch',
            'vaporwave', 'future funk', 'art rock', 'complex', 'idm', 'math rock', 'alternative']

F_GENRES = ['soul', 'blues', 'emo', 'emotional', 'ballad', 'romantic', 'vocal', 'acapella',
            'gospel', 'singer-songwriter', 'love songs', 'sad', 'sad songs', 'heartbreak', 'indie', 'pop', 'r&b']

T_GENRES = ['metal', 'metalcore', 'deathcore', 'hardcore', 'techno', 'tech house', 'math rock', 
            'idm', 'intelligent dance', 'dubstep', 'bass', 'trance', 'instrumental', 'hardstyle',
            'drum and bass', 'dnb', 'breakcore', 'edm', 'industrial', 'experimental']

ALL_TRAINED_GENRES = E_GENRES + I_GENRES + S_GENRES + N_GENRES + F_GENRES + T_GENRES
HIGH_WEIGHT_GENRES = ['experimental', 'shoegaze', 'synthwave', 'metal', 'metalcore', 'lofi', 'math rock', 'progressive']

# Genre normalization rules
GENRE_REPLACEMENTS = {
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


def normalize_genre(g):
    """Normalize genre string for better matching"""
    if not isinstance(g, str):
        return ""
    
    g = g.lower().strip()
    for key, val in GENRE_REPLACEMENTS.items():
        if key in g:
            g = g.replace(key, val)
    return g


def calculate_genre_mbti_scores(found_genres):
    """
    Calculate MBTI-style genre preferences from a list of genres.
    
    Returns normalized scores (0.0-1.0) for each MBTI dimension:
    - genre_ei: Extraversion (1.0) vs Introversion (0.0)
    - genre_sn: Sensing (1.0) vs Intuition (0.0)
    - genre_tf: Thinking (1.0) vs Feeling (0.0)
    
    Args:
        found_genres: List of genre strings
        
    Returns:
        Dict with 'genre_ei', 'genre_sn', 'genre_tf' keys (all floats 0.0-1.0)
    """
    if not found_genres:
        return {
            'genre_ei': 0.5,
            'genre_sn': 0.5,
            'genre_tf': 0.5
        }
    
    counts = {'e': 0, 'i': 0, 's': 0, 'n': 0, 't': 0, 'f': 0}
    
    for genre in found_genres:
        genre_normalized = normalize_genre(str(genre))
        weight = 2.0 if genre_normalized in HIGH_WEIGHT_GENRES else 1.0
        
        # Map genres to dimensions - a genre can be in multiple dimensions
        if genre_normalized in E_GENRES:
            counts['e'] += weight
        if genre_normalized in I_GENRES:
            counts['i'] += weight
        if genre_normalized in S_GENRES:
            counts['s'] += weight
        if genre_normalized in N_GENRES:
            counts['n'] += weight
        if genre_normalized in T_GENRES:
            counts['t'] += weight
        if genre_normalized in F_GENRES:
            counts['f'] += weight
    
    # Calculate ratios (0.5 is neutral if no preference found)
    genre_ei = counts['e'] / (counts['e'] + counts['i']) if (counts['e'] + counts['i']) > 0 else 0.5
    genre_sn = counts['s'] / (counts['s'] + counts['n']) if (counts['s'] + counts['n']) > 0 else 0.5
    genre_tf = counts['t'] / (counts['t'] + counts['f']) if (counts['t'] + counts['f']) > 0 else 0.5
    
    return {
        'genre_ei': round(genre_ei, 4),
        'genre_sn': round(genre_sn, 4),
        'genre_tf': round(genre_tf, 4)
    }


def match_genre_to_mbti(genre_str):
    """Match a single genre string to trained MBTI genres with smart matching"""
    if not genre_str:
        return None
    
    genre_str = normalize_genre(genre_str)
    
    # Exact match first
    if genre_str in ALL_TRAINED_GENRES:
        return genre_str
    
    # Substring match (prefer longer matches to be more specific)
    best_match = None
    best_score = 0
    for trained_genre in ALL_TRAINED_GENRES:
        if genre_str in trained_genre or trained_genre in genre_str:
            score = len(trained_genre)  # Prefer longer matches
            if score > best_score:
                best_score = score
                best_match = trained_genre
    
    return best_match
