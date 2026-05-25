def get_desc(audio_path,tokenizer,ALM):
    query = tokenizer.from_list_format([
        {'audio': audio_path}, # Either a local path or an url
        {"text": """
        Listen to the following music and describe it in detail, focusing on emotional and perceptual aspects that reflect the listener’s feelings rather than musical structure.

        In your description, please include:

        The overall emotion or mood of the song (e.g., sad, relaxed, energetic, joyful, tense, melancholic).

        The energy level and tempo feeling (e.g., calm and slow, fast and exciting, gradually building intensity).

        The emotional color of melody and harmony (e.g., bright and uplifting, dark and minor-toned).

        The vocal or instrumental expression (e.g., passionate voice, gentle piano, aggressive drums).

        Focus on how the song makes the listener feel.

        Output one coherent paragraph in English that could help another model infer the song’s valence (positive ↔ negative emotion) and arousal (high ↔ low energy).
        """}])
    response, history = ALM.chat(tokenizer, query=query, history=None)
    return response