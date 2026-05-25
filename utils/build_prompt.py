def build_prompt_all_v(DEMO,q_base_va,q_lyrics,q_desc):
    qv, qa = float(q_base_va[0]), float(q_base_va[1])
    query = f"""### Task
        Lyrics:
        {q_lyrics}

        Description:
        {q_desc}

        Prior Valence: {qv:.2f}
        Now output ONLY ONE number: the delta = corrected_valence − prior_valence.
        The delta can be negative, positive, or close to zero.
        """
    prompt = DEMO + query
    return prompt

def build_prompt_all_a(DEMO,q_base_va,q_lyrics,q_desc):
    qv, qa = float(q_base_va[0]), float(q_base_va[1])
    query = f"""### Task
        Lyrics:
        {q_lyrics}

        Description:
        {q_desc}

        Prior Arousal: {qa:.2f}
        Now output ONLY ONE number: the delta = corrected_arousal − prior_arousal.
        The delta can be negative, positive, or close to zero.
        """
    prompt = DEMO + query
    return prompt

def build_prompt_all(DEMO, q_base_va, q_lyrics, q_desc):

    qv, qa = float(q_base_va[0]), float(q_base_va[1])

    query = f"""### Task

        Lyrics:
        {q_lyrics}

        Description:
        {q_desc}

        Prior Valence: {qv:.2f}
        Prior Arousal: {qa:.2f}

        Now output ONLY TWO decimal numbers in the format:

        [valence_delta, arousal_delta]

        where:
        valence_delta = corrected_valence − prior_valence
        arousal_delta = corrected_arousal − prior_arousal

        The deltas can be negative, positive, or close to zero.
        """

    prompt = DEMO + query

    return prompt

def build_prompt_bd_v(DEMO,q_base_va,q_desc):
    qv, qa = float(q_base_va[0]), float(q_base_va[1])
    query = f"""### Task
        Description:
        {q_desc}

        Prior Valence: {qv:.2f}
        Now output ONLY ONE number: the delta = corrected_valence − prior_valence.
        The delta can be negative, positive, or close to zero.
        """
    prompt = DEMO + query
    return prompt

def build_prompt_bd_a(DEMO,q_base_va,q_desc):
    qv, qa = float(q_base_va[0]), float(q_base_va[1])
    query = f"""### Task
        Description:
        {q_desc}

        Prior Arousal: {qa:.2f}
        Now output ONLY ONE number: the delta = corrected_arousal − prior_arousal.
        The delta can be negative, positive, or close to zero.
        """
    prompt = DEMO + query
    return prompt

def build_prompt_bd(DEMO, q_base_va, q_desc):

    qv, qa = float(q_base_va[0]), float(q_base_va[1])

    query = f"""### Task

            Description:
            {q_desc}

            Prior Valence: {qv:.2f}
            Prior Arousal: {qa:.2f}

            Now output ONLY TWO decimal numbers in the format:

            [valence_delta, arousal_delta]

            where:
            valence_delta = corrected_valence − prior_valence
            arousal_delta = corrected_arousal − prior_arousal

            The deltas can be negative, positive, or close to zero.
            """

    prompt = DEMO + query

    return prompt