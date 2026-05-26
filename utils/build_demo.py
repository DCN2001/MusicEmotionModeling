import os
import numpy as np
import argparse
import glob
import re
import textwrap

def contains_chinese(string):
    return re.search(r'[\u4e00-\u9fa5]', string) is not None

def create_example(args):
    examples = []
    song_list = os.listdir(args.lyric_dir)
    for song in song_list:
        if contains_chinese(song): 
            with open(os.path.join(args.lyric_dir,song), "r", encoding="utf-8") as f:
                lyrics = f.read()
            with open(os.path.join(args.desc_dir,song), "r", encoding="utf-8") as f:
                des = f.read()
            base_va = np.load(os.path.join(args.base_dir,song.replace(".txt",".npy")))
            gt_va = np.load(os.path.join(args.gt_dir,song.replace(".txt",".npy")))
            examples.append({
                "lyrics": lyrics,
                "descriptions": des,
                "prior_va": base_va,
                "gt_va": gt_va
            })
        if len(examples) >= args.N:
            break
    return examples

def build_demo_all_v(examples,demo_path):
    #Header
    header = textwrap.dedent("""\
        You are a music emotion rater.

        Valence model for music emotion (range [1, 9]):
        - Valence reflects the emotional positivity or pleasantness conveyed by the song.
        - Low valence (1–3): sadness, darkness, bitterness, tension.
        - Mid valence (~5): emotional neutrality, ambiguity, mixed feelings, reflective tone.
        - High valence (7–9): happiness, warmth, optimism, comfort, tenderness, serenity.

        You are given THREE complementary sources of information:
        1. **Lyrics** – semantic and narrative emotional cues.
        2. **Description** – text generated from an audio-language model summarizing the musical mood.
        3. **Valence prior** – a rough numeric estimate obtained from an audio-based linear model.

        NEW TASK DEFINITION:
        Instead of predicting the final valence directly, your goal is to predict the **correction delta**, 
        defined as:

                delta = GroundTruth_Valence − Prior_Valence

        The delta may be positive (lyrics/descriptions make the emotion more positive),
        negative (lyrics/descriptions make the emotion more negative),
        or near zero (prior already matches the emotional content).

        Your job:
        - Integrate lyrics + description + prior.
        - Infer **how much** the prior should be moved up or down.
        - Output exactly **ONE number** representing the delta (can be negative or positive).
        - Do NOT output the final valence. ONLY output the delta.

        Output format: ONE number only (e.g., 0.85 or -1.30).

        I will now give you several examples showing how to infer delta from the three information sources.
        Learn from these examples before performing the final task.
    """)

    # Examples
    blocks = []
    for i, ex in enumerate(examples, 1):
        ev = float(ex["prior_va"][0])
        gv = float(ex["gt_va"][0])  # 只取 GT 的 valence
        delta = gv - ev
        lyr = ex["lyrics"].strip()
        desc = ex.get("descriptions", None)
        if desc:
            block = f"""### Example {i}
                    Lyrics:
                    {lyr}

                    Description:
                    {desc.strip()}

                    Prior Valence: {ev:.2f}
                    GroundTruth Valence: {gv:.2f}
                    Delta: {delta:.2f}
                    """
        blocks.append(block)
    examples_text = "\n".join(blocks)

    DEMO_prompt = header + "\n" + examples_text + "\n"
    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(DEMO_prompt)

def build_demo_all_a(examples,demo_path):
    #Header
    header = textwrap.dedent("""\
        You are a music emotion rater.

        Arousal model for music emotion (range [1, 9]):
        - Arousal reflects the emotional intensity or energy conveyed by the song.
        - Low arousal (1–3): calm, relaxed, peaceful, sleepy, gentle.
        - Mid arousal (~5): moderate emotional activity, balanced energy.
        - High arousal (7–9): energetic, intense, excited, aggressive, tense.

        You are given THREE complementary sources of information:
        1. **Lyrics** – semantic and narrative emotional cues.
        2. **Description** – text generated from an audio-language model summarizing the musical mood.
        3. **Arousal prior** – a rough numeric estimate obtained from an audio-based linear model.

        NEW TASK DEFINITION:
        Instead of predicting the final arousal directly, your goal is to predict the **correction delta**, 
        defined as:

                delta = GroundTruth_Arousal − Prior_Arousal

        The delta may be positive (lyrics/descriptions indicate higher intensity),
        negative (lyrics/descriptions indicate calmer emotion),
        or near zero (prior already matches the emotional intensity).

        Your job:
        - Integrate lyrics + description + prior.
        - Infer **how much** the prior should be moved up or down.
        - Output exactly **ONE number** representing the delta (can be negative or positive).
        - Do NOT output the final arousal. ONLY output the delta.

        Output format: ONE number only (e.g., 0.85 or -1.30).

        I will now give you several examples showing how to infer delta from the three information sources.
        Learn from these examples before performing the final task.
    """)

    # Examples
    blocks = []
    for i, ex in enumerate(examples, 1):
        ev = float(ex["prior_va"][1])
        gv = float(ex["gt_va"][1])  # 只取 GT 的 valence
        delta = gv - ev
        lyr = ex["lyrics"].strip()
        desc = ex.get("descriptions", None)
        if desc:
            block = f"""### Example {i}
                    Lyrics:
                    {lyr}

                    Description:
                    {desc.strip()}

                    Prior Arousal: {ev:.2f}
                    GroundTruth Arousal: {gv:.2f}
                    Delta: {delta:.2f}
                    """
        blocks.append(block)
    examples_text = "\n".join(blocks)

    DEMO_prompt = header + "\n" + examples_text + "\n"
    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(DEMO_prompt)

def build_demo_all(examples, demo_path):
    header = textwrap.dedent("""\
        You are a music emotion rater.

        Music emotion is represented in a two-dimensional Valence–Arousal (VA) space with range [1, 9].

        Valence:
        - Reflects emotional positivity or pleasantness.
        - Low valence (1–3): sadness, darkness, bitterness, tension.
        - Mid valence (~5): emotional neutrality, ambiguity, mixed feelings, reflective tone.
        - High valence (7–9): happiness, warmth, optimism, comfort, tenderness, serenity.

        Arousal:
        - Reflects emotional intensity or energy.
        - Low arousal (1–3): calm, relaxed, peaceful, sleepy, gentle.
        - Mid arousal (~5): moderate emotional activity, balanced energy.
        - High arousal (7–9): energetic, intense, excited, aggressive, tense.

        You are given THREE complementary sources of information:
        1. Lyrics – semantic and narrative emotional cues.
        2. Description – text generated from an audio-language model summarizing the musical mood.
        3. Prior VA – rough numeric estimates obtained from an audio-based regression model.

        NEW TASK:
        Instead of predicting the final VA directly, predict correction deltas:

            valence_delta = GroundTruth_Valence − Prior_Valence
            arousal_delta = GroundTruth_Arousal − Prior_Arousal

        Your job:
        - Integrate lyrics + description + prior.
        - Infer how much the prior VA should be adjusted.
        - Output ONLY the correction deltas.

        Output format:
            [valence_delta, arousal_delta]

        Example:
            [-0.85, 0.72]

        Do NOT output explanations.
    """)

    blocks = []

    for i, ex in enumerate(examples, 1):

        pv = float(ex["prior_va"][0])
        pa = float(ex["prior_va"][1])

        gv = float(ex["gt_va"][0])
        ga = float(ex["gt_va"][1])

        delta_v = gv - pv
        delta_a = ga - pa

        lyr = ex["lyrics"].strip()

        desc = ex.get("descriptions", None)

        if desc:

            block = f"""### Example {i}

Lyrics:
{lyr}

Description:
{desc.strip()}

Prior Valence: {pv:.2f}
Prior Arousal: {pa:.2f}

GroundTruth Valence: {gv:.2f}
GroundTruth Arousal: {ga:.2f}

Output:
[{delta_v:.2f}, {delta_a:.2f}]
"""

            blocks.append(block)

    examples_text = "\n".join(blocks)

    DEMO_prompt = header + "\n" + examples_text + "\n"

    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(DEMO_prompt)

def build_demo_base_desc_v(examples,demo_path):
    #Header
    header = textwrap.dedent("""\
        You are a music emotion rater.

        Valence model for music emotion (range [1, 9]):
        - Valence reflects the emotional positivity or pleasantness conveyed by the song.
        - Low valence (1–3): sadness, darkness, bitterness, tension.
        - Mid valence (~5): emotional neutrality or mixed emotion.
        - High valence (7–9): happiness, warmth, optimism, comfort.

        You are given TWO sources of information:
        1. **Description** – text generated from an audio-language model summarizing the musical mood.
        2. **Valence prior** – a rough numeric estimate obtained from an audio-based linear model.

        NEW TASK:
        Instead of predicting the final valence directly, predict the correction delta:

            delta = GroundTruth_Valence − Prior_Valence

        Your job:
        - Use the description to judge whether the emotional positivity should increase or decrease.
        - Adjust the prior valence accordingly.

        Output exactly ONE number representing the delta.
        Do NOT output explanations.

        Example format:
            -0.85
    """)

    # Examples
    blocks = []
    for i, ex in enumerate(examples, 1):
        ev = float(ex["prior_va"][0])
        gv = float(ex["gt_va"][0])  # 只取 GT 的 valence
        delta = gv - ev
        desc = ex.get("descriptions", None)
        if desc:
            block = f"""### Example {i}
                    Description:
                    {desc.strip()}

                    Prior Valence: {ev:.2f}
                    GroundTruth Valence: {gv:.2f}
                    Delta: {delta:.2f}
                    """
        blocks.append(block)
    examples_text = "\n".join(blocks)

    DEMO_prompt = header + "\n" + examples_text + "\n"
    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(DEMO_prompt)

def build_demo_base_desc_a(examples,demo_path):
    #Header
    header = textwrap.dedent("""\
        You are a music emotion rater.

        Arousal model for music emotion (range [1, 9]):
        - Arousal reflects the emotional intensity or energy conveyed by the song.
        - Low arousal (1–3): calm, relaxed, peaceful.
        - Mid arousal (~5): moderate energy.
        - High arousal (7–9): energetic, intense, excited.

        You are given TWO sources of information:
        1. **Description** – text generated from an audio-language model summarizing the musical mood.
        2. **Arousal prior** – a rough numeric estimate obtained from an audio-based linear model.

        NEW TASK:
        Predict the correction delta:

            delta = GroundTruth_Arousal − Prior_Arousal

        Your job:
        - Use the description to judge whether emotional intensity should increase or decrease.
        - Adjust the prior arousal accordingly.

        Output exactly ONE number representing the delta.
        Do NOT output explanations.

        Example format:
            0.75
    """)

    # Examples
    blocks = []
    for i, ex in enumerate(examples, 1):
        ev = float(ex["prior_va"][1])
        gv = float(ex["gt_va"][1])  # 只取 GT 的 valence
        delta = gv - ev
        desc = ex.get("descriptions", None)
        if desc:
            block = f"""### Example {i}
                    Description:
                    {desc.strip()}

                    Prior Arousal: {ev:.2f}
                    GroundTruth Arousal: {gv:.2f}
                    Delta: {delta:.2f}
                    """
        blocks.append(block)
    examples_text = "\n".join(blocks)

    DEMO_prompt = header + "\n" + examples_text + "\n"
    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(DEMO_prompt)

def build_demo_base_desc(examples, demo_path):
    header = textwrap.dedent("""\
        You are a music emotion rater.

        Music emotion is represented in a two-dimensional Valence–Arousal (VA) space with range [1, 9].

        Valence:
        - Reflects emotional positivity or pleasantness.
        - Low valence (1–3): sadness, darkness, bitterness, tension.
        - Mid valence (~5): emotional neutrality or mixed emotion.
        - High valence (7–9): happiness, warmth, optimism, comfort.

        Arousal:
        - Reflects emotional intensity or energy.
        - Low arousal (1–3): calm, relaxed, peaceful.
        - Mid arousal (~5): moderate energy.
        - High arousal (7–9): energetic, intense, excited.

        You are given TWO sources of information:
        1. Description – text generated from an audio-language model summarizing the musical mood.
        2. Prior VA – rough numeric estimates obtained from an audio-based regression model.

        NEW TASK:
        Instead of predicting the final VA directly, predict correction deltas:

            valence_delta = GroundTruth_Valence − Prior_Valence
            arousal_delta = GroundTruth_Arousal − Prior_Arousal

        Your job:
        - Use the description to judge whether emotional positivity and energy should increase or decrease.
        - Adjust the prior VA accordingly.

        Output exactly TWO numbers in the following format:

            [valence_delta, arousal_delta]

        Do NOT output explanations.

        Example format:
            [-0.85, 0.72]
    """)

    blocks = []

    for i, ex in enumerate(examples, 1):

        pv = float(ex["prior_va"][0])
        pa = float(ex["prior_va"][1])

        gv = float(ex["gt_va"][0])
        ga = float(ex["gt_va"][1])

        delta_v = gv - pv
        delta_a = ga - pa

        desc = ex.get("descriptions", None)

        if desc:

            block = f"""### Example {i}

Description:
{desc.strip()}

Prior Valence: {pv:.2f}
Prior Arousal: {pa:.2f}

GroundTruth Valence: {gv:.2f}
GroundTruth Arousal: {ga:.2f}

Output:
[{delta_v:.2f}, {delta_a:.2f}]
"""

            blocks.append(block)

    examples_text = "\n".join(blocks)

    DEMO_prompt = header + "\n" + examples_text + "\n"

    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(DEMO_prompt)

#------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=80, help="Number of demonstrations to use")
    parser.add_argument('--base_dir', type=str, default='/home/dcn2001/Mood_preprocessed/base_va/Mood_base_CH', help="Directory of base VA")
    parser.add_argument('--lyric_dir', type=str, default='/home/dcn2001/Mood_preprocessed/lyrics', help="Directory of lyrics")
    parser.add_argument('--desc_dir', type=str, default='/home/dcn2001/Mood_preprocessed/descriptions', help="Directory of descriptions")
    parser.add_argument('--gt_dir', type=str, default='/home/dcn2001/Mood_preprocessed/label/all', help="Directory of labels")
    # parser.add_argument('--demo_all_path', type=str, default='/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_all.txt', help="Path of the builded demo (.txt file)")
    # parser.add_argument('--demo_bd_path', type=str, default='/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_base_desc.txt', help="Path of the builded demo (.txt file)")
    args = parser.parse_args()
    
    examples = create_example(args)
    build_demo_all_v(examples, '/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_all_v.txt')
    build_demo_all_a(examples, '/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_all_a.txt')
    build_demo_all(examples, '/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_all.txt')
    build_demo_base_desc_v(examples, '/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_base_desc_v.txt')
    build_demo_base_desc_a(examples, '/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_base_desc_a.txt')
    build_demo_base_desc(examples, '/home/dcn2001/workspace/fatcat/ALM/VA-SEQ/TXT/DEMO_base_desc.txt')