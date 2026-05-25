import math
import numpy as np
import torch

def get_base_va(song, BEATs_model, base_model):
    sr = 16000
    segment_samples = 5 * sr

    total_samples = len(song)

    if total_samples > sr * 30:
        mid = total_samples // 2
        start = mid - sr * 15
        end = mid + sr * 15
        input_song = song[start:end]
    else:
        input_song = song

    num_segments = max(1, math.ceil(len(input_song) / segment_samples))

    song_features = []

    for i in range(num_segments):

        start = i * segment_samples
        end = min((i + 1) * segment_samples, len(input_song))

        segment = input_song[start:end]

        if len(segment) < segment_samples:
            pad_len = segment_samples - len(segment)
            segment = np.pad(segment, (0, pad_len))

        segment_tensor = torch.tensor(
            segment,
            dtype=torch.float32
        ).unsqueeze(0).to('cuda')

        with torch.no_grad():
            features, _ = BEATs_model.extract_features(segment_tensor)
            feature_chunk = features.mean(dim=1).squeeze(0).cpu().numpy()

        song_features.append(feature_chunk)

    song_features = np.stack(song_features)

    org_features = song_features.mean(axis=0)

    with torch.no_grad():

        org_input = (
            torch.tensor(org_features)
            .unsqueeze(0)
            .to('cuda')
        )

        predict_va = base_model(org_input)

    return predict_va.squeeze(0).cpu().numpy()