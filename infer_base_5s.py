from model.BEATs.BEATs import BEATs, BEATsConfig
from model.linear_model import FF_model
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch

#---------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="music.mp3", help="The path of target audio.")
    parser.add_argument("--ckpt_BEATs_path", type=str, default="./model_state/BEATs.pt", help="The path of model weight.")
    parser.add_argument("--ckpt_path1", type=str, default="./model_state/Full_1.pth", help="The path of model weight 1.")
    parser.add_argument("--ckpt_path2", type=str, default="./model_state/Full_2.pth", help="The path of model weight 2(CH).")
    args = parser.parse_args()

    ## Model setup
    #BEATs
    checkpoint = torch.load(args.ckpt_BEATs_path,weights_only=True)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg).to('cuda') 
    BEATs_model.predictor = None
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval() 
    #Linear for valence 
    linear_v = FF_model().to('cuda')
    linear_v.load_state_dict(torch.load(args.ckpt_path1, weights_only=True))
    linear_v.eval()
    #Linear for arousal 
    linear_a = FF_model().to('cuda')
    linear_a.load_state_dict(torch.load(args.ckpt_path2, weights_only=True))
    linear_a.eval()

    ## Load audio
    song_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    full_audio, sr = librosa.load(args.audio_path, sr=16000)

    segment_samples = int(5 * sr)          # window = 5s
    stride_samples = int(2.5 * sr)         # stride = 2.5s

    total_samples = len(full_audio)

    ## Start inference (overlap sliding window)
    valence_list, arousal_list = [], []
    time_list = []

    start = 0
    while start + segment_samples <= total_samples:
        end = start + segment_samples
        segment = full_audio[start:end]

        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to('cuda')

        with torch.no_grad():
            features, _ = BEATs_model.extract_features(segment_tensor)
            feature_chunk = features.mean(dim=1)

            valence_seg = linear_v(feature_chunk).squeeze(0).cpu().numpy()[0]
            arousal_seg = linear_a(feature_chunk).squeeze(0).cpu().numpy()[1]

        valence_list.append(valence_seg)
        arousal_list.append(arousal_seg)

        center_time = (start + end) / 2 / sr
        time_list.append(center_time)
        start += stride_samples


    ## Plot
    duration_sec = total_samples / sr
    plt.figure(figsize=(10,4))
    plt.plot(time_list, valence_list, marker="o", label="Valence")
    plt.plot(time_list, arousal_list, marker="s", label="Arousal")

    plt.xlim(0, duration_sec)
    plt.ylim(1, 9)
    plt.xticks(np.arange(0, duration_sec + 1, 10))

    plt.xlabel("Time (seconds)")
    plt.ylabel("Value")
    plt.title("Valence & Arousal over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./plot/{song_name}_base.png", dpi=300)