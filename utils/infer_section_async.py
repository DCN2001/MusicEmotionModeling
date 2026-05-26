from transformers import AutoModelForCausalLM, AutoTokenizer
from model.BEATs.BEATs import BEATs, BEATsConfig
from model.linear_model import FF_model
import whisper
from openai import OpenAI
import torch
import argparse
from utils.get_desc import get_desc
from utils.get_base_va import get_base_va
from utils.build_prompt import *
import os
import numpy as np
import librosa
import soundfile as sf 
import allin1
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def vocal_detect(seg_vocal, threshold=1e-3):
    energy = np.mean(seg_vocal ** 2)
    return energy > threshold

#--------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="/song.mp3", help="The path of target audio.")
    parser.add_argument("--ckpt_BEATs_path", type=str, default="./model_state/BEATs.pt", help="The path of BEATs model weight.")
    parser.add_argument("--ckpt_base_path", type=str, default="./model_state/Full_1.pth", help="The path of base model weight.")
    parser.add_argument("--env_path", type=str, default="/.../.env", help="The path of .env")
    args = parser.parse_args()
    load_dotenv(dotenv_path=args.env_path)
    
    # BEATs model loading
    checkpoint = torch.load(args.ckpt_BEATs_path,weights_only=True)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg).to('cuda') 
    BEATs_model.predictor = None
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    # Base model loading
    base_model = FF_model().to('cuda')
    base_model.load_state_dict(torch.load(args.ckpt_base_path,weights_only=True))
    base_model.eval()
    # QwenAudio loading
    ALM_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
    ALM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True).eval()
    # Whisper loading
    whisper_model = whisper.load_model("large-v3", device="cpu")
    # Initialze LLM client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    #Start Inference
    segment_results = []
    allin1_output = allin1.analyze(args.audio_path,keep_byproducts=True)
    full_audio,sr = librosa.load(args.audio_path, sr=16000)
    song_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    vocal_audio, sr = librosa.load(f"./demix/htdemucs/{song_name}/vocals.wav", sr=16000)
    for i, seg in enumerate(allin1_output.segments):
        start, end, sec_name = seg.start, seg.end, seg.label
        #Skip the section shorter than 5 seconds
        if (end - start)<5:
            result_va = np.array([5.0, 5.0])
            segment_results.append({"start": start,"end": end,"label": sec_name,"va": result_va})
            continue
        start_sample, end_sample = int(seg.start * sr), int(seg.end * sr)
        seg_audio = full_audio[start_sample:end_sample]
        seg_path = f"./demix/{i}_{sec_name}.wav"
        sf.write(seg_path, seg_audio, sr)

        #Get base VA
        base_va = get_base_va(seg_audio,BEATs_model,base_model)
        #Get desc
        desc = get_desc(seg_path,ALM_tokenizer,ALM)
        #Get Lyric
        seg_vocal = vocal_audio[start_sample:end_sample]
        if vocal_detect(seg_vocal):
            trans_result = whisper_model.transcribe(seg_vocal)
            lines = [seg["text"].strip() for seg in trans_result["segments"]]
            lyric = "\n".join(lines)
            #Build LLM prompt
            with open("./utils/TXT/DEMO_all_v.txt", "r", encoding="utf-8") as f:
                DEMO_v = f.read()
            with open("./utils/TXT/DEMO_all_a.txt", "r", encoding="utf-8") as f:
                DEMO_a = f.read()
            prompt_v, prompt_a = build_prompt_all_v(DEMO_v,base_va,lyric,desc), build_prompt_all_a(DEMO_a,base_va,lyric,desc)
        else:
            lyric = None
            #Build LLM prompt
            with open("./utils/TXT/DEMO_base_desc_v.txt", "r", encoding="utf-8") as f:
                DEMO_v = f.read()
            with open("./utils/TXT/DEMO_base_desc_a.txt", "r", encoding="utf-8") as f:
                DEMO_a = f.read()
            prompt_v, prompt_a = build_prompt_bd_v(DEMO_v,base_va,desc), build_prompt_bd_a(DEMO_a,base_va,desc)
        
        #Get LLM refined result
        response_v, reponse_a = client.responses.create(model="gpt-5",input=prompt_v), client.responses.create(model="gpt-5",input=prompt_a)
        output_text_v, output_text_a = response_v.output_text, reponse_a.output_text
        #Post process
        base_v, base_a = float(base_va[0]), float(base_va[1])
        delta_v, delta_a = float(output_text_v.strip()), float(output_text_a.strip())
        adj_v = base_v + delta_v
        adj_a = base_a + delta_a
        #Final output
        result_va = np.array([adj_v,adj_a])
        segment_results.append({"start": start,"end": end,"label": sec_name,"va": result_va})
    
    #Plot
    fig, ax = plt.subplots(figsize=(18, 6))
    for idx, seg in enumerate(segment_results):
        start, end = seg["start"], seg["end"]
        label = seg["label"]

        valence, arousal = seg["va"][0], seg["va"][1]
        center = (start + end) / 2
        # Valence
        ax.plot([start, end],[valence, valence],linewidth=3,color="blue",label="Valence" if idx == 0 else "")
        # Arousal
        ax.plot([start, end],[arousal, arousal],linewidth=3,linestyle="--",color="red",label="Arousal" if idx == 0 else "")

        if idx < len(segment_results) - 1:
            next_seg = segment_results[idx + 1]
            next_valence = next_seg["va"][0]
            next_arousal = next_seg["va"][1]

            # Valence vertical transition
            ax.plot(
                [end, end],
                [valence, next_valence],
                linewidth=2,
                color="blue"
            )

            # Arousal vertical transition
            ax.plot(
                [end, end],
                [arousal, next_arousal],
                linewidth=2,
                linestyle="--",
                color="red"
            )

        # =====================================================
        # Section boundary
        # =====================================================

        ax.axvline(
            start,
            linestyle=":",
            alpha=0.35,
            color="gray"
        )

        # =====================================================
        # Section label
        # =====================================================

        ax.text(
            center,
            9.15,
            f"{label}\n({start:.1f}-{end:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=9
        )

        # =====================================================
        # Value text
        # =====================================================

        ax.text(
            center,
            valence + 0.12,
            f"V={valence:.2f}",
            ha="center",
            fontsize=8,
            color="blue"
        )

        ax.text(
            center,
            arousal - 0.22,
            f"A={arousal:.2f}",
            ha="center",
            fontsize=8,
            color="red"
        )


    ax.axvline(
        segment_results[-1]["end"],
        linestyle=":",
        alpha=0.35,
        color="gray"
    )

    # Axis setup
    ax.set_xlim(segment_results[0]["start"],segment_results[-1]["end"])
    ax.set_ylim(1, 9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VA Value")

    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(f"./plot/{song_name}_section.png",dpi=300,bbox_inches="tight")
    plt.close()

    #Delete Allin1 subproduct
    shutil.rmtree("./demix")
    shutil.rmtree("./spec")