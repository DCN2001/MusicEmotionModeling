from transformers import AutoModelForCausalLM, AutoTokenizer
from model.BEATs.BEATs import BEATs, BEATsConfig
from model.linear_model import FF_model
import whisper
from demucs.pretrained import get_model
from demucs.apply import apply_model
from openai import OpenAI
import torch
import argparse
from utils.get_desc import get_desc
from utils.get_base_va import get_base_va
from utils.build_prompt import *
import os
import numpy as np
import librosa
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
    # Initialize DEMUCS and Whisper for getting lyrics
    whisper_model = whisper.load_model("large-v3", device="cpu")
    demucs_model = get_model("htdemucs").to("cuda").eval()
    # Initialze LLM client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    #Start Inference
    full_audio,sr = librosa.load(args.audio_path, sr=16000)
    song_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    #Get base VA
    base_va = get_base_va(full_audio,BEATs_model,base_model)
    #Get desc
    desc = get_desc(args.audio_path,ALM_tokenizer,ALM)
    #Get Lyric
    wav = torch.tensor(full_audio,dtype=torch.float32)
    wav = wav.unsqueeze(0).repeat(2, 1)
    wav = wav.unsqueeze(0).cuda()
    with torch.no_grad():
        sources = apply_model(demucs_model,wav,)
    vocals = sources[0, -1]
    vocal = vocals.mean(dim=0).cpu().numpy()
    if vocal_detect(vocal):
        trans_result = whisper_model.transcribe(vocal)
        lines = [seg["text"].strip() for seg in trans_result["segments"]]
        lyric = "\n".join(lines)
        #Build LLM prompt
        with open("./TXT/DEMO_all_v.txt", "r", encoding="utf-8") as f:
            DEMO_v = f.read()
        with open("./TXT/DEMO_all_a.txt", "r", encoding="utf-8") as f:
            DEMO_a = f.read()
        prompt_v, prompt_a = build_prompt_all_v(DEMO_v,base_va,lyric,desc), build_prompt_all_a(DEMO_a,base_va,lyric,desc)
    else:
        lyric = None
        #Build LLM prompt
        with open("./TXT/DEMO_base_desc_v.txt", "r", encoding="utf-8") as f:
            DEMO_v = f.read()
        with open("./TXT/DEMO_base_desc_a.txt", "r", encoding="utf-8") as f:
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
    print("Final VA:")
    print(result_va)