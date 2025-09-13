import os
from pathlib import Path
import torch

import librosa

from datetime import datetime
import re

print(f" ~ | torch_can_cuda={torch.cuda.is_available()}")  # should return True

import soundfile as sf
import transformers as tf

def _ensure_top_level(name: str, module_path: str, attr: str | None = None):
    """
    Ensure transformers.<name> exists by importing from module_path and
    assigning to the transformers namespace if needed.
    """
    print(f"[ensure] name={name!r}, module_path={module_path!r}, attr={attr!r}")
    try:
        getattr(tf, name)
        print(f"[ensure] transformers.{name} already present")
        return
    except AttributeError:
        print(f"[ensure] transformers.{name} missing; importing from {module_path}")

    try:
        obj_name = attr or name
        mod = __import__(module_path, fromlist=[obj_name])
        obj = getattr(mod, obj_name)
        setattr(tf, name, obj)
        print(f"[ensure] set transformers.{name} = {module_path}.{obj_name}")
    except Exception as e:
        print(f"[ensure] FAILED to set transformers.{name}: {e}")
        raise

# xcodec2 expects these at top-level on older HF versions:
_ensure_top_level("PreTrainedModel", "transformers.modeling_utils")


from xcodec2.modeling_xcodec2 import XCodec2Model

from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers.utils import logging
logging.set_verbosity_error()

# -------------------- CONFIG --------------------

LLASA_MODEL = "HKUSTAudio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko"
XCODEC_MODEL = "HKUSTAudio/xcodec2"

LLASA_CACHE = Path("data/hf/tts/llasa")
XCODEC_CACHE = Path("data/hf/tts/xcodec2")

PROMPT_WAV = Path("data/tts_prompts/mika_prompt.wav")
OUTPUT_WAV = Path("data/generated/mika_line.wav")

#prompt_text = "Princess Mika gazes at the wreckage of a terran bunker."
#target_text = "Ah~ Your bunker collapsed like a soufflé on fire. How delightfully tragic."

# -------------------- HELPERS --------------------

def is_model_cached(local_dir: Path):
    return local_dir.exists() and any(local_dir.glob("**/*.bin"))

def check_model_size(model_name):
    api = HfApi()
    try:
        repo_info = api.repo_info(model_name)
        size_gb = repo_info.usedStorage / (1024 ** 3)
        print(f"Model size: {size_gb:.2f} GB")
        return size_gb
    except Exception as e:
        print(f"Could not fetch model size info: {e}")
        return 0

def ids_to_speech_tokens(speech_ids):
    return [f"<|s_{sid}|>" for sid in speech_ids]

def extract_speech_ids(speech_tokens_str):
    out = []
    for token in speech_tokens_str:
        if token.startswith('<|s_') and token.endswith('|>'):
            out.append(int(token[4:-2]))
        #else:
            #print("Unexpected token:", token)
    return out


def tokenize_input_text(tokenizer, xcodec, input_text, prompt_wav = None):
    print("-"*80)
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
     # Tokenize the text
    chat = [
        {"role": "user", "content": f"Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" }
    ]
    # Encode the prompt wav
    if prompt_wav is not None:
        secs_to_keep = 0.6
        frames_to_keep = int(16000 * secs_to_keep)
        prompt_wav_tail = prompt_wav[:, -frames_to_keep:]

        vq_code_prompt = xcodec.encode_code(input_waveform=prompt_wav_tail)  # shape [1, 1, T]
        vq_code_prompt = vq_code_prompt[0, 0, :]

        # Determine tokens to keep more precisely
        #tokens_per_sec = vq_code_prompt.shape / secs_to_keep
        #tokens_to_keep = int(tokens_per_sec * secs_to_keep)  # could just be shape[0] too

        #vq_code_prompt = vq_code_prompt[:tokens_to_keep]
        # Convert int 12345 to token <|s_12345|>
        print(f" ~ | prompt_wav_shape: {vq_code_prompt.shape}")
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
        #prefix = "Princess Mika speaks in a playful, sing-song tone. She's charming and eerie, with childlike glee."

        # Tokenize the text
        chat = [
            {"role": "user", "content": f"Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"+ ''.join(speech_ids_prefix) }
        ]

    print(f" ~ | input_chat: {chat}]\n")   

    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True,
        padding=True  # ensures attention_mask is present
    )
    input_ids = input_ids.to('cuda')
    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    return  input_ids, speech_end_id 



def safe_filename(text: str, max_len: int = 20) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())  # clean and safe
    return text[:max_len].rstrip("_")

def save_audio(gen_wav: torch.Tensor, text: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_stem = f"mika_{timestamp}_{safe_filename(text)}"
    
    out_dir = Path("data/generated/mika")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw waveform
    path_raw = out_dir / f"{fname_stem}_raw.wav"
    wav_np = gen_wav[0, 0, :].cpu().numpy()
    sf.write(path_raw, wav_np, 16000)
    print(f"Saved raw 16kHz: {path_raw}")

    y_48k = librosa.resample(wav_np, orig_sr=16000, target_sr=48000)

    # Adjust pitch and speed
    y_stretched = librosa.effects.time_stretch(y_48k, rate=0.98)
    #y_shifted = librosa.effects.pitch_shift(y_stretched, sr=48000, n_steps=0.2)

    # Save adjusted 16kHz
    #path_16k_adj = out_dir / f"{fname_stem}_adjusted.wav"
    #sf.write(path_16k_adj, y_stretched, 16000)
    #print(f"Saved adjusted 16kHz: {path_16k_adj}")

    # Resample adjusted to 48kHz
    #y_48k = librosa.resample(y_stretched, orig_sr=16000, target_sr=48000)
    path_48k_adj = out_dir / f"{fname_stem}_adjusted_48k.wav"
    sf.write(path_48k_adj, y_stretched, 48000)
    print(f"Saved adjusted 48kHz: {path_48k_adj}")

model = None
tokenizer = None
xcodec = None
compiled = False

def load_models():
    global tokenizer, model, xcodec
    print(" ~ | Loading models...\n")
    tokenizer = AutoTokenizer.from_pretrained(LLASA_MODEL, cache_dir=LLASA_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        LLASA_MODEL, cache_dir=LLASA_CACHE, torch_dtype=torch.float16
    ).eval().cuda()
    xcodec = XCodec2Model.from_pretrained(XCODEC_MODEL, cache_dir=XCODEC_CACHE).eval().cuda()
    print(" ~ | Models loaded.\n")


def compile_models():
    global model, xcodec, compiled
    if not compiled:
        print(" ~ | Compiling models with torch.compile...\n")
        model = torch.compile(model)
        xcodec = torch.compile(xcodec)
        compiled = True
        print(" ~ | Models compiled!\n")

def run_tts_pipeline(input_text: str, prompt_wav: torch.Tensor = None):
    global model, tokenizer, xcodec
    print("="*80)
    print(f" ~ | prompt_text: {input_text}\n")
    input_ids, speech_end_id = tokenize_input_text(tokenizer, xcodec, input_text, prompt_wav)
    print(f" ~ | input_ids shape: {input_ids.shape} | eos_id: {speech_end_id}\n")

    outputs = model.generate(
        input_ids,
        max_length=2048,
        eos_token_id=speech_end_id,
        do_sample=True,
        #max_new_tokens=512,
        top_p=1.0,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        #no_repeat_ngram_size=3,
        #repetition_penalty=1.1
    )

    #generated_ids = outputs[0][input_ids.shape[1]:-1] # back by 1
    backtrack = 1
    generated_ids = outputs[0][input_ids.shape[1]-backtrack:] #
    print(f" ~ | generated_ids: {generated_ids.shape[0]} tokens\n")

    #generated_ids = outputs[0][input_ids.shape[1]-10:]  # backtrack 10 tokens

    # gen_start = input_ids.shape[1]
    # gen_full_len = outputs.shape[1] - gen_start

    # # Backtrack up to 2 seconds worth (~100 tokens), but not more than 25% of generated length
    # backtrack = min(100, int(0.25 * gen_full_len))

    # start_idx = max(gen_start - backtrack, 0)
    # generated_ids = outputs[0][start_idx:]
    # print(f" ~ | [Generated] gen_full_len:{gen_full_len} tokens | backtracking {backtrack} tokens")



    decoded_speech_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    speech_ids = extract_speech_ids(decoded_speech_ids)
    out_speech_tokens = torch.tensor(speech_ids).unsqueeze(0).unsqueeze(0).to("cuda")

    if out_speech_tokens.shape[-1] == 0:
        print(" ~ | WARNING: No speech tokens generated. Skipping synthesis.")
        return

    print(" ~ | Decoding audio...\n")
    with torch.no_grad():
        gen_wav = xcodec.decode_code(out_speech_tokens)
        save_audio(gen_wav, input_text)


def run_tts_paragraph(paragraph_text: str, prompt_wav: torch.Tensor = None):
    print(f"\n ~ | Splitting paragraph:\n{paragraph_text}\n")
    sentences = split_mika_lines(paragraph_text)
    print(f" ~ | Found {len(sentences)} sentences.\n")

    # Pre-tokenize prompt wav into <|s_###|> tokens once
    speech_ids_prefix = []
    if prompt_wav is not None:
        vq_code_prompt = xcodec.encode_code(input_waveform=prompt_wav)
        print(f" ~ | prompt_wav_shape: {vq_code_prompt.shape}")   
        vq_code_prompt = vq_code_prompt[0, 0, :]
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    for i, sentence in enumerate(sentences):
        print(f"\n ~ | Generating sentence {i + 1}/{len(sentences)}: {sentence}")

        #sentence = "Mmm..." + sentence
        
        # Inject speech prefix into chat
        chat = [
            {"role": "user", "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{sentence}<|TEXT_UNDERSTANDING_END|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True,
            padding=True
        ).to("cuda")

        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        outputs = model.generate(
            input_ids,
            max_length=2048,
            max_new_tokens=256,
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1.0,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        generated_ids = outputs[0][input_ids.shape[1]:-1]
        decoded_speech_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_ids = extract_speech_ids(decoded_speech_ids)
        out_speech_tokens = torch.tensor(speech_ids).unsqueeze(0).unsqueeze(0).to("cuda")

        if out_speech_tokens.shape[-1] == 0:
            print(" ~ | WARNING: No speech tokens generated.")
            continue

        with torch.no_grad():
            gen_wav = xcodec.decode_code(out_speech_tokens)
            save_audio(gen_wav, sentence)  # Will save with timestamp and text

import re

def split_mika_lines(text):
    # Split on sentence-ending punctuation with optional whitespace
    chunks = re.split(r'(?<=[.!?])\s+', text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# -------------------- MAIN --------------------

kagami_speech2 = '''
Tomo-Ori-Ami

Together in the Woven Web.

Gran sang the song when I was small.
Well...smaller.

It was my favorite lullaby.

A great web of everyone's thoughts, feelings and emotions.

From the squire to the sage, beggars and barons.

Anyone could share in the song, join the choir, enjoy the music between the stars.

I asked why the world wasn't like that anymore.

She just ruffled my hair and said:

They began charging rent.
'''



mika_test_lines = [
    "One, Two! spines pierce through! Three, Four! ready for more? Five, Six! Hey! Dodge this!",
    "Zerglings come in pairs! Just like kidneys, and best friends!",
    "Who's ready for hide and go stab!",
    "Watch your step… Or rather, watch THEIR step! Squish!",
    "We will not go quietly into the night! We will go loudly!",
    # Infestation Pit
    "Oh! Let’s play dress-up! You be you, covered in goo! And we’ll be you too!",
    "Wriggle, twist, don’t try to flee! Soon, you’ll be a part of we!",
    "Nowhere to hide, nowhere to run! We’re gonna have so much fun!",

    # Evolution Chamber
    "Wanna know a secret? The secret ingredient is love! Just kidding! It's actually people!",
    "A cut. A stitch. A dab of goo! Oh dear. Was that inside of you? Cocooned up tight! You’ll hatch good as new!"
]


def main():
    LLASA_CACHE.mkdir(parents=True, exist_ok=True)
    XCODEC_CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)

    load_models()
    compile_models()  # Optional: only if you want faster decode + repeated runs

    # Load prompt audio (optional)
    prompt_wav, _ = sf.read("test.wav")
    prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)#.to("cuda")

    #prompt_wav, sr = sf.read("mika_actually_people.wav")  # sr should be 16000

    # Get last 2 seconds worth of samples
    #samples_per_second = sr
    #last_2_sec = prompt_wav[-2 * samples_per_second:]

    # Save to new file
    #sf.write("mika_actually_people_tail.wav", last_2_sec, sr)
    #print("Saved last 2 seconds to: mika_actually_people_tail.wav")

    run_tts_pipeline(input_text=kagami_speech2, prompt_wav=prompt_wav)

    run_tts_pipeline(input_text="Wanna know a secret? The secret ingredient is love! Just kidding! It's actually people!", prompt_wav=prompt_wav)
    
    #run_tts_pipeline(input_text="We will not go quietly into the night! We will go loudly!", prompt_wav=prompt_wav)
    
    #run_tts_pipeline(input_text="Zerglings come in pairs! Just like kidneys, and best friends!", prompt_wav=prompt_wav)
    # Run on multiple lines or a single one
    # for line in mika_test_lines:
    #     run_tts_pipeline(input_text=line, prompt_wav=prompt_wav)
    #     #run_tts_pipeline(input_text=line, prompt_wav=None)


if __name__ == "__main__":
    main()
