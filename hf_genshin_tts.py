
from typing import List

import numpy as np

import os
from pathlib import Path
from datetime import datetime
import re

# === key imports
import torch

allow_cuda = torch.cuda.is_available()
print(f" ~ | torch_can_cuda={torch.cuda.is_available()}")  # should return True

import librosa
import soundfile as sf

from huggingface_hub import HfApi


from with_progress_bar import wrap_progress




def _ensure_top_level(name: str, module_path: str, attr: str | None = None):
    """
    Ensure transformers.<name> exists by importing from module_path and
    assigning to the transformers namespace if needed.
    """
    import transformers as tf
    #print(f"[ensure] name={name!r}, module_path={module_path!r}, attr={attr!r}")
    try:
        getattr(tf, name)
        #print(f"[ensure] transformers.{name} already present")
        return
    except AttributeError:
        print(f"[ensure] transformers.{name} missing; importing from {module_path}")

    try:
        obj_name = attr or name
        mod = __import__(module_path, fromlist=[obj_name])
        obj = getattr(mod, obj_name)
        setattr(tf, name, obj)
        #print(f"[ensure] set transformers.{name} = {module_path}.{obj_name}")
    except Exception as e:
        print(f"[ensure] FAILED to set transformers.{name}: {e}")
        raise

# xcodec2 expects these at top-level on older HF versions:
_ensure_top_level("PreTrainedModel", "transformers.modeling_utils")

from transformers import AutoTokenizer, AutoModelForCausalLM

# xcodec
from xcodec2.modeling_xcodec2 import XCodec2Model


# === set verbosity
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

# --- sentence/newline chunking (replace your split_mika_lines with this) ---
_S_ENDS_STR = r'(?<=[.!?])\s+'
_SENT_END = re.compile(_S_ENDS_STR)



def split_mika_lines(text):
    # Split on sentence-ending punctuation with optional whitespace
    out_lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        chunks = re.split(_S_ENDS_STR, ln)
        out_lines.extend(out_lines)
    return out_lines

def split_into_segments(text: str):
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        for seg in _SENT_END.split(ln):
            seg = seg.strip()
            if seg:
                yield seg

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

def save_audio(
    gen_wav: torch.Tensor,
    text: str,
    *,
    adjust: bool = False,
    stretch_rate: float = 0.98,
    save_48k: bool = False,
    target_sr: int = 48000,
):
    """
    Always saves RAW 16k. If save_48k=True:
      - when adjust=False: also saves a plain 48k resample
      - when adjust=True:  applies time-stretch (and you can add pitch) then saves 48k
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_stem = f"mika_{timestamp}_{safe_filename(text)}"

    out_dir = Path("data/generated/mika")
    out_dir.mkdir(parents=True, exist_ok=True)

    # raw 16k mono
    path_raw = out_dir / f"{fname_stem}_raw.wav"
    wav_np = gen_wav[0, 0, :].detach().cpu().numpy().astype("float32", copy=False)
    sf.write(path_raw, wav_np, 16000)
    print(f"Saved raw 16kHz: {path_raw}")

    if not save_48k:
        return

    # simple resample to 48k
    y_48k = librosa.resample(wav_np, orig_sr=16000, target_sr=target_sr)

    if adjust:
        # adjust path (time-stretch; pitch-shift can be added if desired)
        y_adj = librosa.effects.time_stretch(y_48k, rate=stretch_rate)
        path_48k_adj = out_dir / f"{fname_stem}_adjusted_{target_sr//1000}k.wav"
        sf.write(path_48k_adj, y_adj, target_sr)
        print(f"Saved adjusted {target_sr//1000}kHz: {path_48k_adj}")
    else:
        # clean resample only
        path_48k_plain = out_dir / f"{fname_stem}_{target_sr//1000}k.wav"
        sf.write(path_48k_plain, y_48k, target_sr)
        print(f"Saved {target_sr//1000}kHz (no adjust): {path_48k_plain}")


wrap_progress(globals(), {
    'save_audio': 'Save Audio',
    'tokenize_input_text': "Tokenize Input Text"
})


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

# after your load_models() and compile_models() definitions:
wrap_progress(globals(), {
    'load_models': 'Load Models',
    'compile_models': 'Compile Models',
})

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

    backtrack = 1
    generated_ids = outputs[0][input_ids.shape[1]-backtrack:] #
    print(f" ~ | generated_ids: {generated_ids.shape[0]} tokens\n")

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

def run_tts_paragraph(input_text: str, prompt_wav: torch.Tensor = None):
    print(f"\n ~ | Splitting paragraph:\n{input_text}\n")
    sentences = list(split_into_segments(input_text))
    print(f" ~ | Found {len(sentences)} sentences.\n")

    # Pre-tokenize prompt wav into <|s_###|> tokens once
    speech_ids_prefix = []
    if prompt_wav is not None:
        vq_code_prompt = xcodec.encode_code(input_waveform=prompt_wav)
        print(f" ~ | prompt_wav_shape: {vq_code_prompt.shape}")   
        vq_code_prompt = vq_code_prompt[0, 0, :]
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    apply_chat_template = tokenizer.apply_chat_template

    convert_tokens_to_ids = tokenizer.convert_tokens_to_ids

    model_generate = model.generate

    batch_decode = tokenizer.batch_decode

    torch_tensor = torch.tensor

    for i, sentence in enumerate(sentences):
        print(f"\n ~ | Generating sentence {i + 1}/{len(sentences)}: {sentence}")

        #sentence = "Mmm..." + sentence
        
        # Inject speech prefix into chat
        chat = [
            {"role": "user", "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{sentence}<|TEXT_UNDERSTANDING_END|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]

        input_ids = apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True,
            padding=True
        ).to("cuda")

        speech_end_id = convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        outputs = model_generate(
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
        decoded_speech_ids = batch_decode(generated_ids, skip_special_tokens=True)
        speech_ids = extract_speech_ids(decoded_speech_ids)
        out_speech_tokens = torch_tensor(speech_ids).unsqueeze(0).unsqueeze(0).to("cuda")

        if out_speech_tokens.shape[-1] == 0:
            print(" ~ | WARNING: No speech tokens generated.")
            continue

        with torch.no_grad():
            gen_wav = xcodec.decode_code(out_speech_tokens)
            save_audio(gen_wav, sentence)  # Will save with timestamp and text

# --- batched/document TTS with progress bars ---
def run_tts_document(input_text: str, prompt_wav: torch.Tensor = None):
    global tokenizer, model, xcodec
    segments = list(split_into_segments(doc_text))
    #pbar = tqdm(total=len(segments), desc="Mika TTS", unit="seg") if tqdm else None
    for seg in segments:
        run_tts_pipeline(input_text=seg, prompt_wav=prompt_wav)


wrap_progress(globals(), {
    'run_tts_pipeline': 'TTS Sentence Pipeline',
    'run_tts_paragraph': 'TTS Paragragh Pipeline',
    'run_tts_document': 'TTS Document Pipeline',
})


# =========================================
# EXPERIMENTAL

def _decode_segment(sentence: str,
                    speech_ids_prefix_tokens: List[str]) -> np.ndarray:
    """
    Generate one sentence and return mono float32 waveform at 16kHz (numpy 1D).
    Returns empty array if no speech tokens were produced.
    """
    chat = [
        {"role": "user",
         "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{sentence}<|TEXT_UNDERSTANDING_END|>"},
        {"role": "assistant",
         "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix_tokens)}
    ]

    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, return_tensors='pt',
        continue_final_message=True, padding=True
    ).to("cuda")

    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
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
    if not speech_ids:
        return np.array([], dtype=np.float32)

    out_speech_tokens = torch.tensor(speech_ids).unsqueeze(0).unsqueeze(0).to("cuda")
    with torch.no_grad():
        gen_wav = xcodec.decode_code(out_speech_tokens)   # [1,1,T]
    return gen_wav[0, 0, :].float().cpu().numpy()         # 16kHz mono

def _append_with_gap_xfade(base: np.ndarray,
                           add: np.ndarray,
                           sr: int = 16000,
                           gap_s: float = 0.15,
                           xfade_s: float = 0.02) -> np.ndarray:
    """
    Concatenate base + (gap) + add, with small crossfade to avoid clicks.
    """
    if add.size == 0:
        return base

    if base.size == 0:
        # just prefix with a tiny fade-in
        if xfade_s > 0:
            n = min(len(add), int(sr * xfade_s))
            if n > 0:
                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                add = add.copy()
                add[:n] *= ramp
        return add

    gap = np.zeros(int(sr * max(gap_s, 0.0)), dtype=np.float32)

    if xfade_s <= 0.0:
        return np.concatenate([base, gap, add])

    xN = min(len(base), len(add), int(sr * xfade_s))
    if xN <= 0:
        return np.concatenate([base, gap, add])

    # linear crossfade on the boundary between base and (gap+add)
    base_tail = base[-xN:].copy()
    add_head = add[:xN].copy()
    fade_out = np.linspace(1.0, 0.0, xN, dtype=np.float32)
    fade_in  = 1.0 - fade_out
    xfade = base_tail * fade_out + add_head * fade_in

    return np.concatenate([base[:-xN], xfade, add[xN:]])


def run_tts_document_joined(
    doc_text: str,
    prompt_wav: torch.Tensor = None,
    out_path: Path = Path("data/generated/mika/kagami_joined.wav"),
    gap_s: float = 0.15,
    xfade_s: float = 0.02,
    *,
    adjust: bool = False,
    stretch_rate: float = 0.98,
    save_48k: bool = True,
    target_sr: int = 48000,
):
    # ... generate `joined` at 16kHz ...

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, joined, 16000)
    print(f"Saved joined 16kHz: {out_path}")

    if save_48k:
        y_48k = librosa.resample(joined, orig_sr=16000, target_sr=target_sr)
        if adjust:
            y_48k = librosa.effects.time_stretch(y_48k, rate=stretch_rate)
            out48 = out_path.with_name(out_path.stem + f"_adjusted_{target_sr//1000}k" + out_path.suffix)
        else:
            out48 = out_path.with_name(out_path.stem + f"_{target_sr//1000}k" + out_path.suffix)
        sf.write(out48, y_48k, target_sr)
        print(f"Saved joined {target_sr//1000}kHz{' adjusted' if adjust else ''}: {out48}")


# =========================================


# --- near-real-time streaming pipeline (LLM -> TTS) ---
class StreamingTTSPipeline:
    def __init__(self, prompt_wav: torch.Tensor = None, max_queue: int = 32):
        self.q = queue.Queue(maxsize=max_queue)
        self.prompt_wav = prompt_wav
        self._stop = False
        self._thr = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._thr.start()

    def submit_text(self, text: str):
        self.q.put(text)

    def close(self):
        self._stop = True
        self.q.join()
        self._thr.join(timeout=1)

    def _worker(self):
        buf = []
        def flush_sentences():
            if not buf:
                return
            text = "".join(buf)
            for seg in split_into_segments(text):
                run_tts_pipeline(input_text=seg, prompt_wav=self.prompt_wav)
            buf.clear()

        while not (self._stop and self.q.empty()):
            try:
                chunk = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:  # flush signal
                flush_sentences()
                self.q.task_done()
                continue
            buf.append(chunk)
            # flush when we see a likely sentence boundary to keep latency low
            if any(chunk.endswith(x) for x in (". ", "! ", "? ", "\n")):
                flush_sentences()
            self.q.task_done()
        flush_sentences()



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
    enable_load_sample_audio = False

    prompt_wav = None
    if enable_load_sample_audio:
        prompt_wav, _ = sf.read("test.wav")
        if allow_cuda:
            prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0).to("cuda")
        else:
            prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)#.to("cuda")

    #prompt_wav, sr = sf.read("mika_actually_people.wav")  # sr should be 16000

    # Get last 2 seconds worth of samples
    #samples_per_second = sr
    #last_2_sec = prompt_wav[-2 * samples_per_second:]

    # Save to new file
    #sf.write("mika_actually_people_tail.wav", last_2_sec, sr)
    #print("Saved last 2 seconds to: mika_actually_people_tail.wav")

    #run_tts_paragraph(input_text=kagami_speech2, prompt_wav=prompt_wav)

    #run_tts_pipeline(input_text=kagami_speech2, prompt_wav=prompt_wav)

    #run_tts_pipeline(input_text="Wanna know a secret? The secret ingredient is love! Just kidding! It's actually people!", prompt_wav=prompt_wav)
    
    #run_tts_pipeline(input_text="We will not go quietly into the night! We will go loudly!", prompt_wav=prompt_wav)
    
    #run_tts_pipeline(input_text="Zerglings come in pairs! Just like kidneys, and best friends!", prompt_wav=prompt_wav)
    # Run on multiple lines or a single one
    for line in mika_test_lines:
        run_tts_pipeline(input_text=line, prompt_wav=prompt_wav)
        #run_tts_pipeline(input_text=line, prompt_wav=None)


if __name__ == "__main__":
    main()
