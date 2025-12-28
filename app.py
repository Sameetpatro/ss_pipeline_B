from flask import Flask, request, jsonify, send_file
import os
import torch
import threading
import subprocess
import shlex
import traceback
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from indic_transliteration.sanscript import transliterate

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Paths
# --------------------------------------------------
UPLOAD_FOLDER = "uploads"
GENERATED_AUDIO_FOLDER = "generated_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_AUDIO_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "m4a", "flac"}

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --------------------------------------------------
# Globals
# --------------------------------------------------
whisper_model = None
nllb_model = None
nllb_tokenizer = None
whisper_lock = threading.Lock()

# --------------------------------------------------
# Language maps (FINAL, CORRECT)
# --------------------------------------------------
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "od": "ory_Orya",
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "gu": "guj_Gujr",
}

GTTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "mr": "mr",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "gu": "gu",
}

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav_mono_16k(src):
    base = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(UPLOAD_FOLDER, f"{base}_16k.wav")

    cmd = f'ffmpeg -y -i "{src}" -ac 1 -ar 16000 -acodec pcm_s16le "{out}"'
    try:
        subprocess.run(
            shlex.split(cmd),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return out
    except Exception:
        logger.warning("FFmpeg failed, using original audio")
        return src

# --------------------------------------------------
# Whisper ASR
# --------------------------------------------------
def load_whisper_model():
    global whisper_model
    logger.info("Loading Whisper medium...")

    whisper_model = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-medium",
        device=0 if device == "cuda" else -1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        model_kwargs={"low_cpu_mem_usage": True},
    )

    whisper_model.model.eval()
    torch.set_grad_enabled(False)
    logger.info("Whisper ready")


def transcribe_audio(path):
    with whisper_lock:
        result = whisper_model(path, return_timestamps=False)
    return result["text"].strip()

# --------------------------------------------------
# NLLB Translation
# --------------------------------------------------
def load_nllb_model():
    global nllb_model, nllb_tokenizer
    logger.info("Loading NLLB-200 distilled model...")

    model_name = "facebook/nllb-200-distilled-600M"

    nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if device == "cuda":
        nllb_model = nllb_model.to(device)

    nllb_model.eval()
    logger.info("NLLB ready")


def translate_text(text, src, tgt):
    src_lang = NLLB_LANG_MAP[src]
    tgt_lang = NLLB_LANG_MAP[tgt]

    nllb_tokenizer.src_lang = src_lang

    inputs = nllb_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    forced_bos_token_id = nllb_tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        outputs = nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=5,
        )

    return nllb_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# --------------------------------------------------
# Odia Romanization
# --------------------------------------------------
def odia_to_roman(text):
    roman = transliterate(text, "oriya", "itrans")
    roman = (
        roman.replace("M", "n")
             .replace("A", "a")
             .replace(".", "")
    )
    return roman

# --------------------------------------------------
# TTS
# --------------------------------------------------
def generate_speech(text, lang, out_path):
    if lang in GTTS_LANG_MAP:
        gTTS(text=text, lang=GTTS_LANG_MAP[lang]).save(out_path)
        return out_path

    if lang == "od":
        roman = odia_to_roman(text)
        gTTS(text=roman, lang="en").save(out_path)
        return out_path

    raise ValueError("Unsupported TTS language")

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/translate", methods=["POST"])
def translate_api():
    try:
        start = datetime.now()

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        src = request.form.get("input_lang")
        tgt = request.form.get("target_lang")

        if src not in NLLB_LANG_MAP or tgt not in NLLB_LANG_MAP:
            return jsonify({"error": "Unsupported language"}), 400

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload = os.path.join(
            UPLOAD_FOLDER, f"{stamp}_{secure_filename(file.filename)}"
        )
        file.save(upload)

        audio = convert_to_wav_mono_16k(upload)
        text = transcribe_audio(audio)
        translated = translate_text(text, src, tgt)

        out_audio = os.path.join(GENERATED_AUDIO_FOLDER, f"{stamp}_out.mp3")
        generate_speech(translated, tgt, out_audio)

        return jsonify({
            "recognized_text": text,
            "translated_text": translated,
            "audio_url": f"/audio/{os.path.basename(out_audio)}",
            "processing_time": (datetime.now() - start).total_seconds(),
        })

    except Exception:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal error"}), 500


@app.route("/audio/<name>")
def serve_audio(name):
    return send_file(
        os.path.join(GENERATED_AUDIO_FOLDER, name),
        mimetype="audio/mpeg"
    )

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    load_whisper_model()
    load_nllb_model()
    app.run(host="0.0.0.0", port=8000, debug=False)
