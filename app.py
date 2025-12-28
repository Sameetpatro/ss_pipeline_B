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
from IndicTransToolkit import IndicProcessor
from gtts import gTTS

from indic_transliteration.sanscript import transliterate

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

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
translation_models = {}
whisper_lock = threading.Lock()

# --------------------------------------------------
# Language maps (UPDATED)
# --------------------------------------------------
INDIC_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "mr": "mar_Deva",   
    "od": "ory_Orya",
    "ta": "tel_Telu",
    "te": "tam_Taml",
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
    logger.info("Loading Whisper small...")

    whisper_model = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
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
# IndicTrans2 Translation
# --------------------------------------------------
def get_translation_model(src, tgt):
    key = f"{src}-{tgt}"
    if key in translation_models:
        return translation_models[key]

    if src == "en":
        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
    elif tgt == "en":
        model_name = "ai4bharat/indictrans2-indic-en-dist-320M"
    else:
        model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"

    logger.info(f"Loading IndicTrans2: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if device == "cuda":
        model = model.to(device)

    model.eval()
    processor = IndicProcessor(inference=True)

    translation_models[key] = (model, tokenizer, processor)
    return model, tokenizer, processor


def translate_text(text, src, tgt):
    model, tokenizer, processor = get_translation_model(src, tgt)

    batch = processor.preprocess_batch(
        [text],
        src_lang=INDIC_LANG_MAP[src],
        tgt_lang=INDIC_LANG_MAP[tgt],
    )

    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return processor.postprocess_batch(decoded, lang=INDIC_LANG_MAP[tgt])[0]

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
# TTS (gTTS)
# --------------------------------------------------
def generate_speech(text, lang, out_path):
    if lang in GTTS_LANG_MAP:
        tts = gTTS(text=text, lang=GTTS_LANG_MAP[lang])
        tts.save(out_path)
        return out_path

    if lang == "od":
        roman = odia_to_roman(text)
        logger.info(f"Odia Romanized: {roman}")
        tts = gTTS(text=roman, lang="en")
        tts.save(out_path)
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

        if src not in INDIC_LANG_MAP or tgt not in INDIC_LANG_MAP:
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
    app.run(host="0.0.0.0", port=5000, debug=False)
