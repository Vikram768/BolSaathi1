from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# 🔥 PERFORMANCE SETTINGS
torch.set_num_threads(8)  # CPU threads (adjust karo)

model_name = "facebook/nllb-200-distilled-600M"

# ✅ LOAD ONCE
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32   # stable + fast CPU
)

model.eval()  # 🔥 important

# 🌍 Language mapping
lang_map = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "sd": "snd_Arab",
    "sa": "san_Deva",

    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "sv": "swe_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "da": "dan_Latn",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "uk": "ukr_Cyrl",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "bg": "bul_Cyrl",
    "sr": "srp_Cyrl",
    "hr": "hrv_Latn",
    "ms": "zsm_Latn",
    "id": "ind_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai"
}

# 📥 Request schema
class TranslateRequest(BaseModel):
    text: str
    target: str
    source: str | None = None

# 🚀 TRANSLATE FUNCTION (FAST)
def fast_translate(text, target_lang):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128   # 🔥 speed boost
    )

    with torch.no_grad():  # 🔥 huge speed gain
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            max_length=128,
            num_beams=1,   # 🔥 fast (no beam search)
            early_stopping=True
        )

    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

# 🚀 API
@app.post("/translate")
def translate(req: TranslateRequest):

    target_lang = lang_map.get(req.target, "eng_Latn")

    translated_text = fast_translate(req.text, target_lang)

    return {
        "translated": translated_text,
        "detected_source": req.source or "auto"
    }