import streamlit as st
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pyttsx3

# -- Page config first -- #
st.set_page_config(page_title="AI Language Translator", layout="centered")

# -- UI Language Selection for localization -- #
ui_lang = st.radio("Chagua lugha (Choose UI language):", ["English", "Kiswahili"])
texts = {
    "English": {
        "title": "🌍 AI Language Translator",
        "source_label": "Source Language",
        "target_label": "Target Language",
        "enter_text": "Enter text:",
        "translate_btn": "Translate",
        "warning_empty": "Please enter some text to translate...",
        "expander": "Translation & Paraphrases",
        "direct_sub": "Direct Translation",
        "paraphrase_sub": "Paraphrased Versions",
        "tts_btn": "🔊 Play TTS",
        "footer": "Designed by Baramia_Abdul-salami • Baxfort Technology"
    },
    "Kiswahili": {
        "title": "🌍 Mtafsiri wa Lugha kwa AI",
        "source_label": "Lugha ya Chanzo",
        "target_label": "Lugha ya Lengo",
        "enter_text": "Weka maandishi:",
        "translate_btn": "Tafsiri",
        "warning_empty": "Tafadhali andika maandishi ya kutafsiri...",
        "expander": "Tafsiri & Matoleo Mbadala",
        "direct_sub": "Tafsiri ya Moja kwa Moja",
        "paraphrase_sub": "Toleo Mbadala za Uandishi",
        "tts_btn": "🔊 Cheza TTS",
        "footer": "Imetengenezwa na Baramia_Abdul-salami • Baxfort Technology"
    }
}
lang_text = texts[ui_lang]

# -- Load paraphrasing model (cached) -- #
@st.cache_resource
def load_paraphraser():
    model_name = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_paraphraser()

def paraphrase_text(text, num_return_sequences=2, num_beams=5):
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def translate_text(text, source, target):
    try:
        return GoogleTranslator(source=source.lower(), target=target.lower()).translate(text)
    except Exception as e:
        return f"Error: {str(e)}"

def text_to_speech_local(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# -- UI CONFIG -- #
st.title(lang_text['title'])

# Source and Target language selection
language_options = {
    "en": "🇬🇧 English",
    "sw": "🇰🇪 Swahili",
    "fr": "🇫🇷 French",
    "de": "🇩🇪 German",
    "ar": "🇸🇦 Arabic"
}
lang_keys = list(language_options.keys())
source_lang = st.selectbox(lang_text['source_label'], lang_keys, format_func=lambda x: language_options[x])
target_lang = st.selectbox(lang_text['target_label'], lang_keys, format_func=lambda x: language_options[x])

# Text input
text_to_translate = st.text_area(lang_text['enter_text'], height=150)

# Green button style
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5em 1.5em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Translate & Paraphrase -- #
if st.button(lang_text['translate_btn']):
    if not text_to_translate.strip():
        st.warning(lang_text['warning_empty'])
    else:
        result = translate_text(text_to_translate, source_lang, target_lang)
        with st.expander(lang_text['expander']):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(lang_text['direct_sub'])
                st.code(result)
                if st.button(lang_text['tts_btn'], key="tts_local"):
                    text_to_speech_local(result)
            with col2:
                st.subheader(lang_text['paraphrase_sub'])
                try:
                    variations = paraphrase_text(result)
                    for i, p in enumerate(variations, 1):
                        st.code(f"{i}. {p}")
                except Exception as e:
                    st.warning(f"Paraphrasing failed: {e}")

# -- Footer -- #
st.markdown("---")
st.markdown(
    f"<p style='text-align: center;'>{lang_text['footer']}</p>", unsafe_allow_html=True
)
