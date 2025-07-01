import streamlit as st
import nltk
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

st.set_page_config(page_title="Duolingo Story Editor", layout="wide")

# --------------------
# Load T5 Model
# --------------------
@st.cache_resource
def load_t5_model():
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    return tokenizer, model

tokenizer, model = load_t5_model()

# --------------------
# Rule-Based Dicts
# --------------------
ALLOWED_POS = {
    'verb': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'noun': {'NN', 'NNS', 'NNP', 'NNPS'},
    'adj': {'JJ', 'JJR', 'JJS'},
    'adv': {'RB', 'RBR', 'RBS'},
}

CEFR_SIMPLE_WORDS = {
    "commence": ("start", "verb"),
    "terminate": ("end", "verb"),
    "consume": ("eat", "verb"),
    "purchase": ("buy", "verb"),
    "assist": ("help", "verb"),
    "construct": ("build", "verb"),
    "utilize": ("use", "verb"),
    "individual": ("person", "noun"),
    "requirement": ("need", "noun"),
    "obtain": ("get", "verb"),
    "approximately": ("about", "adv"),
    "endeavor": ("try", "verb"),
    "inclement": ("bad", "adj"),
}

CEFR_COMPLEX_WORDS = {v[0]: (k, v[1]) for k, v in CEFR_SIMPLE_WORDS.items()}

# --------------------
# Functions
# --------------------
def score_difficulty(text):
    return {
        'Flesch Reading Ease': round(textstat.flesch_reading_ease(text), 2),
        'Flesch-Kincaid Grade': round(textstat.flesch_kincaid_grade(text), 2),
        'Avg Sentence Length': round(textstat.avg_sentence_length(text), 2),
        'Word Count': textstat.lexicon_count(text),
        'Syllables per Word': round(textstat.syllable_count(text) / max(1, textstat.lexicon_count(text)), 2),
        'Difficult Word Ratio': round(textstat.difficult_words(text) / max(1, textstat.lexicon_count(text)), 2),
    }

def simplify_text_rule(text, replacement_dict=CEFR_SIMPLE_WORDS):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    simplified = []
    for word, tag in pos_tags:
        lower = word.lower()
        if lower in replacement_dict:
            simple_word, expected_pos = replacement_dict[lower]
            if tag in ALLOWED_POS.get(expected_pos, set()):
                simple_word = simple_word.capitalize() if word[0].isupper() else simple_word
                simplified.append(simple_word)
                continue
        simplified.append(word)
    return ' '.join(simplified)

def complexify_text(text, replacement_dict=CEFR_COMPLEX_WORDS):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    complexified = []
    for word, tag in pos_tags:
        lower = word.lower()
        if lower in replacement_dict:
            complex_word, expected_pos = replacement_dict[lower]
            if tag in ALLOWED_POS.get(expected_pos, set()):
                complex_word = complex_word.capitalize() if word[0].isupper() else complex_word
                complexified.append(complex_word)
                continue
        complexified.append(word)
    return ' '.join(complexified)

def rewrite_text_t5(text, mode):
    prefix = 'simplify: ' if mode == 'simplify' else 'paraphrase: '
    input_text = prefix + text
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --------------------
# Streamlit UI
# --------------------
st.title("📚 Duolingo Story Difficulty Analyzer & Rewriter")

st.markdown("This app measures the difficulty of a story, and allows you to simplify or complexify it using rule-based logic or a transformer model like `T5`.")

with st.expander("📸 Show Example Story"):
    st.image("https://cdn.openai.com/dall-e-2/demos/text2img/1.png", use_column_width=True)

story = st.text_area("✍️ Enter your story:", 
"""Although the weather was inclement, she decided to commence her journey, undeterred by the storm’s fury. She endeavored to reach her destination despite the challenges.""")

if st.button("Analyze & Rewrite"):
    col1, col2,col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Original")
        st.write(story)
        st.metric("Flesch Reading Ease", score_difficulty(story)['Flesch Reading Ease'])
        st.json(score_difficulty(story))

    with col2:
        st.subheader("🟢 Simplified")
        simplified_rule = simplify_text_rule(story)
        st.write(simplified_rule)
        st.metric("Flesch Reading Ease", score_difficulty(simplified_rule)['Flesch Reading Ease'])
        
    with col3:
        st.subheader("🔴 Complexified")
        compli_rule = complexify_text(story)
        st.write(compli_rule)
        st.metric("Flesch Reading Ease", score_difficulty(compli_rule)['Flesch Reading Ease'])

    # st.subheader("🤖 Simplified (Transformer)")
    # simplified_t5 = rewrite_text_t5(story, mode='simplify')
    # st.code(simplified_t5)
    # st.json(score_difficulty(simplified_t5))

    # st.subheader("🔴 Complexified (Transformer)")
    # complexified = rewrite_text_t5(story, mode='paraphrase')
    # st.code(complexified)
    # st.json(score_difficulty(complexified))

   
