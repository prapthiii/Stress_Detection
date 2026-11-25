import os
import html
import io
import json
import re
import csv
import uuid
import hashlib
import textwrap
from datetime import datetime
from urllib.parse import quote
import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.preprocessing import Preprocessor
from src.suggestions import SupportSuggestions
from src.evaluation import Evaluator
from src.questionnaire import Questionnaire
from src.ocr_utils import extract_text_from_image, EASYOCR_AVAILABLE, TESSERACT_AVAILABLE
from src.feature_extraction import FeatureExtractor
from src.classification import StressClassifier
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

ocr_available = EASYOCR_AVAILABLE or TESSERACT_AVAILABLE
ocr_detail = "EasyOCR" if EASYOCR_AVAILABLE else "Tesseract" if TESSERACT_AVAILABLE else "None"
CRISIS_KEYWORDS = [
    "suicide",
    "die",
    "kill myself",
    "dont want to live",
    "don't want to live",
    "ending it",
    "no reason to live",
]
PROFILE_STORE = os.path.join("data", "user_profiles.json")


def load_profile_store():
    if not os.path.exists(PROFILE_STORE):
        return {"users": {}, "history": []}
    try:
        with open(PROFILE_STORE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {"users": {}, "history": []}
    data.setdefault("users", {})
    data.setdefault("history", [])
    return data


def save_profile_store(data):
    os.makedirs(os.path.dirname(PROFILE_STORE), exist_ok=True)
    with open(PROFILE_STORE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def normalize_profile_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()
    return cleaned or "Guest"


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def generate_profile_id():
    return uuid.uuid4().hex


def register_profile(name, password_hash, profile_id=None):
    profile_id = profile_id or generate_profile_id()
    store = load_profile_store()
    store["users"].setdefault(profile_id, {
        "name": name,
        "password_hash": password_hash,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_profile_store(store)
    return profile_id


def find_profiles_by_name(name):
    store = load_profile_store()
    return [profile_id for profile_id, profile in store["users"].items() if profile.get("name") == name]


def get_profile_by_credentials(name, password):
    password_hash = hash_password(password or "")
    store = load_profile_store()
    for profile_id, profile in store["users"].items():
        if profile.get("name") == name and profile.get("password_hash") == password_hash:
            return profile_id
    return None


def append_history_record(record):
    store = load_profile_store()
    store["history"].append(record)
    save_profile_store(store)


def get_user_history(profile_id=None):
    history = load_profile_store()["history"]
    if profile_id:
        return [item for item in history if item.get("user_id") == profile_id]
    return history


def update_history_record(index, updates):
    store = load_profile_store()
    history = store["history"]
    if 0 <= index < len(history):
        history[index].update(updates)
        save_profile_store(store)
        return True
    return False


def generate_history_csv(history):
    output = io.StringIO()
    writer = csv.writer(output)
    header = [
        "User",
        "Module",
        "Stress Level",
        "Stress Score",
        "Created At",
        "Clinical Notes",
    ]
    writer.writerow(header)
    for item in history:
        writer.writerow([
            item.get("user", "Guest"),
            item.get("source", ""),
            item.get("stress_level", ""),
            item.get("stress_score", ""),
            item.get("created_at", ""),
            item.get("clinical_note", ""),
        ])
    return output.getvalue()


def build_clinician_summary(history):
    lines = [
        "Clinician Summary Report",
        "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "",
        f"Total records: {len(history)}",
        "",
    ]
    for item in history:
        lines.extend([
            f"User: {item.get('user', 'Guest')}",
            f"Module: {item.get('source', '')}",
            f"Stress Level: {item.get('stress_level', '')}",
            f"Stress Score: {item.get('stress_score', '')}",
            f"Created At: {item.get('created_at', '')}",
            f"Clinical Notes: {item.get('clinical_note', '')}",
            "-" * 40,
        ])
    return "\n".join(lines)


def render_simple_table(rows, columns):
    if not rows:
        st.info("No saved rows available.")
        return
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body += f"<tr>{cells}</tr>"
    st.markdown(
        f"""
        <table class="simple-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{body}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def is_crisis_message(text):
    return any(keyword in text.lower() for keyword in CRISIS_KEYWORDS)

def extract_features(audio, sr):
    """Return the current baseline voice feature vector.

    This is the trained application baseline. It is NOT conventional MFCC
    extraction. The implementation currently performs these exact operations:

    1. Normalize the incoming waveform by dividing by max(abs(audio)).
    2. Split the signal into 13 equal-position segment-energy coefficients using
       coefficient_i = mean(audio[i::13]) for i = 0..12.
    3. Append energy = mean(audio**2) / len(audio)
    4. Append zero-crossing rate zcr = mean(abs(diff(sign(audio))))

    The final vector is therefore a 15-dimensional segment-energy coefficient
    representation: 13 coefficients + energy + ZCR.
    """
    # Normalize
    audio = audio.astype(float)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    # Segment-energy coefficients (audio's equivalent of the legacy pipeline)
    coeffs = []
    for i in range(13):
        coeffs.append(np.mean(audio[i::13]))
    coeffs = np.array(coeffs)

    # Extra features
    energy = np.sum(audio**2) / len(audio)
    zcr = np.mean(np.abs(np.diff(np.sign(audio))))

    return np.hstack([coeffs, energy, zcr])


def get_rule_based_chatbot_response(user_input, history):
    """Free offline chatbot support based on keywords, sentiment, and chat context."""
    input_lower = user_input.lower()

    if is_crisis_message(user_input):
        return (
            "I'm really sorry you're feeling this way. If you might be in immediate danger, "
            "please contact your local emergency number now or reach out to someone you trust "
            "who can stay with you. You deserve real support in this moment."
        )

    sentiment = sentiment_analyzer.polarity_scores(user_input)
    compound = sentiment["compound"]

    stress_keywords = ["stress", "stressed", "overwhelmed", "anxious", "anxiety", "worried", "tired", "exhausted", "pressure"]
    study_keywords = ["exam", "assignment", "deadline", "study", "marks", "college", "class", "project"]
    sleep_keywords = ["sleep", "insomnia", "can't sleep", "cant sleep", "woke up", "night"]
    anger_keywords = ["angry", "irritated", "annoyed", "frustrated"]
    positive_keywords = ["good", "better", "happy", "relaxed", "calm", "fine"]
    negative_keywords = ["bad", "sad", "depressed", "scared", "lonely", "low"]

    is_study_related = any(keyword in input_lower for keyword in study_keywords)
    is_sleep_related = any(keyword in input_lower for keyword in sleep_keywords)
    is_angry = any(keyword in input_lower for keyword in anger_keywords)
    is_stress_related = any(keyword in input_lower for keyword in stress_keywords)
    is_positive = any(keyword in input_lower for keyword in positive_keywords)
    is_negative = any(keyword in input_lower for keyword in negative_keywords)

    if is_study_related:
        responses = [
            "That sounds like academic pressure. Try choosing one small task you can finish in the next 20 minutes, then take a short break. What deadline is feeling most urgent right now?",
            "Study stress can feel huge when everything is mixed together. List the tasks, mark the top two, and start with the easiest first to build momentum.",
            "It makes sense to feel tense around exams or deadlines. Try 25 minutes of focus, 5 minutes of break, then repeat once. What subject or task is weighing on you most?"
        ]
    elif is_sleep_related:
        responses = [
            "Poor sleep can make stress feel much heavier. If you can, dim the screen, relax your shoulders, and breathe out slowly for a few rounds. Has this been happening often?",
            "That sounds exhausting. Try writing down tomorrow's worries in one place so your mind does not have to keep holding them tonight.",
            "Sleep stress is frustrating. A small reset may help: inhale for 4 seconds, exhale for 6 seconds, and repeat until your body starts to settle."
        ]
    elif is_angry:
        responses = [
            "That frustration makes sense. Before reacting, try taking a short pause and naming the exact thing that upset you. What happened?",
            "Anger can be a sign that something feels unfair or too much. Step away for two minutes if you can, then decide what you want to say calmly.",
            "I hear the frustration. Try unclenching your jaw and relaxing your shoulders first. What part of the situation feels most out of your control?"
        ]
    elif compound < -0.1 or is_stress_related or is_negative:
        responses = [
            "It sounds really heavy right now. Can you tell me what is making you feel this way?",
            "I'm sorry you're going through this. Let's make it smaller: what is the main thing on your mind right now?",
            "If you can, take one slow breath in and a longer breath out. Then tell me what feels hardest in this moment.",
            "You're doing your best in a tough moment. One small care step could be water, a short walk, or messaging someone you trust. Which feels possible?"
        ]
    elif compound > 0.3 or is_positive:
        responses = [
            "I'm glad to hear that. What helped you feel a little better?",
            "That is good progress. Try to notice what worked, so you can use it again next time.",
            "Nice. What is one small thing you can do today to keep that calmer feeling going?"
        ]
    else:
        responses = [
            "Thanks for sharing. What would you like help with right now: calming down, planning your tasks, or talking it through?",
            "I'm here with you. Tell me a little more about what has been happening today.",
            "That makes sense. If you had to rate your stress from 1 to 10 right now, what number would it be?"
        ]

    if history and any("stress" in message.lower() for _, message in history[-4:]):
        responses.extend([
            "Earlier you mentioned stress. Is it feeling better, worse, or about the same now?",
            "Since stress has been coming up, we can try a quick grounding step: name one thing you can see, one thing you can hear, and one thing you can feel."
        ])

    return np.random.choice(responses)


def get_chatbot_response(user_input, history):
    return get_rule_based_chatbot_response(user_input, history)

    sentiment = sentiment_analyzer.polarity_scores(user_input)
    compound = sentiment['compound']

    # Keywords
    input_lower = user_input.lower()
    crisis_keywords = ["suicide", "die", "kill myself", "dont want to live", "ending it", "no reason to live"]
    stress_keywords = ["stress", "stressed", "overwhelmed", "anxious", "worried", "tired", "exhausted"]
    positive_keywords = ["good", "better", "happy", "relaxed", "calm"]
    negative_keywords = ["bad", "sad", "depressed", "angry", "frustrated", "scared"]

    # Crisis detection first
    if any(kw in input_lower for kw in crisis_keywords):
        return np.random.choice([
            "I'm really sorry you're feeling this way. If you're feeling unsafe, please reach out to a trusted person right now or contact a crisis hotline.",
            "It sounds like you're going through a lot. If you can, please consider calling a local emergency number or a suicide prevention hotline. You deserve support.",
            "You don't have to handle this alone. If you're able, please connect with someone who can help you right now — a friend, family member, or a professional."
        ])

    # Determine tone
    if compound > 0.5:
        tone = "positive"
    elif compound > 0.1:
        tone = "mild_positive"
    elif compound > -0.1:
        tone = "neutral"
    elif compound > -0.5:
        tone = "mild_negative"
    else:
        tone = "negative"

    # Check for stress-related
    is_stress_related = any(kw in input_lower for kw in stress_keywords)
    is_positive = any(kw in input_lower for kw in positive_keywords)
    is_negative = any(kw in input_lower for kw in negative_keywords)

    # Core responses
    if tone == "negative" or is_stress_related:
        responses = [
            "It sounds really heavy right now. Can you tell me more about what's making you feel this way?",
            "I'm sorry you're going through this. What has been most difficult for you recently?",
            "Sometimes it helps to write down what's on your mind. Would you like to try that?",
            "If you can, take a moment to breathe deeply: inhale for 4 seconds, exhale for 6. How does that feel?",
            "It can help to reach out to someone you trust. Is there someone you could talk to right now?",
            "You're doing your best in a tough moment. What is one small thing you could do to care for yourself right now?"
        ]
    elif tone == "mild_negative":
        responses = [
            "It seems like you're carrying a lot. A short walk or some music might help—what sounds good to you?",
            "I hear you. Sometimes just putting the feeling into words helps. What's on your mind?",
            "You're doing your best, and that matters. Would you like a suggestion for a quick self-care step?",
            "If it feels okay, try listing one thing that eases you even a little. I'm here to listen."
        ]
    elif tone == "neutral":
        responses = [
            "Thanks for sharing. What else would you like to talk about?",
            "I appreciate you opening up. Is there something specific you'd like to explore together?",
            "If you want to go deeper on anything, I'm here to listen."
        ]
    elif tone == "mild_positive" or is_positive:
        responses = [
            "That's great to hear! What is one thing you enjoyed today?",
            "I'm glad you're feeling good. What's something you're proud of right now?",
            "Nice! It can help to savor the positive moments — what's one you remember from today?"
        ]
    else:
        responses = [
            "That sounds wonderful! What made you feel that way?",
            "I'm happy for you — would you like to share more about what's going well?",
            "Keep holding onto that feeling. What's one thing you can do to keep it going?"
        ]

    # Add follow-up if stress appeared earlier
    if history and any("stress" in msg.lower() for _, msg in history[-4:]):
        responses.extend([
            "Earlier you mentioned stress — how are you feeling about that now?",
            "Since we talked about stress before, do you want to try a calming technique together?"
        ])

    return np.random.choice(responses)


def stress_score_from_level(stress_level_num):
    return int(round((stress_level_num / 3) * 100))


def emotion_from_sentiment(compound):
    if compound >= 0.4:
        return "Positive / calm"
    if compound >= 0.05:
        return "Slightly positive"
    if compound > -0.05:
        return "Neutral"
    if compound > -0.4:
        return "Concerned / low mood"
    return "Distressed"


def voice_emotion_from_level(stress_level_num):
    return {
        0: "Calm voice pattern",
        1: "Mild tension indicators",
        2: "Moderate stress indicators",
        3: "High arousal / stress indicators",
    }.get(stress_level_num, "Not available")


def get_voice_indicators(audio, sr):
    duration = len(audio) / sr if sr else 0
    energy = float(np.mean(audio ** 2)) if len(audio) else 0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio))) if len(audio) else 0
    rms = librosa.feature.rms(y=audio)[0] if len(audio) else np.array([0])
    return {
        "Duration": f"{duration:.2f}s",
        "Energy": f"{energy:.4f}",
        "Voice variation": f"{float(np.std(rms)):.4f}",
        "Zero-crossing rate": f"{zcr:.4f}",
    }


def plot_waveform(audio, sr):
    duration = len(audio) / sr if sr else 0
    timeline = np.linspace(0, duration, num=len(audio)) if len(audio) else []
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(timeline, audio, color="#2563eb", linewidth=0.8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Voice Waveform")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def save_report_result(source, stress_level, stress_level_num, suggestion, score=None, emotions=None, details=None):
    if "report_results" not in st.session_state:
        st.session_state.report_results = {}

    result = {
        "source": source,
        "stress_level": stress_level,
        "stress_level_num": int(stress_level_num),
        "stress_score": int(score if score is not None else stress_score_from_level(stress_level_num)),
        "emotions": emotions or [],
        "recommendations": [suggestion],
        "details": details or {},
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.report_results[source] = result

    user_name = st.session_state.get("current_user", "Guest")
    user_id = st.session_state.get("current_user_id", "guest")
    signature = json.dumps(
        {
            "user_id": user_id,
            "user_name": user_name,
            "source": source,
            "stress_level": stress_level,
            "stress_score": result["stress_score"],
            "details": details or {},
        },
        sort_keys=True,
    )
    if st.session_state.get(f"last_saved_{source}") != signature:
        append_history_record({"user_id": user_id, "user_name": user_name, **result})
        st.session_state[f"last_saved_{source}"] = signature


def get_combined_report():
    results = list(st.session_state.get("report_results", {}).values())
    if not results:
        return None, []

    combined_score = int(round(np.mean([item["stress_score"] for item in results])))
    if combined_score <= 20:
        combined_level = "No Stress"
        combined_level_num = 0
    elif combined_score <= 45:
        combined_level = "Mild Stress"
        combined_level_num = 1
    elif combined_score <= 70:
        combined_level = "Moderate Stress"
        combined_level_num = 2
    else:
        combined_level = "High Stress"
        combined_level_num = 3

    combined = {
        "source": "Combined Smart Report",
        "stress_level": combined_level,
        "stress_level_num": combined_level_num,
        "stress_score": combined_score,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return combined, results


def build_email_summary(combined, results):
    lines = [
        "Stress Management Smart Report",
        f"Generated: {combined['created_at']}",
        "",
        f"Overall stress score: {combined['stress_score']}/100",
        f"Overall stress level: {combined['stress_level']}",
        "",
        "Module summary:",
    ]
    for item in results:
        emotions = ", ".join(item["emotions"]) if item["emotions"] else "Not available"
        lines.extend([
            f"- {item['source']}: {item['stress_level']} ({item['stress_score']}/100)",
            f"  Detected emotions: {emotions}",
            f"  Recommendation: {item['recommendations'][0]}",
        ])
    lines.extend([
        "",
        "Note: This report is generated by a demonstration stress-management system and is not a medical diagnosis.",
    ])
    return "\n".join(lines)


def create_report_pdf(combined, results):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.94, "Stress Management Smart Report", fontsize=20, weight="bold", color="#102a43")
        fig.text(0.08, 0.91, f"Generated: {combined['created_at']}", fontsize=10, color="#52606d")
        fig.text(0.08, 0.85, f"Overall Stress Score: {combined['stress_score']}/100", fontsize=15, weight="bold")
        fig.text(0.08, 0.82, f"Overall Stress Level: {combined['stress_level']}", fontsize=14)

        y = 0.76
        for item in results:
            fig.text(0.08, y, item["source"], fontsize=13, weight="bold", color="#1b4965")
            y -= 0.026
            fig.text(0.1, y, f"Stress: {item['stress_level']} ({item['stress_score']}/100)", fontsize=10)
            y -= 0.024
            emotions = ", ".join(item["emotions"]) if item["emotions"] else "Not available"
            fig.text(0.1, y, f"Detected emotions: {emotions}", fontsize=10)
            y -= 0.024
            wrapped = textwrap.wrap(f"Recommendation: {item['recommendations'][0]}", width=92)
            for line in wrapped[:4]:
                fig.text(0.1, y, line, fontsize=10)
                y -= 0.021
            y -= 0.02
            if y < 0.16:
                break

        fig.text(0.08, 0.08, "This report is not a medical diagnosis. Seek qualified support for persistent or severe distress.", fontsize=9, color="#52606d")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 5.5))
        sources = [item["source"] for item in results]
        scores = [item["stress_score"] for item in results]
        colors = ["#2a9d8f", "#2563eb", "#f59e0b", "#ef4444"][:len(results)]
        ax.bar(sources, scores, color=colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Stress score")
        ax.set_title("Stress Score by Module")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=15, ha="right")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()

# Page config
st.set_page_config(
    page_title="🧠 Stress Management System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    body {
        background-color: #f7f9fc;
    }

    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 14px;
        margin: 1rem 0;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 7px 18px rgba(0, 0, 0, 0.06);
    }

    .stress-level {
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .no-stress { background-color: #d4edda; color: #155724; }
    .mild-stress { background-color: #fff3cd; color: #856404; }
    .moderate-stress { background-color: #ffeaa7; color: #d68910; }
    .high-stress { background-color: #f8d7da; color: #721c24; }

    /* Chat styling */
    .chat-hero {
        background: linear-gradient(135deg, #102a43 0%, #1b4965 48%, #2a9d8f 100%);
        color: #ffffff;
        padding: 1.25rem 1.4rem;
        border-radius: 8px;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 14px 32px rgba(16, 42, 67, 0.20);
    }

    .chat-hero h2 {
        margin: 0 0 0.35rem 0;
        font-size: 1.75rem;
        letter-spacing: 0;
    }

    .chat-hero p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.98rem;
    }

    .support-status {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-left: 5px solid #2a9d8f;
        border-radius: 8px;
        padding: 0.8rem 0.95rem;
        margin-bottom: 1rem;
        color: #243b53;
    }

    .composer-panel {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
        margin-bottom: 1rem;
    }

    .section-label {
        color: #486581;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0.3rem 0 0.65rem 0;
    }

    .chat-container {
        background: #f8fafc;
        padding: 1.1rem;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
        margin-bottom: 1rem;
    }

    .empty-chat {
        background: #ffffff;
        border: 1px dashed #bcccdc;
        border-radius: 8px;
        padding: 1rem;
        color: #52606d;
        text-align: left;
    }

    .analysis-hero {
        background: linear-gradient(135deg, #0f172a 0%, #334155 55%, #0f766e 100%);
        color: #ffffff;
        padding: 1.25rem 1.4rem;
        border-radius: 8px;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
    }

    .analysis-hero h2 {
        margin: 0 0 0.35rem 0;
        font-size: 1.75rem;
        letter-spacing: 0;
    }

    .analysis-hero p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.98rem;
    }

    .analysis-panel {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
        margin-bottom: 1rem;
    }

    .text-preview {
        background: #f8fafc;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.9rem;
        min-height: 132px;
        color: #243b53;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .text-empty {
        color: #829ab1;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
        margin-top: 0.75rem;
    }

    .result-title {
        color: #102a43;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .simple-table {
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        overflow: hidden;
        font-size: 0.92rem;
    }

    .simple-table th {
        background: #f8fafc;
        color: #243b53;
        text-align: left;
        padding: 0.7rem;
        border-bottom: 1px solid #d9e2ec;
    }

    .simple-table td {
        padding: 0.65rem 0.7rem;
        border-bottom: 1px solid #edf2f7;
        color: #102a43;
    }

    .chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.85rem;
    }

    .chat-row.user {
        justify-content: flex-end;
    }

    .chat-bubble {
        max-width: min(76%, 780px);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(16, 42, 67, 0.10);
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .chat-bubble.user {
        background: #2563eb;
        color: #ffffff;
        border-bottom-right-radius: 2px;
    }

    .chat-bubble.bot {
        background: #ffffff;
        color: #102a43;
        border: 1px solid #d9e2ec;
        border-bottom-left-radius: 2px;
    }

    .chat-buttons {
        margin-bottom: 0.75rem;
    }

    .stButton>button {
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
prep = Preprocessor()
extractor = FeatureExtractor()
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
labels = {
    0: "No Stress",
    1: "Mild Stress",
    2: "Moderate Stress",
    3: "High Stress"
}
text_classifier = None
if os.path.exists("text_stress_model.pkl"):
    text_model = joblib.load("text_stress_model.pkl")
    text_classifier = StressClassifier(text_model)

if not ocr_available:
    st.sidebar.warning("Handwritten image OCR is unavailable. Install EasyOCR or Tesseract to use image-based text extraction.")
else:
    st.sidebar.success(f"OCR available: {ocr_detail}")

suggester = SupportSuggestions()
evaluator = Evaluator()
questionnaire = Questionnaire()

# Sidebar
st.sidebar.title("🧠 Stress Management System")
st.sidebar.markdown("---")
st.sidebar.markdown("### User Profile")
if "current_user" not in st.session_state:
    st.session_state.current_user = "Guest"
    st.session_state.current_user_id = "guest"

profile_name = normalize_profile_name(
    st.sidebar.text_input(
        "Profile name",
        value=st.session_state.current_user,
        key="profile_name_input",
    )
)
password = st.sidebar.text_input("Password", type="password", key="profile_password_input")
confirm_password = st.sidebar.text_input("Confirm password", type="password", key="profile_confirm_password_input")

if st.sidebar.button("Use Profile", use_container_width=True):
    if profile_name == "Guest":
        st.session_state.current_user = "Guest"
        st.session_state.current_user_id = "guest"
        st.sidebar.success("Active profile: Guest")
    elif not password:
        st.sidebar.error("Please enter a password for the profile.")
    elif find_profiles_by_name(profile_name) and not get_profile_by_credentials(profile_name, password):
        st.sidebar.error("This profile name already exists. Enter the correct password or choose a different name.")
    elif not find_profiles_by_name(profile_name) and password != confirm_password:
        st.sidebar.error("Passwords do not match. Please confirm the password.")
    else:
        profile_id = get_profile_by_credentials(profile_name, password)
        if profile_id:
            st.session_state.current_user = profile_name
            st.session_state.current_user_id = profile_id
            st.sidebar.success(f"Logged in as {profile_name}")
        else:
            new_id = register_profile(profile_name, hash_password(password))
            st.session_state.current_user = profile_name
            st.session_state.current_user_id = new_id
            st.sidebar.success(f"Created and logged in as {profile_name}")

st.sidebar.caption(f"Active: {st.session_state.current_user} ({st.session_state.current_user_id[:8]})")
st.sidebar.markdown("---")

analysis_type = st.sidebar.selectbox(
    "Choose Analysis Method",
    ["🏠 Home", "🎤 Voice Analysis", "📝 Text Analysis", "📊 Self Assessment", "📄 Smart Report", "🩺 Admin/Doctor View", "💬 Chatbot Support"],
    help="Select how you want to assess stress"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This system analyzes stress through multiple modalities: voice, text, and self-assessment questionnaires.")
st.sidebar.warning("This system is intended for research and educational purposes only and is not a clinical diagnostic tool.")

# Main content
if analysis_type == "🏠 Home":
    st.markdown('<h1 class="main-header">🧠 Welcome to Stress Management System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>🎤 Voice Analysis</h3>
        <p>Upload audio files to detect stress through speech patterns, pitch, and energy levels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>📝 Text Analysis</h3>
        <p>Analyze written text or extract text from images to identify emotional indicators.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
        <h3>📊 Self Assessment</h3>
        <p>Take an interactive questionnaire to evaluate your current stress levels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### How It Works")
    st.info("Choose your preferred analysis method from the sidebar. Each method provides personalized stress assessment and coping suggestions.")
    st.warning("Heuristic stress estimate: this research prototype is not a clinical diagnosis tool.")

elif analysis_type == "🎤 Voice Analysis":
    st.markdown(
        """
        <div class="analysis-hero">
            <h2>Voice Stress Analysis</h2>
            <p>Record your voice or upload a WAV file to review speech stress level, voice indicators, waveform, and recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns([1.1, 1])
    with upload_col:
        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Audio Input</div>', unsafe_allow_html=True)
        input_method = st.radio(
            "Input method",
            ["🎙️ Record with microphone", "📁 Upload WAV file"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if input_method == "🎙️ Record with microphone":
            uploaded_file = st.audio_input(
                "Record your voice",
                help="Click the microphone, speak naturally for 5–15 seconds (e.g. describe your day), then stop.",
            )
            input_name = "Microphone recording"
        else:
            uploaded_file = st.file_uploader(
                "Upload WAV recording",
                type="wav",
                help="Use a clear WAV recording with minimal background noise.",
            )
            input_name = uploaded_file.name if uploaded_file is not None else None
        if uploaded_file is not None:
            if input_method == "📁 Upload WAV file":
                st.audio(uploaded_file, format="audio/wav")
            st.caption(f"Source: {input_name} · Size: {uploaded_file.size / 1024:.1f} KB")
        else:
            st.info("Record your voice or upload a WAV file to begin voice stress analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">What This Checks</div>', unsafe_allow_html=True)
        st.write("Voice stress analysis uses patterns such as energy, variation, and signal changes to estimate stress level.")
        st.write("For better results, use a short recording in a quiet place.")
        st.write("Microphone recording needs browser permission and works on localhost or HTTPS.")
        st.markdown('</div>', unsafe_allow_html=True)

    y, sr = None, None
    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getvalue())
        # Check loudness before preprocessing, which normalizes quiet audio up to full scale
        raw_audio, _ = librosa.load("temp.wav", sr=None, mono=True)
        y, sr = prep.preprocess_voice("temp.wav")
        if len(raw_audio) == 0 or np.max(np.abs(raw_audio)) < 0.01:
            st.warning("No voice detected in the recording. Check that your microphone is working and not muted, then try again.")
            y = None
        elif len(y) < sr * 0.5:
            st.warning("The recording is too short. Please speak for at least a few seconds and try again.")
            y = None

    if y is not None:
        with st.spinner("Processing audio..."):
            features = extract_features(y, sr)
            features = scaler.transform([features])

            pred = model.predict(features)[0]
            stress_level = labels[pred]
            stress_level_num = pred
            suggestion = suggester.get_suggestion(stress_level_num)
            voice_indicators = get_voice_indicators(y, sr)
            detected_voice_emotion = voice_emotion_from_level(stress_level_num)
            save_report_result(
                "Voice Analysis",
                stress_level,
                stress_level_num,
                suggestion,
                emotions=[detected_voice_emotion],
                details={"Input": input_name, **voice_indicators},
            )

        result_col, chart_col = st.columns([1, 1.35])
        with result_col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Voice Result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stress-level {"no-stress" if stress_level_num == 0 else "mild-stress" if stress_level_num == 1 else "moderate-stress" if stress_level_num == 2 else "high-stress"}">{stress_level}</div>', unsafe_allow_html=True)
            st.metric("Stress Score", f"{stress_score_from_level(stress_level_num)}/100")
            st.write(f"**Detected voice pattern:** {detected_voice_emotion}")
            st.success(suggestion)
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col:
            st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Waveform</div>', unsafe_allow_html=True)
            st.pyplot(plot_waveform(y, sr), clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Voice Indicators</div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        for metric_col, (name, value) in zip(metric_cols, voice_indicators.items()):
            metric_col.metric(name, value)
        st.markdown('</div>', unsafe_allow_html=True)

elif analysis_type == "📝 Text Analysis":
    st.markdown(
        """
        <div class="analysis-hero">
            <h2>Text Stress Analysis</h2>
            <p>Analyze typed notes, messages, journal entries, or extracted image text for stress indicators.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, preview_col = st.columns([1.2, 1])
    text_input = ""

    with input_col:
        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Input Source</div>', unsafe_allow_html=True)
        input_type = st.segmented_control(
            "Input Source",
            ["Direct Text", "Image OCR"],
            default="Direct Text",
            label_visibility="collapsed",
        )

        if input_type == "Direct Text":
            text_input = st.text_area(
                "Text to analyze",
                height=220,
                placeholder="Type your thoughts, messages, journal note, or any text you want to analyze...",
                key="direct_text_input",
            )
        elif input_type == "Image OCR":
            if not ocr_available:
                st.warning("OCR is not available in this environment. Use direct text input instead.")
                st.info("Install EasyOCR or Tesseract, then restart the app to enable image text extraction.")
            else:
                uploaded_file = st.file_uploader(
                    "Upload a clear image",
                    type=["png", "jpg", "jpeg"],
                    help="Use a clear, well-lit image with readable text.",
                )

                if uploaded_file is not None:
                    with open("temp.png", "wb") as f:
                        f.write(uploaded_file.getvalue())

                    with st.spinner("Extracting text from image..."):
                        extracted = extract_text_from_image("temp.png")

                    if "[ERROR]" in extracted:
                        st.error(extracted)
                    elif len(extracted.strip()) == 0:
                        st.warning("No text could be extracted. Try a clearer image or type the text manually.")
                    else:
                        st.success(f"Extracted {len(extracted)} characters.")

                    text_input = st.text_area(
                        "Extracted text",
                        value=extracted,
                        height=220,
                        key="extracted_text_input",
                        help="You can edit the extracted text before analysis.",
                    )
                else:
                    st.info("Upload an image to extract text for analysis.")

        analyze_clicked = st.button("Analyze Text", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with preview_col:
        char_count = len(text_input)
        word_count = len(text_input.split()) if text_input.strip() else 0
        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Characters", char_count)
        metric_col2.metric("Words", word_count)
        preview_text = html.escape(text_input[:500])
        if text_input.strip() and len(text_input) > 500:
            preview_text += "..."
        if text_input.strip():
            st.markdown(f'<div class="text-preview">{preview_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="text-preview text-empty">Your text preview will appear here.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if text_input.strip() != "":
            with st.spinner("Analyzing text for stress indicators..."):

                # Crisis detection
                crisis_keywords = [
                    "suicide", "kill myself", "don't want to live",
                    "end my life", "die", "no reason to live"
                ]

                if any(k in text_input.lower() for k in crisis_keywords):
                    stress_level = "High Stress"
                    stress_level_num = 3
                    suggestion = "Please reach out to someone you trust or a mental health professional immediately."
                else:
                    text_feat = extractor.extract_text_features(text_input)

                    if text_classifier is not None:
                        stress_level, _ = text_classifier.classify(text_feat)
                        stress_level_num = list(text_classifier.labels.keys())[list(text_classifier.labels.values()).index(stress_level)]
                    else:
                        # Fallback: use sentiment thresholds if no text model is available
                        sentiment = sentiment_analyzer.polarity_scores(text_input)
                        compound = sentiment["compound"]
                        if compound >= 0.4:
                            stress_level_num = 0
                        elif compound >= 0.0:
                            stress_level_num = 1
                        elif compound >= -0.4:
                            stress_level_num = 2
                        else:
                            stress_level_num = 3
                        stress_level = labels[stress_level_num]
                suggestion = suggester.get_suggestion(stress_level_num)
                sentiment = sentiment_analyzer.polarity_scores(text_input)
                detected_emotion = emotion_from_sentiment(sentiment["compound"])
                save_report_result(
                    "Text Analysis",
                    stress_level,
                    stress_level_num,
                    suggestion,
                    emotions=[detected_emotion],
                    details={
                        "Input source": input_type,
                        "Words": len(text_input.split()),
                        "Characters": len(text_input),
                    },
                )

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Analysis Result</div>', unsafe_allow_html=True)
            result_col1, result_col2, result_col3 = st.columns([1.2, 0.8, 2])
            with result_col1:
                st.markdown(f'<div class="stress-level {"no-stress" if stress_level_num == 0 else "mild-stress" if stress_level_num == 1 else "moderate-stress" if stress_level_num == 2 else "high-stress"}">{stress_level}</div>', unsafe_allow_html=True)
            with result_col2:
                st.metric("Level", stress_level_num + 1, delta="out of 4")
            with result_col3:
                st.success(suggestion)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No text found!")

elif analysis_type == "📊 Self Assessment":
    st.markdown('<h2 class="main-header">📊 Self Assessment Questionnaire (PSS-10)</h2>', unsafe_allow_html=True)
    st.write("Complete this 10-question Perceived Stress Scale assessment based on how you felt during the last month.")

    questions = questionnaire.questions
    options = questionnaire.options
    scores = questionnaire.scores
    max_score = questionnaire.max_score

    st.markdown("### Please answer all questions:")
    
    answered_questions = 0
    
    for i, q in enumerate(questions):
        with st.expander(f"Question {i+1}: {q}", expanded=False):
            answer = st.radio(
                f"Select your answer for Q{i+1}",
                options,
                key=f"pss_q{i}",
                index=None,
                label_visibility="collapsed",
            )
            if answer is not None:
                answered_questions += 1

    progress = answered_questions / len(questions)
    st.progress(progress, text=f"Progress: {answered_questions}/{len(questions)} questions answered")

    if st.button("📈 Calculate Stress Level", type="primary", use_container_width=True):
        if answered_questions == len(questions):
            responses = [
                scores[options.index(st.session_state[f"pss_q{i}"])]
                for i in range(len(questions))
            ]
            total_score = questionnaire.calculate_score(responses)
            interpretation, stress_level_num = questionnaire.interpret_score(total_score)

            stress_level = labels[stress_level_num]
            suggestion = suggester.get_suggestion(stress_level_num)
            save_report_result(
                "Self Assessment",
                stress_level,
                stress_level_num,
                suggestion,
                score=int(round((total_score / max_score) * 100)),
                emotions=[interpretation],
                details={
                    "PSS-10 score": f"{total_score}/{max_score}",
                    "Interpretation": interpretation,
                },
            )

            # Results
            st.balloons()
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.metric("Total Score", f"{total_score}/{max_score}")
            
            with col2:
                st.markdown(f'<div class="stress-level {"no-stress" if stress_level_num == 0 else "mild-stress" if stress_level_num == 1 else "moderate-stress" if stress_level_num == 2 else "high-stress"}">{stress_level}</div>', unsafe_allow_html=True)
            
            with col3:
                st.metric("Stress Level", stress_level_num + 1, delta="out of 4")

            st.progress(total_score / max_score, text=f"Stress Indicator: {total_score/max_score:.1%}")
            st.success(f"🎯 {suggestion}")
            st.info(f"Interpretation: {interpretation}")
            
            # Additional features based on stress level
            if stress_level_num == 2:  # Moderate stress
                with st.expander("🫁 Breathing Exercise"):
                    st.write("Try this 4-7-8 breathing technique:")
                    st.code("""
1. Inhale quietly through your nose for 4 seconds.
2. Hold your breath for 7 seconds.
3. Exhale completely through your mouth for 8 seconds.
4. Repeat 4 times.
                    """, language=None)
            
            elif stress_level_num == 3:  # High stress
                st.markdown("### 🧘 Guided Meditation")
                st.write("Take a few minutes to relax with this simple guided meditation:")
                meditation_text = """
1. Find a quiet place and sit comfortably.
2. Close your eyes and take a deep breath in for 4 counts.
3. Hold for 4 counts.
4. Exhale slowly for 4 counts.
5. Focus on your breath and let go of tension.
6. Repeat for 5-10 minutes.
                """
                st.code(meditation_text, language=None)
                
                st.markdown("### 📝 Thought Journal")
                st.write("Writing about your thoughts can help process emotions:")
                thoughts = st.text_area("Share your thoughts...", height=150, placeholder="What's on your mind? How are you feeling?")
                if thoughts.strip():
                    response = "Thank you for sharing. Acknowledging your feelings is a positive step. Remember, seeking support is okay."
                    st.info(f"🤖 {response}")
            
            # Additional insights
            with st.expander("📊 Detailed Results"):
                st.write(f"**Score Breakdown:** {total_score} points out of {max_score} maximum")
                st.write("**Scoring:** Questions 4, 5, 7, and 8 are reverse-scored.")
                st.write(f"**Classification:** {stress_level}")
                if total_score >= 27:
                    st.warning("Consider professional support if stress persists.")
                else:
                    st.info("Great job managing stress! Continue positive habits.")
        else:
            st.error(f"Please answer all questions. {len(questions) - answered_questions} remaining.")

elif analysis_type == "📄 Smart Report":
    st.markdown(
        """
        <div class="analysis-hero">
            <h2>Smart Stress Report</h2>
            <p>Generate a structured summary with stress score, detected emotions, recommendations, charts, PDF download, and email-ready text.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    combined, report_results = get_combined_report()

    if not report_results:
        st.info("Run Voice Analysis, Text Analysis, or Self Assessment first. Your report will appear here after at least one result is available.")
    else:
        top1, top2, top3 = st.columns(3)
        top1.metric("Overall Stress Score", f"{combined['stress_score']}/100")
        top2.metric("Overall Stress Level", combined["stress_level"])
        top3.metric("Completed Modules", len(report_results))

        chart_col, summary_col = st.columns([1.35, 1])
        with chart_col:
            st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Stress Score Chart</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            modules = [item["source"] for item in report_results]
            scores = [item["stress_score"] for item in report_results]
            ax.bar(modules, scores, color=["#2a9d8f", "#2563eb", "#f59e0b", "#ef4444"][:len(modules)])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Stress score")
            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xticks(rotation=12, ha="right")
            st.pyplot(fig, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with summary_col:
            st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Detected Emotions</div>', unsafe_allow_html=True)
            for item in report_results:
                emotions = ", ".join(item["emotions"]) if item["emotions"] else "Not available"
                st.write(f"**{item['source']}**: {emotions}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Module Breakdown</div>', unsafe_allow_html=True)
        for item in report_results:
            with st.expander(f"{item['source']} - {item['stress_level']} ({item['stress_score']}/100)", expanded=True):
                st.write(f"**Stress level:** {item['stress_level']}")
                st.write(f"**Stress score:** {item['stress_score']}/100")
                emotions = ", ".join(item["emotions"]) if item["emotions"] else "Not available"
                st.write(f"**Detected emotions:** {emotions}")
                st.write(f"**Recommendation:** {item['recommendations'][0]}")
                if item["details"]:
                    st.write("**Details:**")
                    for key, value in item["details"].items():
                        st.write(f"- {key}: {value}")

        email_summary = build_email_summary(combined, report_results)
        pdf_bytes = create_report_pdf(combined, report_results)

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"stress_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with action_col2:
            st.download_button(
                "Download Email Summary",
                data=email_summary,
                file_name=f"stress_report_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        mailto_link = f"mailto:?subject={quote('Stress Management Smart Report')}&body={quote(email_summary)}"
        st.markdown(f"[Open Email Draft]({mailto_link})")
        st.caption("Reports are generated locally from the latest results in this app session.")

        user_history = get_user_history(st.session_state.get("current_user_id", "guest"))
        if user_history:
            with st.expander(f"My Saved Report History — {st.session_state.get('current_user', 'Guest')} ({st.session_state.get('current_user_id', 'guest')[:8]})"):
                rows = [
                    {
                        "Date": item.get("created_at", ""),
                        "Module": item.get("source", ""),
                        "Stress Level": item.get("stress_level", ""),
                        "Stress Score": item.get("stress_score", 0),
                    }
                    for item in user_history[-20:][::-1]
                ]
                render_simple_table(rows, ["Date", "Module", "Stress Level", "Stress Score"])

elif analysis_type == "🩺 Admin/Doctor View":
    st.markdown(
        """
        <div class="analysis-hero">
            <h2>Admin / Doctor Dashboard</h2>
            <p>Review saved user reports, stress trends, and high-stress alerts from local profile history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    admin_code = st.text_input("Access code", type="password", help="Demo access code: doctor123")
    if admin_code != "doctor123":
        st.info("Enter the admin/doctor access code to view saved report history.")
    else:
        all_history = get_user_history()
        if not all_history:
            st.warning("No saved profile history yet. Run analyses under user profiles first.")
        else:
            users = sorted({item.get("user_name", item.get("user", "Guest")) for item in all_history})
            selected_user = st.selectbox("User profile", ["All Users"] + users)
            filtered_history = (
                all_history
                if selected_user == "All Users"
                else [
                    item
                    for item in all_history
                    if item.get("user_name", item.get("user", "Guest")) == selected_user
                ]
            )

            total_reports = len(filtered_history)
            high_alerts = [item for item in filtered_history if item.get("stress_level_num", 0) >= 3 or item.get("stress_score", 0) >= 75]
            avg_score = int(round(np.mean([item.get("stress_score", 0) for item in filtered_history]))) if filtered_history else 0

            admin_col1, admin_col2, admin_col3 = st.columns(3)
            admin_col1.metric("Saved Results", total_reports)
            admin_col2.metric("Average Stress Score", f"{avg_score}/100")
            admin_col3.metric("High-Stress Alerts", len(high_alerts))

            csv_data = generate_history_csv(filtered_history)
            summary_text = build_clinician_summary(filtered_history)
            export_col1, export_col2 = st.columns(2)
            export_col1.download_button(
                "Export filtered history CSV",
                data=csv_data,
                file_name=f"clinician_history_{selected_user.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            export_col2.download_button(
                "Export summary report",
                data=summary_text,
                file_name=f"clinician_summary_{selected_user.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            history_indices = []
            trend_rows = []
            for index, item in enumerate(all_history):
                user_name = item.get("user_name", item.get("user", "Guest"))
                if selected_user == "All Users" or user_name == selected_user:
                    history_indices.append(index)
                    trend_rows.append(
                        {
                            "Index": len(history_indices),
                            "User": user_name,
                            "Module": item.get("source", ""),
                            "Stress Score": item.get("stress_score", 0),
                            "Stress Level": item.get("stress_level", ""),
                            "Clinical Notes": item.get("clinical_note", ""),
                            "Created At": item.get("created_at", ""),
                        }
                    )

            st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Stress Trend</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot([row["Index"] for row in trend_rows], [row["Stress Score"] for row in trend_rows], marker="o", color="#2563eb")
            ax.axhline(75, color="#ef4444", linestyle="--", linewidth=1, label="High-stress threshold")
            ax.set_ylim(0, 100)
            ax.set_xlabel("Saved result order")
            ax.set_ylabel("Stress score")
            ax.grid(alpha=0.25)
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if high_alerts:
                st.markdown('<div class="section-label">High-Stress Alerts</div>', unsafe_allow_html=True)
                for item in high_alerts[-5:][::-1]:
                    st.error(
                        f"{item.get('user', 'Guest')} · {item.get('source')} · "
                        f"{item.get('stress_level')} ({item.get('stress_score')}/100) · {item.get('created_at')}"
                    )
            else:
                st.success("No high-stress alerts in the selected history.")

            st.markdown('<div class="section-label">Saved Report History</div>', unsafe_allow_html=True)
            render_simple_table(trend_rows[-30:][::-1], ["Index", "User", "Module", "Stress Score", "Stress Level", "Clinical Notes", "Created At"])

            if history_indices:
                selected_report_index = st.selectbox(
                    "Select a report to add clinician notes",
                    [
                        f"{row['Index']} | {row['User']} | {row['Module']} | {row['Stress Level']} | {row['Created At']}"
                        for row in trend_rows
                    ],
                    index=0,
                )
                selected_row = trend_rows[[row['Index'] for row in trend_rows].index(int(selected_report_index.split(' | ')[0]))]
                selected_history_global_index = history_indices[selected_row['Index'] - 1]
                existing_note = all_history[selected_history_global_index].get('clinical_note', '')
                updated_note = st.text_area(
                    "Clinician notes / remarks",
                    value=existing_note,
                    height=180,
                )
                if st.button("Save clinician note", use_container_width=True):
                    if update_history_record(selected_history_global_index, {"clinical_note": updated_note}):
                        st.success("Clinician note saved.")
                    else:
                        st.error("Unable to save clinician note. Please try again.")

elif analysis_type == "💬 Chatbot Support":
    st.markdown(
        """
        <div class="chat-hero">
            <h2>Chatbot Support</h2>
            <p>Talk through stress, study pressure, sleep issues, or difficult feelings with private rule-based support.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    composer_col, conversation_col = st.columns([1.05, 1.45])
    with composer_col:
        st.markdown('<div class="composer-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Quick Support</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-buttons">', unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        btn_col3, btn_col4 = st.columns(2)
        btn_col5, clear_col = st.columns(2)
        if btn_col1.button("Breathing", icon="🧘", use_container_width=True):
            st.session_state.chat_history.append(("Bot", "Let's try a quick breathing exercise: inhale for 4 seconds, hold for 4 seconds, then exhale for 6 seconds. Repeat this 3 times and see how you feel."))
        if btn_col2.button("Grounding", icon="📝", use_container_width=True):
            st.session_state.chat_history.append(("Bot", "Try grounding: name 5 things you can see, 4 things you can touch, 3 sounds you hear, 2 things you can smell, and 1 thing you can taste."))
        if btn_col3.button("Journal", icon="✍️", use_container_width=True):
            st.session_state.chat_history.append(("Bot", "Writing can help you process emotions. Try writing for a few minutes about what you're feeling and why. No need to share it, just let it out."))
        if btn_col4.button("Affirm", icon="🧡", use_container_width=True):
            st.session_state.chat_history.append(("Bot", "You are doing your best in a tough moment, and that matters. It's okay to take things one step at a time."))
        if btn_col5.button("Mood", icon="🌤️", use_container_width=True):
            st.session_state.chat_history.append(("Bot", "How would you rate your mood on a scale of 1-10 right now? It's okay to be honest. I'm here to listen."))
        if clear_col.button("Clear", icon="🧹", type="secondary", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Message</div>', unsafe_allow_html=True)
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Share your thoughts",
                placeholder="Type what is happening, what you are feeling, or what you need help with...",
                key="chat_input",
                height=96,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send Message", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            response = get_chatbot_response(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
    elif submitted and not user_input.strip():
        st.warning("Please enter a message.")

    with conversation_col:
        user_messages = sum(1 for speaker, _ in st.session_state.chat_history if speaker == "You")
        bot_messages = sum(1 for speaker, _ in st.session_state.chat_history if speaker == "Bot")
        st.markdown(
            f"""
            <div class="support-status">
                <strong>Rule-based support</strong> · Works offline · You: {user_messages} · Bot: {bot_messages}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown(
                '<div class="empty-chat">No messages yet. Start with a quick support option or type a message.</div>',
                unsafe_allow_html=True,
            )
        else:
            for speaker, message in st.session_state.chat_history:
                role_class = "user" if speaker == "You" else "bot"
                speaker_label = "You" if speaker == "You" else "Bot"
                safe_message = html.escape(message)
                st.markdown(
                    f"<div class='chat-row {role_class}'>"
                    f"<div class='chat-bubble {role_class}'><strong>{speaker_label}</strong><br>{safe_message}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This is a demonstration system. For professional mental health support, consult qualified healthcare providers.*")