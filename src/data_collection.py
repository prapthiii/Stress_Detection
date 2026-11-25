import streamlit as st
import numpy as np
import librosa
try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ocr_available = True
except Exception as e:
    ocr_available = False
    st.sidebar.warning("OCR not available. Image upload feature disabled.")

from src.preprocessing import Preprocessor
from src.feature_extraction import FeatureExtractor
from src.feature_fusion import FeatureFusion
from src.model import StressModel
from src.classification import StressClassifier
from src.suggestions import SupportSuggestions
from src.evaluation import Evaluator
from src.questionnaire import Questionnaire
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def get_chatbot_response(user_input, history):
    """Rule-based chatbot with a more natural conversational style."""
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
    .chat-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }

    .chat-row.user {
        justify-content: flex-end;
    }

    .chat-bubble {
        max-width: 70%;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        box-shadow: 0 5px 16px rgba(0,0,0,0.08);
        line-height: 1.4;
    }

    .chat-bubble.user {
        background: linear-gradient(135deg, #4a90e2, #3b79d4);
        color: #ffffff;
        border-bottom-right-radius: 4px;
    }

    .chat-bubble.bot {
        background: #f1f3f5;
        color: #1f2d3d;
        border-bottom-left-radius: 4px;
    }

    .chat-buttons {
        margin-bottom: 1rem;
    }

    .stButton>button {
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
prep = Preprocessor()
extractor = FeatureExtractor()
fusion = FeatureFusion()
model = StressModel()
if not model.load_model():
    # Dummy training if no saved model
    # Voice features: 16 (13 MFCC + pitch + energy + speech_rate)
    # Text features: 14 (4 basic + 10 TF-IDF)
    # Total: 30 features
    X_dummy = np.random.rand(50, 30)
    y_dummy = np.random.randint(0, 4, 50)
    model.train(X_dummy, y_dummy)
classifier = StressClassifier(model)
suggester = SupportSuggestions()
evaluator = Evaluator()
questionnaire = Questionnaire()

# Sidebar
st.sidebar.title("🧠 Stress Management System")
st.sidebar.markdown("---")

analysis_type = st.sidebar.selectbox(
    "Choose Analysis Method",
    ["🏠 Home", "🎤 Voice Analysis", "📝 Text Analysis", "📊 Self Assessment", "💬 Chatbot Support"],
    help="Select how you want to assess stress"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This system analyzes stress through multiple modalities: voice, text, and self-assessment questionnaires.")

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

elif analysis_type == "🎤 Voice Analysis":
    st.markdown('<h2 class="main-header">🎤 Voice Stress Analysis</h2>', unsafe_allow_html=True)
    st.write("Upload a WAV audio file to analyze stress indicators in speech.")

    uploaded_file = st.file_uploader("Choose a WAV file", type="wav", help="Upload a voice recording for analysis")

    if uploaded_file is not None:
        with st.spinner("Processing audio..."):
            # Save temporarily
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Process
            y, sr = prep.preprocess_voice("temp.wav")
            voice_feat = extractor.extract_voice_features(y, sr)
            text_feat = extractor.extract_text_features("")
            fused_feat = fusion.fuse_features(voice_feat, text_feat)

            stress_level, confidence = classifier.classify(fused_feat)
            stress_level_num = list(classifier.labels.keys())[list(classifier.labels.values()).index(stress_level)]
            suggestion = suggester.get_suggestion(stress_level_num)

        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f'<div class="stress-level {"no-stress" if stress_level_num == 0 else "mild-stress" if stress_level_num == 1 else "moderate-stress" if stress_level_num == 2 else "high-stress"}">{stress_level}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.2f}")
        
        with col2:
            st.success(f"**Analysis Complete!** {suggestion}")

elif analysis_type == "📝 Text Analysis":
    st.markdown('<h2 class="main-header">📝 Text Stress Analysis</h2>', unsafe_allow_html=True)
    st.write("Enter text directly or upload an image containing handwritten or printed text for emotional analysis.")

    input_type = st.radio("Input Type", ["✍️ Direct Text", "🖋️ Handwritten Text (Image)"], horizontal=True)

    text_input = ""

    if input_type == "✍️ Direct Text":
        text_input = st.text_area("Enter your text here", height=150, placeholder="Type your thoughts, messages, or any text to analyze...")
    elif input_type == "🖋️ Handwritten Text (Image)" and ocr_available:
        st.info("📝 Upload a clear image of handwritten or printed text. For best results, ensure good lighting and clear handwriting.")
        
        uploaded_image = st.file_uploader("Upload an image with handwritten/printed text", type=["png", "jpg", "jpeg"], key="handwriting_upload")
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            # OCR extraction with advanced options
            with st.spinner("Extracting text from handwriting..."):
                try:
                    # Extract text with confidence information
                    text_input = pytesseract.image_to_string(image)
                    
                    # Get detailed data including confidence
                    data = pytesseract.image_to_data(image, output_type='dict')
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    with col2:
                        st.success("✅ Text extracted successfully")
                    
                    st.markdown("### 📄 Extracted Text:")
                    st.text_area("Extracted text (edit if needed):", value=text_input, height=150, key="extracted_text")
                    text_input = st.session_state.get("extracted_text", text_input)
                    
                    if avg_confidence < 60:
                        st.warning("⚠️ Low confidence in text extraction. The image may be unclear. Try uploading a clearer image for better results.")
                        
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    text_input = ""
    elif input_type == "🖋️ Handwritten Text (Image)" and not ocr_available:
        st.error("❌ OCR not available on this system. Please use Direct Text input instead.")

    analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

    if analyze_button:
        if text_input.strip():
            with st.spinner("Analyzing text for stress indicators..."):
                text_tokens = prep.preprocess_text(text_input)
                text_feat = extractor.extract_text_features(" ".join(text_tokens))
                voice_feat = np.random.rand(16)  # Dummy voice features
                fused_feat = fusion.fuse_features(voice_feat, text_feat)

                stress_level, confidence = classifier.classify(fused_feat)
                stress_level_num = list(classifier.labels.keys())[list(classifier.labels.values()).index(stress_level)]
                suggestion = suggester.get_suggestion(stress_level_num)
                
                # Calculate sentiment score
                sentiment = sentiment_analyzer.polarity_scores(text_input)
                compound = sentiment['compound']
                
                # Additional text analysis metrics
                word_count = len(text_tokens)
                char_count = len(text_input)
                avg_word_length = char_count / word_count if word_count > 0 else 0

            # Results display with more details
            st.markdown("### 📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment Score", f"{compound:.2f}", help="Range: -1 (negative) to +1 (positive)")
            with col2:
                st.metric("Word Count", word_count)
            
            st.progress((compound + 1) / 2, text="Emotional Valence")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f'<div class="stress-level {"no-stress" if stress_level_num == 0 else "mild-stress" if stress_level_num == 1 else "moderate-stress" if stress_level_num == 2 else "high-stress"}">{stress_level}</div>', unsafe_allow_html=True)
            
            with col2:
                st.info(f"💡 {suggestion}")
            
            # Detailed analysis
            with st.expander("📈 Detailed Analysis"):
                st.write("**Emotional Indicators:**")
                if compound < -0.5:
                    st.warning("Strong negative sentiment detected")
                elif compound < 0:
                    st.info("Negative sentiment present")
                elif compound > 0.5:
                    st.success("Strong positive sentiment detected")
                elif compound > 0:
                    st.success("Positive sentiment present")
                else:
                    st.info("Neutral sentiment")
                
                st.write(f"**Text Statistics:**")
                st.write(f"- Words: {word_count}")
                st.write(f"- Characters: {char_count}")
                st.write(f"- Average word length: {avg_word_length:.1f} characters")
                st.write(f"- Unique words: {len(set(text_tokens))}")
        else:
            st.warning("Please provide text or upload an image to analyze.")

elif analysis_type == "📊 Self Assessment":
    st.markdown('<h2 class="main-header">📊 Self Assessment Questionnaire (PHQ-9)</h2>', unsafe_allow_html=True)
    st.write("Complete this 9-question assessment to evaluate your current depression/stress levels.")

    questions = questionnaire.questions
    options = questionnaire.options
    scores = questionnaire.scores

    st.markdown("### Please answer all questions:")
    
    total_score = 0
    answered_questions = 0
    
    for i, q in enumerate(questions):
        with st.expander(f"Question {i+1}", expanded=False):
            st.write(q)
            answer = st.radio(f"Select your answer for Q{i+1}", options, key=f"q{i}", label_visibility="collapsed")
            if answer:
                total_score += scores[options.index(answer)]
                answered_questions += 1

    progress = answered_questions / len(questions)
    st.progress(progress, text=f"Progress: {answered_questions}/{len(questions)} questions answered")

    if st.button("📈 Calculate Stress Level", type="primary", use_container_width=True):
        if answered_questions == len(questions):
            responses = [scores[options.index(answer)] for answer in [st.session_state.get(f"q{i}", None) for i in range(len(questions))] if st.session_state.get(f"q{i}", None)]
            total_score = questionnaire.calculate_score(responses)
            interpretation, stress_level_num = questionnaire.interpret_score(total_score)

            stress_level = classifier.labels[stress_level_num]
            suggestion = suggester.get_suggestion(stress_level_num)

            # Results
            st.balloons()
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.metric("Total Score", f"{total_score}/27")
            
            with col2:
                st.markdown(f'<div class="stress-level {"no-stress" if stress_level_num == 0 else "mild-stress" if stress_level_num == 1 else "moderate-stress" if stress_level_num == 2 else "high-stress"}">{stress_level}</div>', unsafe_allow_html=True)
            
            with col3:
                st.metric("Stress Level", stress_level_num + 1, delta="out of 4")

            st.progress(total_score / 27, text=f"Stress Indicator: {total_score/27:.1%}")
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
                st.write(f"**Score Breakdown:** {total_score} points out of 30 maximum")
                st.write(f"**Classification:** {stress_level}")
                if total_score > 20:
                    st.warning("Consider professional support if stress persists.")
                else:
                    st.info("Great job managing stress! Continue positive habits.")
        else:
            st.error(f"Please answer all questions. {len(questions) - answered_questions} remaining.")

elif analysis_type == "💬 Chatbot Support":
    st.markdown('<h2 class="main-header">💬 Chatbot Support</h2>', unsafe_allow_html=True)
    st.write("Talk to our supportive chatbot about your feelings and stress. It's here to listen and offer gentle guidance.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for speaker, message in st.session_state.chat_history:
        role_class = "user" if speaker == "You" else "bot"
        speaker_label = "You" if speaker == "You" else "Bot"
        st.markdown(
            f"<div class='chat-row {role_class}'>"
            f"<div class='chat-bubble {role_class}'><strong>{speaker_label}:</strong> {message}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick support buttons
    st.markdown('<div class="chat-buttons">', unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1, 1, 1, 1, 1])
    if btn_col1.button("🧘 Breathing Exercise"):
        st.session_state.chat_history.append(("Bot", "Let's try a quick breathing exercise: inhale for 4 seconds, hold for 4 seconds, then exhale for 6 seconds. Repeat this 3 times and see how you feel."))
    if btn_col2.button("📝 Grounding Exercise"):
        st.session_state.chat_history.append(("Bot", "Try grounding: name 5 things you can see, 4 things you can touch, 3 sounds you hear, 2 things you can smell, and 1 thing you can taste."))
    if btn_col3.button("✍️ Write It Down"):
        st.session_state.chat_history.append(("Bot", "Writing can help you process emotions. Try writing for a few minutes about what you're feeling and why. No need to share it, just let it out."))
    if btn_col4.button("🧡 Positive Affirmation"):
        st.session_state.chat_history.append(("Bot", "You are doing your best in a tough moment, and that matters. It's okay to take things one step at a time."))
    if btn_col5.button("🌤️ Mood Check"):
        st.session_state.chat_history.append(("Bot", "How would you rate your mood on a scale of 1-10 right now? It's okay to be honest — I'm here to listen."))
    st.markdown('</div>', unsafe_allow_html=True)

    # Ensure input state exists
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # Chat controls
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.form(key="chat_form"):
            user_input = st.text_input(
                "Share your thoughts:",
                value=st.session_state.chat_input,
                placeholder="How are you feeling today?",
                key="chat_input"
            )
            submitted = st.form_submit_button("Send Message", type="primary")
    with col2:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.chat_input = ""
            st.experimental_rerun()

    if submitted and user_input.strip():
        response = get_chatbot_response(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.session_state.chat_input = ""  # Clear the input after sending
    elif submitted and not user_input.strip():
        st.warning("Please enter a message.")

# Footer
st.markdown("---")
st.markdown("*This is a demonstration system. For professional mental health support, consult qualified healthcare providers.*")