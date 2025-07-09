import os
import re
import time
#from dotenv import load_dotenv
import streamlit as st
import PyPDF2
import google.generativeai as genai
import speech_recognition as sr
from random import sample
import random
from html import escape
import asyncio
import edge_tts
import pandas as pd
import tempfile
import traceback
import av
import soundfile as sf
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from twilio.rest import Client  
import logging
import whisper
import speech_recognition as sr
import logging
#model = whisper.load_model("base")

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="GrillMaster", layout="wide")

# Load API key
#load_dotenv()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def convert_frames_to_wav(frames, wav_path):
    audio = b''.join([f.to_ndarray().tobytes() for f in frames])
    with sf.SoundFile(wav_path, mode='x', samplerate=48000, channels=1, subtype='PCM_16') as f:
        f.write(audio)
# Initialize session state
for key, default in {
    "generated_questions": [],
    "current_question_index": 0,
    "answers": [],
    "evaluation_feedback": "",
    "overall_score": 0,
    "percentage_score": 0,
    "is_recording": False,
    "question_played": False,
    "selected_domain": "",
    "response_captured": False,
    "timer_start": None,
    "show_summary": False,
    "recorded_text": "",
    "recording_complete": False,
    "recording_started": False,
    "audio_played": False,
    "question_start_time": 0.0,
    "record_phase": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Utility functions
def extract_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages).strip()

def get_questions(prompt, input_text, num_questions=3, max_retries=10):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    if "previous_questions" not in st.session_state:
        st.session_state["previous_questions"] = set()

    new_questions = []
    retries = 0

    while len(new_questions) < num_questions and retries < max_retries:
        # Add artificial noise/randomness to input
        noise = f" [session: {random.randint(1000,9999)} time: {time.time()}]"
        modified_input = input_text + noise

        response = model.generate_content([prompt, modified_input])
        questions = [q.strip("*•- ") for q in response.text.strip().split("") if q.strip() and "question" not in q.lower()]

        for q in questions:
            if q not in st.session_state["previous_questions"]:
                st.session_state["previous_questions"].add(q)
                new_questions.append(q)
                if len(new_questions) == num_questions:
                    break

        retries += 1

    return new_questions

async def generate_question_audio(question, voice="en-IE-EmilyNeural"):
    clean_question = re.sub(r'[^A-Za-z0-9.,?! ]+', '', question)
    tts = edge_tts.Communicate(text=clean_question, voice=voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        await tts.save(tmp_file.name)
        return tmp_file.name
    
########################################///////////////////////////////////////////////////#########################################

# HR_PARAMETERS_CONFIG - Updated based on your latest Excel sheet (input_file_0.png)
# These are the parameters that can be judged from audio/text responses.
HR_PARAMETERS_CONFIG = {
    "Voice Modulation": { # Non-Verbal Cues
        "weight_original": 5, 
        "rubric": "1-5 (5=Good pace/tone, conversational; 3=Sounds Scripted/Slight Monotony; 1=Flat tone/Robotic)"
    },
    "Confidence": { # Personality
        "weight_original": 7, 
        "rubric": "1-5 (5=Bold & Confident throughout; 3=Confused/Nervous in parts; 1=Extremely nervous/Timid)"
    },
    "Attitude": { # Personality
        "weight_original": 3, 
        "rubric": "1-5 (5=Assertive, Positive, Open; 3=Neutral/Mildly defensive; 1=Aggressive/Pessimistic/Dismissive)"
    },
    "Flow & Fluency": { # Articulation
        "weight_original": 20, 
        "rubric": "1-5 (5=Excellent Fluency, Spontaneous; 3=Initially struggles, then manages/Takes some time; 1=Many fillers/Pauses/Dead silence)"
    },
    "Structured thoughts & Clarity": { # Articulation
        "weight_original": 10, 
        "rubric": "1-5 (5=Organized, Crisp, Coherent thoughts, e.g. STAR method; 3=Ideas are okay but clarity/structure could be better; 1=Incoherent/Rambling/Struggles to put thoughts into words)"
    },
    "Sentence Formation": { # Language Skills
        "weight_original": 20, 
        "rubric": "1-5 (5=Good Clarity, Variety in sentence structure, Good Vocab; 3=Decent communication, might find some words difficult; 1=Talks in fragments/one-liners, Hard to understand)"
    },
    "Basics of Grammar + SVA": { # Language Skills (SVA = Subject-Verb Agreement)
        "weight_original": 10, 
        "rubric": "1-5 (5=Good Command over Language, Minimal errors; 3=Average communicator, some errors but understandable; 1=Makes a lot of Grammatical Errors impacting clarity)"
    },
    "Persuasiveness": { # Rapport Building
        "weight_original": 3, 
        "rubric": "1-5 (5=Impactful, Convincing Answers, Connects with interviewer; 3=Average or Common Answers; 1=Lacks Presence of Mind/No connection)"
    },
    "Quality of Answers": { # Rapport Building
        "weight_original": 7, 
        "rubric": "1-5 (5=Handles questions well, Relevant & Thoughtful Answers, Asks good questions; 3=Very Generic Answers; 1=Vague/Lacks Depth/Shallow/Irrelevant)"
    }
}

# Calculate total original weight for normalization
TOTAL_ORIGINAL_WEIGHT_HR = sum(param_data["weight_original"] for param_data in HR_PARAMETERS_CONFIG.values()) # Should be 85

# Add normalized weights to the config for calculating score out of 100
for param in HR_PARAMETERS_CONFIG:
    HR_PARAMETERS_CONFIG[param]["weight_normalized"] = (HR_PARAMETERS_CONFIG[param]["weight_original"] / TOTAL_ORIGINAL_WEIGHT_HR) * 100


########################################///////////////////////////////////////////////////#########################################
# SUmmary of improvement(function)

def generate_improvement_suggestions():
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    difficulty_level = st.session_state.get("difficulty_level_select", "Beginner")
    level_string = difficulty_level.lower()

    if not st.session_state.get("answers"):
        st.session_state.improvement_suggestions = "No answers were recorded to generate improvement suggestions."
        return

    # Prepare the context for the LLM
    qa_context = []
    for i, entry in enumerate(st.session_state["answers"]):
        qa_context.append(
            f"Question {i+1}: {entry['question']}\n"
            f"Candidate's Answer {i+1}: {str(entry.get('response', '[No response provided]'))}"
        )
    full_qa_context = "\n\n".join(qa_context)

    initial_evaluation_feedback = st.session_state.get("evaluation_feedback", "Initial evaluation not available.")

    # Remove any previous "Total Calculated Score..." line from the initial feedback
    # to avoid confusing the LLM when it sees it as part of the context.
    initial_evaluation_lines = initial_evaluation_feedback.splitlines()
    cleaned_initial_evaluation = "\n".join(
        line for line in initial_evaluation_lines if not line.strip().startswith("**Total Calculated Score:**")
    )


    improvement_prompt_template = """
    You are an expert interview coach. You have the following information about a candidate's mock interview:
    - Candidate's Level: {level_string}
    - Questions Asked and Candidate's Answers:
    {full_qa_context}
    - Initial Evaluation Feedback Provided to Candidate:
    ---
    {cleaned_initial_evaluation}
    ---

    Based on all this information, your task is to provide DETAILED and CONSTRUCTIVE suggestions for each question to help the candidate improve. Be supportive and encouraging.

    For EACH question, please provide:
    1.  **How to Improve This Answer:** Specific, actionable advice on what the candidate could have added, clarified, or approached differently to make their answer better for their {level_string} level. Focus on 1-2 key improvement points.
    2.  **Hints for an Ideal Answer:** Briefly mention 2-3 key concepts, terms, or elements that a strong answer (appropriate for their {level_string} level) would typically include. DO NOT provide a full model answer, just hints and pointers.

    Keep the tone positive and focused on learning.

    Structure your response clearly for each question. Example for one question:

    ---
    **Regarding Question X: "[Original Question Text Here]"**

    *How to Improve This Answer:*
    [Your specific suggestion 1 for improvement...]
    [Your specific suggestion 2 for improvement...]

    *Hints for an Ideal Answer (Key Points to Consider):*
    - Hint 1 or Key concept 1
    - Hint 2 or Key concept 2
    - Hint 3 or Key element 3 (optional)
    ---
    (Repeat this structure for all questions)
    """

    formatted_improvement_prompt = improvement_prompt_template.format(
        level_string=level_string,
        full_qa_context=full_qa_context,
        cleaned_initial_evaluation=cleaned_initial_evaluation
    )

    try:
        st.info("🤖 Generating detailed improvement suggestions... Please wait.")
        response = model.generate_content(formatted_improvement_prompt)
        st.session_state.improvement_suggestions = response.text.strip()
        st.session_state.improvement_suggestions_generated = True
        st.success("Detailed suggestions generated!")
    except Exception as e:
        st.error(f"Error generating improvement suggestions: {e}")
        st.session_state.improvement_suggestions = f"Could not generate suggestions due to an error: {e}"
        st.session_state.improvement_suggestions_generated = False

########################################///////////////////////////////////////////////////#########################################

# Evaluate candidate answers - YOUR FUNCTION



def evaluate_answers():
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    # difficulty_level_select is the key for the difficulty selectbox in your sidebar
    difficulty_level = st.session_state.get("difficulty_level_select", "Beginner") 
    level_string = difficulty_level.lower()
    num_answered_questions = len(st.session_state.get("answers", []))

    # Reset improvement suggestions flag when re-evaluating
    st.session_state.improvement_suggestions_generated = False
    st.session_state.improvement_suggestions = ""

    meaningful_answers_exist = False
    if st.session_state.get("answers"):
        for entry in st.session_state["answers"]:
            response_text = str(entry.get('response', '')).strip().lower()
            no_response_placeholders = [
                "", "[no response provided]", "[no response - timed out]",
                "[no response]", "no response", "[could not understand audio]",
                "[no clear response recorded]", "[no action - timed out before recording]",
                "[no speech detected in recording time]", "[no speech recorded - time up]",
                "[recording stopped manually, possibly empty]",
                "[no action - did not start recording]",
                "[no speech detected in recording phase]" 
            ]
            if response_text not in no_response_placeholders:
                meaningful_answers_exist = True
                break
    
    if not meaningful_answers_exist:
        no_answer_feedback_qualitative = "No meaningful answers were provided for evaluation.\n\n"
        if st.session_state.selected_domain == "Soft Skills":
            hr_params_na = "\n".join([f"- {param}: 0/5" for param in HR_PARAMETERS_CONFIG.keys()])
            no_answer_feedback = (
                 "No meaningful answers were provided for evaluation.\n\n"
                f"**Parameter Scores (1-5):**\n{hr_params_na}\n\n"
                "**Overall Qualitative Feedback:**\nCandidate did not provide responses to evaluate soft skills."
            )
            st.session_state["hr_parameter_scores_dict"] = {param: 0.0 for param in HR_PARAMETERS_CONFIG.keys()} # Store zeroed scores
        else: # Non-HR domains
            no_answer_feedback = (
                "No meaningful answers were provided.\n"
                "**Total Calculated Score:** 0.0 / 0.0 (0.0%)\n\n" # Placeholder for non-HR if no answers
                "**Overall Evaluation Summary:** N/A"
            )
        st.session_state["evaluation_feedback"] = no_answer_feedback
        st.session_state["overall_score"] = 0.0
        st.session_state["percentage_score"] = 0.0
        return

    # --- BRANCHING FOR HR (SOFT SKILLS) VS OTHER DOMAINS ---
    if st.session_state.selected_domain == "Soft Skills":
        hr_prompt_parameter_list = ""
        for param, config in HR_PARAMETERS_CONFIG.items():
            hr_prompt_parameter_list += f"- **{param}:** {config['rubric']}\n"
        
        hr_prompt_template = f"""
        You are an experienced HR interview evaluator assessing a candidate's soft skills based on their answers to interview questions.
        The candidate's performance across ALL answers should inform your scores for the following parameters.

        **Parameters to Score (Assign a score from 1 to 5 for each):**
        {hr_prompt_parameter_list}

        After providing a score (1-5) for each of the above parameters, also write an **Overall Qualitative Feedback** section. 
        This section should summarize the candidate's general soft skill strengths and areas for improvement, based on their communication, engagement, and professionalism throughout the interview.

        **REQUIRED OUTPUT FORMAT (Strictly Adhere):**

        **Parameter Scores (1-5):**
        Voice Modulation: [score]
        Confidence: [score]
        Attitude: [score]
        Flow & Fluency: [score]
        Structured thoughts & Clarity: [score]
        Sentence Formation: [score]
        Basics of Grammar + SVA: [score]
        Persuasiveness: [score]
        Quality of Answers: [score]

        **Overall Qualitative Feedback:**
        [Your holistic qualitative feedback here. Be encouraging and constructive.]
        """
        candidate_responses_formatted_hr = "\n\n".join(
            [f"Question {i+1}: {entry['question']}\nCandidate's Answer {i+1}: {str(entry.get('response', '[No response provided]'))}" 
             for i, entry in enumerate(st.session_state["answers"])]
        )
        #full_prompt_for_hr_evaluation = f"{hr_prompt_template}\n\nCandidate's Interview Answers:\n{candidate_responses_formatted_hr}"
        full_prompt_for_hr_evaluation = f"{hr_prompt_template}\n\nCandidate's Interview Answers (Consider all of these for holistic parameter scoring):\n{candidate_responses_formatted_hr}"

        try:
            response_content = model.generate_content(full_prompt_for_hr_evaluation)
            full_llm_response_text = response_content.text.strip()
            print("--- FULL LLM SOFT SKILLS RESPONSE ---")
            print(full_llm_response_text)
            print("------ END RESPONSE ------")
            print("--- AI Full Response for Soft Skills ---\n", full_llm_response_text, "\n------------------------")

            hr_parameter_scores_parsed_dict = {} # To store parsed scores for each HR param
            total_weighted_score_percentage = 0.0

            for param_name_config, config_data in HR_PARAMETERS_CONFIG.items():
                # Using a more specific regex, anchored to the start of a line (after optional list marker)
                # re.escape ensures special characters in param_name_config are treated literally.
                param_score_pattern = re.compile(
                    r"^\s*(?:[\*\-]\s*)?" + re.escape(param_name_config.split('(')[0].strip()) + r"\s*[:\-–—]?\s*(\d+(?:\.\d+)?)\b", 
                    re.IGNORECASE | re.MULTILINE
                ) # \b for word boundary after score
                
                match = param_score_pattern.search(full_llm_response_text)
                param_score = 1.0 # Default to 1 (lowest actual score) if not found or unparseable
                if match:
                    try:
                        score_text = match.group(1)
                        param_score = float(score_text)
                        param_score = max(1.0, min(5.0, param_score)) # Clamp score strictly 1-5 for HR
                        print(f"HR Param '{param_name_config}' - Matched text: '{score_text}', Parsed: {param_score}")
                    except ValueError:
                        print(f"HR Param '{param_name_config}' - ValueError parsing score from '{score_text}' in match '{match.group(0)}'. Defaulting to 1.0.")
                        param_score = 1.0 
                else:
                    print(f"HR Param '{param_name_config}' - Score pattern not found. Defaulting to 1.0 for this param.")
                
                hr_parameter_scores_parsed_dict[param_name_config] = param_score
                total_weighted_score_percentage += (param_score / 5.0) * config_data["weight_normalized"] # Use normalized weight
            
            st.session_state["hr_parameter_scores_dict"] = hr_parameter_scores_parsed_dict # Store for table display

            num_qs_in_session = len(st.session_state.get("answers", []))
            max_possible_score = num_qs_in_session * 5.0  # Each Q worth 5
            actual_score = (total_weighted_score_percentage / 100.0) * max_possible_score

            st.session_state["overall_score"] = round(actual_score, 1)
            st.session_state["percentage_score"] = round((actual_score / max_possible_score) * 100, 1)

            
            # Construct the feedback to be displayed: Parsed scores + Qualitative from LLM
            # The full_llm_response_text might still be useful if qualitative parsing is tricky
            parsed_scores_display_text = "**Parsed Parameter Scores (1-5 based on AI Evaluation):**\n"
            for p_name, p_score in hr_parameter_scores_parsed_dict.items():
                parsed_scores_display_text += f"- {p_name}: {p_score:.1f}/5\n"
            
            qualitative_feedback_hr_extract = "Overall qualitative feedback section not clearly identified in AI response."
            qualitative_match_hr = re.search(r"\*\*Overall Qualitative Feedback:\*\*(.*)", full_llm_response_text, re.DOTALL | re.IGNORECASE)
            if qualitative_match_hr:
                qualitative_feedback_hr_extract = qualitative_match_hr.group(1).strip()
            
            st.session_state["evaluation_feedback"] = f"{parsed_scores_display_text}\n\n**Overall Qualitative Feedback from AI:**\n{qualitative_feedback_hr_extract}"

        except Exception as e_hr_eval:
            st.error(f"Error during HR/Soft Skills evaluation processing: {e_hr_eval}")
            print(f"HR EVALUATION PROCESSING TRACEBACK:\n{traceback.format_exc()}")
            st.session_state["evaluation_feedback"] = f"Could not process HR skills evaluation: {e_hr_eval}"
            st.session_state["overall_score"] = 0.0
            st.session_state["percentage_score"] = 0.0

    else: # --- NON-HR (Analytics, Finance) Evaluation Logic ---
        base_assessment_criteria_qualitative_non_hr = """
        For the OVERALL qualitative summary, assess responses based on:
        - Conceptual Understanding (effort and relevance more than perfect accuracy for the level)
        - Communication Clarity (can the core idea be understood?)
        - Depth of Explanation (relative to expected level)
        - Use of Examples (if any, and if appropriate for the level)
        - Logical Flow (is there a basic structure or train of thought?)
        """
        per_question_scoring_guidelines_non_hr = f"""
        For EACH question and its answer, provide a score from 0 to 5 points.
        The candidate is at a {level_string} level.
        Consider the following when assigning the per-question score:
        - Effort and relevance of the answer.
        - Clarity of thought for the candidate's level.
        - Basic logical structure.
        - Use of examples, if any were given and appropriate.
        """
        if level_string == "beginner":
            level_specific_instructions_non_hr = """
            You are an **extremely understanding, encouraging, and supportive** interview evaluator for a **BEGINNER/FRESHER**. Your primary goal is to **build confidence**.
            **Scoring Guidelines for Beginners (0-5 points per question):**
            - **5 points:** Generally correct and relevant, even if brief. Shows clear effort and basic understanding.
            - **4 points:** Good attempt, relevant, shows some understanding or key terms (e.g., one/two relevant words).
            - **3 points:** Tries, somewhat related, or acknowledges question with a vague thought.
            - **1-2 points:** Minimal effort, mostly irrelevant, but an attempt beyond silence.
            - **0 points:** Completely irrelevant, no attempt, or placeholder.
            Provide VERY positive feedback.
            """
        elif level_string == "intermediate":
            level_specific_instructions_non_hr = """Supportive evaluator for **INTERMEDIATE**. Scoring (0-5): 5=Correct/Clear; 3-4=Mostly correct; 1-2=Partial/Gaps; 0=Incorrect."""
        else: # Advanced
            level_specific_instructions_non_hr = """Discerning evaluator for **ADVANCED**. Scoring (0-5): 5=Accurate/Comprehensive; 3-4=Correct lacks nuance; 1-2=Inaccurate; 0=Fundamentally incorrect."""
        
        evaluation_prompt_template_non_hr = f"""
        {level_specific_instructions_non_hr} 
        {per_question_scoring_guidelines_non_hr}
        {base_assessment_criteria_qualitative_non_hr}
        **YOUR RESPONSE MUST STRICTLY FOLLOW THIS FORMAT. PROVIDE SCORES FOR EACH QUESTION.**
        Output format:

        **Per-Question Scores:**
        Question 1 Score: [Score for Q1 out of 5]
        ... (repeat for all {num_answered_questions} questions provided)

        **Overall Evaluation Summary:**
        - Concept Understanding: [Overall qualitative feedback here]
        - Communication: [Overall qualitative feedback here]
        - Depth of Explanation: [Overall qualitative feedback here]
        - Examples: [Overall qualitative feedback here]
        - Logical Flow: [Overall qualitative feedback here]
        [Any additional overall encouraging remarks can optionally follow here]
        """
        candidate_responses_formatted_non_hr = "\n\n".join(
            [f"Question {i+1}: {entry['question']}\nAnswer {i+1}: {str(entry.get('response', '[No response provided]'))}" for i, entry in enumerate(st.session_state["answers"])]
        )
        full_prompt_for_non_hr_evaluation = f"{evaluation_prompt_template_non_hr}\n\nCandidate Responses:\n{candidate_responses_formatted_non_hr}"

        try:
            response_content_non_hr = model.generate_content(full_prompt_for_non_hr_evaluation)
            full_llm_response_text_non_hr = response_content_non_hr.text.strip()
            raw_llm_feedback_non_hr = full_llm_response_text_non_hr
            
            print("--- LLM Output for Non-HR Score Extraction ---"); print(full_llm_response_text_non_hr); print("---")
            
            total_score_non_hr = 0.0; parsed_scores_count_non_hr = 0; per_question_scores_list_non_hr = []
            score_line_pattern_non_hr = re.compile(r"Question\s*(\d+)\s*Score:\s*(\d+(?:\.\d+)?)(?:\s*/\s*5)?", re.IGNORECASE)
            text_to_search_non_hr = full_llm_response_text_non_hr
            scores_block_match_non_hr = re.search(r"(?i)\*\*Per-Question Scores:\*\*(.*?)(?=\*\*Overall Evaluation Summary:\*\*|\Z)", text_to_search_non_hr, re.DOTALL)
            
            if scores_block_match_non_hr: 
                text_to_search_non_hr = scores_block_match_non_hr.group(1).strip()
                print(f"Non-HR: Found 'Per-Question Scores' block:\n{text_to_search_non_hr}")
            else:
                print("Non-HR: No dedicated 'Per-Question Scores' block found; searching entire response.")


            for match_non_hr in score_line_pattern_non_hr.finditer(text_to_search_non_hr):
                q_num_text_non_hr, score_val_text_non_hr = match_non_hr.group(1), match_non_hr.group(2)
                try:
                    score_non_hr = float(score_val_text_non_hr)
                    score_non_hr = max(0.0, min(5.0, score_non_hr))
                    total_score_non_hr += score_non_hr
                    parsed_scores_count_non_hr += 1
                    per_question_scores_list_non_hr.append(f"Question {q_num_text_non_hr}: {score_non_hr:.1f}/5")
                    print(f"Non-HR Matched Q{q_num_text_non_hr} Score: {score_non_hr}")
                except ValueError:
                    print(f"Non-HR Warning: Could not parse score '{score_val_text_non_hr}' from: '{match_non_hr.group(0)}'")
            
            if parsed_scores_count_non_hr != num_answered_questions and meaningful_answers_exist: 
                 st.warning(f"Non-HR Score Count Mismatch: Parsed {parsed_scores_count_non_hr} scores, expected {num_answered_questions}.")
                 print(f"Non-HR Score Count Mismatch: Expected {num_answered_questions}, got {parsed_scores_count_non_hr}")

            if parsed_scores_count_non_hr == 0 and meaningful_answers_exist: 
                 st.warning("CRITICAL (Non-HR): No per-question scores parsed from LLM response. Total score set to 0.")
                 print("CRITICAL (Non-HR): No per-question scores parsed.")
                 total_score_non_hr = 0.0

            max_score_non_hr = num_answered_questions * 5.0
            st.session_state["overall_score"] = total_score_non_hr
            st.session_state["percentage_score"] = (total_score_non_hr / max_score_non_hr) * 100.0 if max_score_non_hr > 0 else 0.0
            
            final_feedback_non_hr = f"**Total Calculated Score:** {st.session_state['overall_score']:.1f} / {max_score_non_hr:.1f}  ({st.session_state['percentage_score']:.1f}%)\n\n"
            if per_question_scores_list_non_hr: 
                final_feedback_non_hr += "**Parsed Per-Question Scores:**\n" + "\n".join(per_question_scores_list_non_hr) + "\n\n"
            
            qual_summary_match_non_hr = re.search(r"\*\*Overall Evaluation Summary:\*\*(.*)", raw_llm_feedback_non_hr, re.DOTALL | re.IGNORECASE)
            if qual_summary_match_non_hr: 
                final_feedback_non_hr += "**Overall Qualitative Summary (from AI):**\n" + qual_summary_match_non_hr.group(1).strip()
            else: 
                final_feedback_non_hr += "\n---\n**Full AI Response (for context if summary parsing failed):**\n" + raw_llm_feedback_non_hr
            st.session_state["evaluation_feedback"] = final_feedback_non_hr.strip()

        except Exception as e_non_hr_eval:
            st.error(f"Error during Non-HR evaluation processing: {e_non_hr_eval}")
            print(f"NON-HR EVALUATION PROCESSING TRACEBACK:\n{traceback.format_exc()}")
            st.session_state["evaluation_feedback"] = f"Could not process Non-HR evaluation: {e_non_hr_eval}"
            st.session_state["overall_score"] = 0.0
            st.session_state["percentage_score"] = 0.0
########################################///////////////////////////////////////////////////#########################################
# --- Prompts for Question Generation ---
BEGINNER_PROMPT = """
You are a friendly mock interview trainer conducting a **Beginner-level** spoken interview in the domain of **{domain}**.
Ask basic verbal interview questions based on the candidate's input: **{input_text}**.

Guidelines:
- Ask simple conceptual questions.
- Avoid jargon and complex examples.
- Use easy language.
- No coding or technical syntax required.
Ensure the questions are clear, to the point, and suitable for a {difficulty_level}-level interview in {selected_domain}.
**New Requirement:**
🚫 **Do NOT repeat any questions from previous generations again and again.** Ensure all generated questions are unique and different from past sessions.

**Guidelines:**
    ✅ Questions should focus on key concepts, best practices, and problem-solving within {selected_domain}.
    ✅ Ensure questions are direct, structured, and relevant to real-world applications.
    ❌ Do NOT include greetings like 'Let's begin' or 'Welcome to the interview'.
    ❌ Avoid vague or open-ended statements—each question should be concise and specific.
"""

INTERMEDIATE_PROMPT = """
You are a professional mock interviewer conducting an **Intermediate-level** spoken interview in the domain of **{domain}**.
Ask moderately challenging verbal interview questions based on the candidate's input: **{input_text}**.

Guidelines:
- Use a mix of conceptual and real-world scenario questions.
- Include light critical thinking.
- Still no need for code, formulas, or complex diagrams.
Ensure the questions are clear, to the point, and suitable for a {difficulty_level}-level interview in {selected_domain}.
**New Requirement:**
🚫 **Do NOT repeat any questions from previous generations again and again.** Ensure all generated questions are unique and different from past sessions.

**Guidelines:**
    ✅ Questions should focus on key concepts, best practices, and problem-solving within {selected_domain}.
    ✅ Ensure questions are direct, structured, and relevant to real-world applications.
    ❌ Do NOT include greetings like 'Let's begin' or 'Welcome to the interview'.
    ❌ Avoid vague or open-ended statements—each question should be concise and specific.
"""

ADVANCED_PROMPT = """
You are a strict mock interviewer conducting an **Advanced-level** spoken interview in the domain of **{domain}**.
Ask deep, analytical, real-world scenario-based questions from the candidate's input: **{input_text}**.

Guidelines:
- Expect detailed, logical, well-structured answers.
- Include challenging “why” and “how” based questions.
- No need for code, but assume candidate has high expertise.
Ensure the questions are clear, to the point, and suitable for a {difficulty_level}-level interview in {selected_domain}.
**New Requirement:**
🚫 **Do NOT repeat any questions from previous generations again and again.** Ensure all generated questions are unique and different from past sessions.

**Guidelines:**
    ✅ Questions should focus on key concepts, best practices, and problem-solving within {selected_domain}.
    ✅ Ensure questions are direct, structured, and relevant to real-world applications.
    ❌ Do NOT include greetings like 'Let's begin' or 'Welcome to the interview'.
    ❌ Avoid vague or open-ended statements—each question should be concise and specific.
"""

########################################///////////////////////////////////////////////////#########################################
# UI styles
st.markdown("""
<style>
    /* Base style for all stButton elements */
    .stButton > button {
        background-color: #007BFF !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        width: 100% !important;
        padding: 0.4rem 0.75rem !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        border: 1px solid transparent !important;
        transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out !important;
        margin-bottom: 8px !important;
        box-sizing: border-box;
    }
    .stButton > button:hover {
        background-color: #0056b3 !important;
        color: white !important;
        border-color: #0056b3 !important;
    }
    .stButton > button:focus,
    .stButton > button:active {
        background-color: #0056b3 !important;
        border-color: #004085 !important;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.5) !important;
        outline: none !important;
    }

    .timer-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #00bcd4;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% {opacity: 1;}
        50% {opacity: 0.4;}
        100% {opacity: 1;}
    }

    .summary-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    /* More specific selector for the pre text color */
    div.summary-card > pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        font-family: inherit !important;
        font-size: 0.95rem !important;
        color: #000000 !important;   /* TRYING PURE BLACK with !important */
        background-color: #ffffff !important; /* Ensure background is white */
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        max-height: 400px !important;
        overflow-y: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; margin-top: -30px; padding-top: 10px;'>
    <h1 style='font-size: 2.8rem; font-weight: 800; color: #003366;'>🎯 Welcome to <span style='color: #007BFF;'>GrillMaster</span></h1>
    <p style='font-size: 1.1rem; color: #555;'>Your AI-powered mock interview assistant</p>
</div>
<hr style='border: 1px solid #e0e0e0; margin: 20px auto;'>
""", unsafe_allow_html=True)

if not st.session_state["generated_questions"]:
    st.markdown("""
    <div style='text-align: center; margin-top: -10px; margin-bottom: 30px;'>
        <h3 style='font-weight: 700; color: #333;'>🚀 Let's get started!</h3>
        <p style='font-size: 1rem; color: #666;'>Select your interview domain and input type to begin your practice session.</p>
    </div>
    <hr style='border: 1px solid #e0e0e0; margin-top: 0px;'>
    """, unsafe_allow_html=True)

# Example soft skills questions for HR/Soft Skills domain
if st.session_state["selected_domain"] == "Soft Skills":
        hr_questions = [
    "Tell me about yourself.",
    "Why should we hire you?",
    "What are your strengths and weaknesses?",
    "What is the difference between hard work and smart work?",
    "Why do you want to work at our company?",
    "How do you feel about working nights and weekends?",
    "Can you work under pressure?",
    "What are your goals?",
    "Are you willing to relocate or travel?",
    "What motivates you to do good job?",
    "What would you want to accomplish within your first 30 days of employment?",
    "What do you prefer working alone or in collaborative environment?",
    "Give me an example of your creativity.",
    "How long would you expect to work for us if hired?",
    "Are not you overqualified for this position?",
    "Describe your ideal company, location and job.",
    "Explain how would you be an asset to this organization?",
    "What are your interests?",
    "Would you lie for the company?",
    "Who has inspired you in your life and why?",
    "What was the toughest decision you ever had to make?",
    "Have you considered starting your own business?",
    "How do you define success and how do you measure up to your own definition?",
    "Tell me something about our company.",
    "How much salary do you expect?",
    "Where do you see yourself five years from now?",
    "Do you have any questions for me?",
    "Are you a manager or a leader?",
    "Imagine that you are not lucky enough to get this job, how will you take it?"
]

# === Sidebar: Domain and Input Configuration ===
st.sidebar.subheader("Select Interview Domain:")
for domain in ["Analytics", "Finance", "Soft Skills"]:
    if st.sidebar.button(domain):
        st.session_state.clear()  # 🔁 Reset entire session state
        st.session_state["selected_domain"] = domain
        st.rerun()

if not st.session_state["selected_domain"]:
    st.sidebar.info("Please select a domain to continue.")
    st.stop()

st.sidebar.markdown(f"**Selected Domain:** {st.session_state['selected_domain']}")
num_qs = st.sidebar.slider("Number of Questions:", 1, 10, 3)

if st.session_state["selected_domain"] == "Soft Skills":
    if st.sidebar.button("Generate Questions"):
        st.session_state["generated_questions"] = sample(hr_questions, num_qs)
        st.session_state["current_question_index"] = 0
        st.rerun()
else:
    section_choice = st.sidebar.radio("Choose Input Type:", ("Resume", "Job Description", "Skills"))
    difficulty = st.sidebar.selectbox("Select Difficulty Level:", ["Beginner", "Intermediate", "Advanced"])
    input_text = ""

    if section_choice == "Resume":
        uploaded_file = st.sidebar.file_uploader("Upload Resume:", type=["pdf", "txt"])
        if uploaded_file:
            input_text = extract_pdf_text(uploaded_file)

    elif section_choice == "Job Description":
        input_text = st.sidebar.text_area("Paste Job Description:")

    elif section_choice == "Skills":
        input_text = ""

        if st.session_state["selected_domain"] == "Finance":
            finance_levels = ["Level-1", "Level-2", "Level-3"]
            selected_level = st.sidebar.selectbox("Select a Finance Level:", finance_levels, key="finance_level_select")

            difficulty = st.session_state.get("difficulty", "Beginner")

            if selected_level != "Level-1":
                st.sidebar.warning(f"🚧 {selected_level} content is still under development. Please select Level-1 to continue.")
                st.stop()

            # Map difficulty level to column in Excel
            column_map = {
                "Beginner": "MODULE 1-EASY",
                "Intermediate": "MODULE 1-MEDIUM",
                "Advanced": "MODULE 1-DIFFICULT"
            }

            selected_column = column_map[difficulty]

            # Load Excel and questions
            excel_path = os.path.join("data", "CIBOP Mock Questions.xlsx")
            try:
                df = pd.read_excel(excel_path, engine="openpyxl")
                questions_from_excel = df[selected_column].dropna().astype(str).tolist()
                input_text = selected_column  # Optional, for tracking
            except Exception as e:
                st.sidebar.error(f"❌ Error reading Excel file: {e}")
                st.stop()

            st.sidebar.success(f"✅ Loaded {difficulty}-level questions from {selected_level}")

        else:
            # For Analytics or any other domain
            skills = {
                "Analytics": ["Python", "SQL", "Machine Learning", "Statistics", "Business Analytics"]
            }
            skill_list = skills.get(st.session_state["selected_domain"], [])
            if skill_list:
                selected_skill = st.sidebar.selectbox("Select a Skill:", skill_list, key="skill_select")
                input_text = selected_skill
                st.sidebar.markdown(f"✅ Selected Skill: **{selected_skill}**")


    if st.sidebar.button("Generate Questions"):
        if not input_text.strip():
            st.warning("⚠️ Please provide input based on the selected method.")
            st.stop()

        if st.session_state["selected_domain"] == "Finance" and section_choice == "Skills":
            st.session_state["generated_questions"] = sample(questions_from_excel, min(num_qs, len(questions_from_excel)))
        else:
            prompt = f"Ask {num_qs} direct and core-level {difficulty} interview questions related to {input_text}. Do not include intros or numbering."
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            response = model.generate_content([prompt, input_text])
            lines = response.text.strip().split("\n")
            questions = [q.strip("* ") for q in lines if q.strip()]
            st.session_state["generated_questions"] = questions[:num_qs]

        st.session_state["current_question_index"] = 0
        st.session_state["answers"] = []
        st.session_state["evaluation_feedback"] = ""
        st.session_state["recorded_text"] = ""
        st.session_state["response_captured"] = False
        st.session_state["timer_start"] = None
        st.session_state["show_summary"] = False
        st.session_state["question_played"] = False
        st.session_state["recording_complete"] = False
        st.rerun()

def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers



# === Main QA Interface ===
if st.session_state["generated_questions"]:
    idx = st.session_state["current_question_index"]
    if idx < len(st.session_state["generated_questions"]):
        question = st.session_state["generated_questions"][idx].lstrip("1234567890. ").strip()

        # Phase 0: Play audio first and wait 5s before countdown
        if not st.session_state.get("question_played"):
            st.session_state["question_audio_file"] = asyncio.run(generate_question_audio(question))
            st.session_state.update({
                "question_played": True,
                "question_start_time": time.time(),
                "record_phase": "audio_playing",
                "recorded_text": ""
            })

        st.markdown(f"**Q{idx + 1}:** {question}")
        st.audio(st.session_state["question_audio_file"], format="audio/mp3")

        now = time.time()
        elapsed = now - st.session_state.get("question_start_time", 0)

        if st.session_state["record_phase"] == "audio_playing":
            if elapsed < 5:
                st.markdown(f"<h4 class='timer-text'>🔊 Playing question audio... Please listen</h4>", unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()
            else:
                st.session_state["record_phase"] = "waiting_to_start"
                st.session_state["question_start_time"] = time.time()
                st.rerun()

        elif st.session_state["record_phase"] == "waiting_to_start":
            remaining = 10 - int(elapsed)
            if remaining > 0:
                st.markdown(f"<h4 class='timer-text'>⏳ {remaining} seconds to click 'Start Recording'...</h4>", unsafe_allow_html=True)
                if st.button("🎙️ Start Recording"):
                    st.session_state.update({
                        "record_phase": "recording",
                        "timer_start": time.time(),
                        "recording_started": False
                    })
                    st.rerun()
                time.sleep(1)
                st.rerun()
            else:
                st.markdown("<div style='padding:10px; background:#fff8e1; border-left:5px solid orange;color: #212529;'>⚠️ <strong>No action detected.</strong> Automatically skipping to next question...</div>", unsafe_allow_html=True)
                st.session_state["answers"].append({"question": question, "response": "[No response]"})
                st.session_state.update({
                    "record_phase": "idle",
                    "question_played": False,
                    "question_start_time": 0.0,
                    "current_question_index": idx + 1
                })
                if st.session_state["current_question_index"] == len(st.session_state["generated_questions"]):
                    evaluate_answers()
                    st.session_state["show_summary"] = True
                st.rerun()

        elif st.session_state["record_phase"] == "recording":
            st.markdown(f"<h4 class='timer-text'>🎙️ Recording... (Speak now and wait to auto-save)</h4>", unsafe_allow_html=True)

            webrtc_ctx = webrtc_streamer(
                key=f"record_{idx}",
                mode=WebRtcMode.SENDRECV,
                audio_receiver_size=1024,
                media_stream_constraints={"video": False, "audio": True},
                async_processing=True
            )

            if webrtc_ctx.state.playing:
                if webrtc_ctx.audio_receiver:
                    try:
                        frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
                        wav_path = f"response_{idx}.wav"
                        convert_frames_to_wav(frames, wav_path)
                        st.audio(wav_path)

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_path) as source:
                            audio = recognizer.record(source)
                            transcript = recognizer.recognize_google(audio)

                        st.success("✅ Transcription completed.")
                        st.session_state["answers"].append({
                            "question": question,
                            "response_file": wav_path,
                            "response": transcript
                        })

                        st.session_state.update({
                            "record_phase": "idle",
                            "recording_started": False,
                            "question_played": False,
                            "question_start_time": 0.0,
                            "current_question_index": idx + 1
                        })

                        if st.session_state["current_question_index"] == len(st.session_state["generated_questions"]):
                            evaluate_answers()
                            st.session_state["show_summary"] = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"🎤 Recording error or timeout: {e}")

                else:
                    st.warning("⚠️ No audio frames received yet. Please speak clearly into the mic.")

            # Time-out after 15 seconds if nothing recorded
            elapsed = time.time() - st.session_state.get("timer_start", 0)
            if elapsed > 15:
                st.warning("⏱️ Recording timed out. Moving to next question.")
                st.session_state["answers"].append({
                    "question": question,
                    "response": "[No response]"
                })
                st.session_state.update({
                    "record_phase": "idle",
                    "recording_started": False,
                    "question_played": False,
                    "question_start_time": 0.0,
                    "current_question_index": idx + 1
                })
                if st.session_state["current_question_index"] == len(st.session_state["generated_questions"]):
                    evaluate_answers()
                    st.session_state["show_summary"] = True
                st.rerun()

                        
                
                                
            else:
                st.markdown("<div style='padding:10px; background:#fff3e0; border-left:5px solid orange;'>⚠️ <strong>No response detected.</strong> Moving to next question...</div>", unsafe_allow_html=True)
                st.session_state["answers"].append({"question": question, "response": "[No response]"})
                st.session_state.update({
                    "record_phase": "idle",
                    "recording_started": False,
                    "question_played": False,
                    "question_start_time": 0.0,
                    "current_question_index": idx + 1
                })
                if st.session_state["current_question_index"] == len(st.session_state["generated_questions"]):
                    evaluate_answers()
                    st.session_state["show_summary"] = True
                st.rerun()

        elif st.session_state["record_phase"] == "listening":
            st.success("🎧 Review your recorded response below:")
            st.audio(st.session_state["response_file"], format="audio/wav")
            
            if st.button("⏹️ Confirm & Next"):
                st.session_state["answers"].append({
                    "question": question,
                    "response_file": st.session_state["response_file"]
                })

                st.session_state.update({
                    "record_phase": "idle",
                    "recording_started": False,
                    "question_played": False,
                    "question_start_time": 0.0,
                    "current_question_index": idx + 1,
                    "response_file": None,
                    "audio_waiting": True
                })
                
                if st.session_state["current_question_index"] == len(st.session_state["generated_questions"]):
                    evaluate_answers()
                    st.session_state["show_summary"] = True
                st.rerun()

# === Summary Display ===
if st.session_state.get("show_summary", False):
    st.subheader("📊 Complete Mock Interview Summary")

    # Fetch values from session state, providing defaults
    feedback_content_for_display = st.session_state.get('evaluation_feedback', "Evaluation details not available.")
    if not isinstance(feedback_content_for_display, str):
        feedback_content_for_display = str(feedback_content_for_display)
    
    # Max score basis is the number of questions that were *generated* for the session
    num_qs_in_session = len(st.session_state.get("generated_questions", [])) 
    if num_qs_in_session == 0 and st.session_state.get("answers"): # Fallback if no generated_questions but answers exist
        num_qs_in_session = len(st.session_state.answers)

    if st.session_state["selected_domain"] == "Soft Skills":
        num_qs_in_session = len(st.session_state.get("answers", []))
        max_score_possible_for_session = num_qs_in_session * 5.0

    else:
        if st.session_state["selected_domain"] == "Soft Skills":
            num_hr_params = len(st.session_state.get("hr_parameter_scores_dict", {}))
            max_score_possible_for_session = num_hr_params * 5.0
        else:
            max_score_possible_for_session = num_qs_in_session * 5.0

    #max_score_possible_for_session = num_qs_in_session * 5.0
    current_percentage_score = st.session_state.get('percentage_score', 0.0)
    current_overall_score = st.session_state.get('overall_score', 0.0)

    if st.session_state["selected_domain"] == "Soft Skills":
        hr_table_data = []
        for param, config in HR_PARAMETERS_CONFIG.items():
            score = st.session_state.get("hr_parameter_scores_dict", {}).get(param, 0.0)
            weight_percent = config["weight_original"]
            contribution = (score / 5.0) * config["weight_normalized"]
            hr_table_data.append({
                "Parameter": param,
                "Weight (Original %)": f"{weight_percent}%",
                "Score (1–5)": round(score, 1),
                "Contribution to Final %": f"{contribution:.1f}%"
            })

        hr_table_data.append({
            "Parameter": "Total",
            "Weight (Original %)": "100%",
            "Score (1–5)": "",
            "Contribution to Final %": f"{current_percentage_score:.1f}%"
            })

        hr_df = pd.DataFrame(hr_table_data)
        st.markdown("### 🧾 Soft Skills Evaluation Breakdown")
        st.dataframe(hr_df, use_container_width=True)

    # Display the calculated score and percentage bar first in a card
    st.markdown(f"""
        <div class='summary-card' style="margin-bottom: 20px;">
            <h4 style="color: #212529;">✅ <strong>Overall Score:</strong> {current_overall_score:.1f} / {max_score_possible_for_session:.1f} 
                ({current_percentage_score:.1f}%)
            </h4>
            <div style='margin:10px 0; position:relative;'>
                <div style="background:#eee; border-radius:10px; overflow:hidden; height:30px; position:relative;">
                    <div style="
                        width:{current_percentage_score}%; 
                        background:#00c851; /* Green for progress */
                        height:100%;
                        border-radius:10px 0 0 10px; /* Keep left radius for progress */
                        transition: width 0.4s ease-in-out;
                    "></div>
                    <div style="
                        position:absolute;
                        top:0;
                        left:0;
                        width:100%;
                        height:100%;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-weight:bold;
                        color: black !important; /* Ensure text is visible on green/grey */
                        font-size: 0.9rem;
                        user-select:none; /* Prevent text selection */
                    ">
                        {current_percentage_score:.1f}% 
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Display the detailed evaluation feedback text in a separate section
    st.markdown("---") 
    st.markdown("<h4 style='color: #212529;'>Detailed Evaluation & Feedback from AI:</h4>", unsafe_allow_html=True)
    
    # Use a styled div for the feedback text block to ensure good readability
    # Replace newlines with <br> for proper HTML multiline display
    html_formatted_feedback = feedback_content_for_display.replace('\n', '<br>')
    st.markdown(f"""
    <div style="background-color: #ffffff; color: #212529; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-top:10px; max-height: 500px; overflow-y: auto; white-space: normal; word-wrap: break-word;">
    {html_formatted_feedback}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---") # Separator

    # Buttons for suggestions, download, practice
    cols_summary_buttons = st.columns([1, 1, 1]) # 3 columns for the buttons

    with cols_summary_buttons[0]:
        if st.button("💡 Get Improvement Suggestions", key="get_suggestions_btn_final", use_container_width=True):
            # Regenerate suggestions if not present or explicitly requested again
            generate_improvement_suggestions() # This function should handle st.info/st.success
            st.rerun() # Rerun to show the expander or updated suggestions

    # Helper function to prepare summary text for download
    def prepare_summary_for_download():
        download_text = f"# GrillMaster Mock Interview Summary\n\n"
        download_text += f"**Selected Domain:** {st.session_state.get('selected_domain', 'N/A')}\n"
        dl_difficulty = st.session_state.get('difficulty_level_select', 'N/A')
        download_text += f"**Difficulty Level:** {dl_difficulty}\n"
        
        num_q_for_max_score = len(st.session_state.get("generated_questions", st.session_state.get("answers",[])))
        max_s_for_dl = num_q_for_max_score * 5.0
        
        download_text += f"**Calculated Overall Score:** {st.session_state.get('overall_score', 0.0):.1f} / {max_s_for_dl:.1f} ({st.session_state.get('percentage_score', 0.0):.1f}%)\n\n"
        
        download_text += "## Questions & Candidate's Answers:\n"
        num_answers_actually_given = len(st.session_state.get("answers", []))
        for i in range(num_q_for_max_score):
            question_text_dl = st.session_state.generated_questions[i] if i < len(st.session_state.generated_questions) else "Question text not found"
            answer_text_dl = "[No answer recorded]"
            if i < num_answers_actually_given:
                answer_text_dl = str(st.session_state.answers[i].get('response', '[No response provided]'))
            
            download_text += f"**Question {i+1}:** {question_text_dl}\n"
            download_text += f"**Your Answer {i+1}:** {answer_text_dl}\n\n"
        
        download_text += "\n## AI Evaluation Details (Includes Parsed Scores and Qualitative Feedback):\n"
        # st.session_state.evaluation_feedback is now already pre-formatted
        download_text += st.session_state.get('evaluation_feedback', "No AI evaluation available.")
        download_text += "\n\n"

        if st.session_state.get("improvement_suggestions_generated", False) and st.session_state.get("improvement_suggestions"):
            download_text += "\n## Detailed Improvement Suggestions from AI:\n"
            download_text += st.session_state.get('improvement_suggestions', "No improvement suggestions were generated.")
        
        return download_text.encode('utf-8')

    with cols_summary_buttons[1]:
        summary_bytes_dl_final = prepare_summary_for_download()
        st.download_button(
            label="💾 Download Full Summary", 
            data=summary_bytes_dl_final,
            file_name=f"GrillMaster_Summary_{st.session_state.get('selected_domain','General')}_{time.strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown", 
            key="download_summary_final_btn", 
            use_container_width=True
        )
    
    

    # Expander for detailed suggestions, shown if generated
    if st.session_state.get("improvement_suggestions_generated", False) and st.session_state.get("improvement_suggestions"):
        with st.expander("🔍 View Detailed Improvement Suggestions", expanded=True): # Default to expanded once generated
            st.markdown(st.session_state.improvement_suggestions, unsafe_allow_html=True) # LLM might use markdown

    # Conditional button for low scores
    if current_percentage_score < 50.0:
        st.warning(f"Your score is {current_percentage_score:.1f}%. Keep practicing! You can also reset all settings to try a new domain or difficulty.")
        if st.button("🔁 Practice Again & Reset All Settings", key="practice_full_reset_final_btn", use_container_width=True):
            # Clear all session state keys and re-initialize to defaults
            keys_to_fully_clear = list(st.session_state.keys())
            for key_to_del_full in keys_to_fully_clear:
                del st.session_state[key_to_del_full]
           
