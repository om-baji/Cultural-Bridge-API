import logging
from fastapi import HTTPException,FastAPI
from typing import Any, List, Dict
import ollama
import asyncio
from dotenv import load_dotenv
import datetime
import uuid
import re
import numpy as np
import tensorflow as tf
import joblib
import os

from app.schemas.schema import ConflictRequest, ConflictResponse, KalkiScore
from app.db.singleton import ChromaDBSingleton

app = FastAPI()

load_dotenv()

TFIDF_PATH = os.getenv("TFIDF_VECTORIZER_PATH", "app/models/tfidf_vectorizer.joblib")
SENTIMENT_MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH", "app/models/sentiment_model.h5")

chroma_client = ChromaDBSingleton()
chroma_collection = chroma_client.get_collection()

# Load the sentiment model and TF-IDF vectorizer
sentiment_model = None
tfidf_vectorizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    global sentiment_model, tfidf_vectorizer
    try:
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        sentiment_model = tf.keras.models.load_model(SENTIMENT_MODEL_PATH)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.exception("Error loading models: %s", str(e))

async def get_model_client():
    return ollama

async def get_embeddings(text: str, client: Any = None):
    if client is None:
        client = await get_model_client()

    response = await asyncio.to_thread(
        client.embeddings,
        model="all-minilm:33m",
        prompt=text
    )
    return response["embedding"]


def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment using the loaded sentiment model.
    Returns a score between -1 (very negative) and 1 (very positive).
    """
    global sentiment_model, tfidf_vectorizer

    if sentiment_model is None or tfidf_vectorizer is None:
        if sentiment_model is None or tfidf_vectorizer is None:
            # Fallback if models still fail to load
            return 0.0

    try:
        # Transform text using TF-IDF
        text_tfidf = tfidf_vectorizer.transform([text])

        # Get sentiment prediction
        prediction = sentiment_model.predict(text_tfidf.toarray())

        # Convert to a sentiment score between -1 and 1
        # Assuming model output is between 0 and 1
        sentiment_score = (prediction[0][0] * 2) - 1

        return float(sentiment_score)
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return 0.0


async def generate_conflict_scenario(request: ConflictRequest, client: Any):
    try:
        conflict_context = {
            "india_pakistan": "the 1947 India-Pakistan partition with tension over borders, refugees, and religious differences",
            "israeli_palestinian": "the Israeli-Palestinian conflict with disputes over territory, security, and self-determination",
            "indigenous_rights": "Indigenous rights movements facing challenges of land rights, sovereignty, and cultural preservation",
            "northern_ireland": "the Northern Ireland conflict (The Troubles) with tension between unionists and nationalists",
            "rwanda": "the ethnic tensions in Rwanda leading up to and following the 1994 genocide"
        }

        faction_description = {
            "side_a": "representing the first main party in the conflict",
            "side_b": "representing the second main party in the conflict",
            "neutral": "as a neutral third party attempting to facilitate peace"
        }

        # Set up system prompt to guide AI behavior
        system_prompt = (
            f"You are simulating a conflict resolution scenario for {conflict_context.get(request.conflict_type, 'a historical conflict')}. "
            f"The user is playing as a {request.player_role} {faction_description.get(request.player_faction, '')}. "
            f"Current tension level is {request.tension_level}/100. "
            f"Provide realistic consequences to the user's actions, detailing how they affect the conflict. "
            f"Include decisions other parties might make in response. "
            f"If the user makes choices that would realistically escalate tensions, reflect that in your response. "
            f"If they make de-escalatory choices, show progress toward resolution. "
            f"Maintain historical accuracy while allowing for counterfactual scenarios based on user choices. "
            f"Important: Include both positive and negative developments as appropriate to the context - not all conflicts resolve easily, "
            f"and diplomatic efforts can backfire or be undermined by external factors."
        )

        # Convert past interactions into chat history
        history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in request.chat_history:
            history.append({"role": "user", "content": turn.user})
            history.append({"role": "assistant", "content": turn.ai})

        # Add current user input
        history.append({"role": "user", "content": request.user_input})

        # Call Ollama with the message history
        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=history,
            options={"temperature": 0.7, "top_p": 0.9}
        )

        reply = response['message']['content'].strip()

        # Calculate new tension level using sentiment analysis
        new_tension = calculate_tension_with_sentiment(
            request.tension_level,
            reply,
            request.user_input,
            request.player_faction
        )

        # Determine if scenario has reached a conclusion
        is_concluded = check_conclusion(new_tension, request.current_stage)

        # Generate next available actions based on new tension
        next_actions = generate_next_actions(new_tension, request.player_faction, request.player_role)

        # Calculate KALKI score if concluded
        kalki_score = None
        if is_concluded:
            kalki_score = await calculate_kalki_score(
                request.chat_history,
                request.user_input,
                reply,
                client,
                request.player_faction
            )

        # Store interaction in vector database
        embeddings = await get_embeddings(reply, client)
        metadata = {
            "mode": "conflict-resolution",
            "conflict_type": request.conflict_type,
            "role": request.player_role,
            "faction": request.player_faction,
            "tension_level": new_tension,
            "stage": request.current_stage,
            "sentiment_score": analyze_sentiment(reply)
        }

        chroma_collection.add(
            documents=[reply],
            embeddings=[embeddings],
            metadatas=[metadata],
            ids=[f"conflict-{request.session_id}-{str(datetime.datetime.now().timestamp())}"]
        )

        return ConflictResponse(
            response=reply,
            tension_level=new_tension,
            current_stage=request.current_stage + (0 if not is_concluded else 1),
            available_actions=next_actions,
            is_concluded=is_concluded,
            metadata=metadata,
            session_id=request.session_id or str(uuid.uuid4()),
            kalki_score=kalki_score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in conflict simulation: {str(e)}")


def calculate_tension_with_sentiment(current_tension: int, ai_response: str, user_input: str, faction: str) -> int:
    # Get sentiment scores for both AI response and user input
    ai_sentiment = analyze_sentiment(ai_response)
    user_sentiment = analyze_sentiment(user_input)

    # Weight the sentiment scores based on faction
    # Neutral parties have less impact on tension
    faction_weight = 1.0
    if faction == "neutral":
        faction_weight = 0.7

    # Combined weighted sentiment (-1 to 1 scale)
    combined_sentiment = (ai_sentiment * 0.6 + user_sentiment * 0.4) * faction_weight

    # Convert sentiment to tension change (-20 to +20 scale)
    # Negative sentiment increases tension, positive sentiment decreases it
    tension_change = -combined_sentiment * 20

    # Add context-specific modifiers using TF-IDF analysis
    combined_text = f"{ai_response} {user_input}"

    # Check for contextual tension indicators
    escalation_indicators = [
        "military", "troops", "violence", "attack", "protest", "riot",
        "conflict", "dispute", "tension", "hostility", "threat", "weapon",
        "ultimatum", "deadline", "sanction", "force", "demand"
    ]

    deescalation_indicators = [
        "peace", "agreement", "treaty", "compromise", "negotiate", "cooperate",
        "collaborate", "understand", "reconcile", "dialogue", "diplomacy",
        "ceasefire", "handshake", "concession", "mediate"
    ]

    # Check for presence of indicators and adjust tension accordingly
    for word in escalation_indicators:
        if word in combined_text.lower():
            tension_change += 2

    for word in deescalation_indicators:
        if word in combined_text.lower():
            tension_change -= 2

    # Random factor to make tension changes less predictable
    random_factor = np.random.normal(0, 3)  # Mean 0, std dev 3
    tension_change += random_factor

    # Apply change and ensure within bounds
    new_tension = max(0, min(100, current_tension + tension_change))
    return int(new_tension)


def check_conclusion(tension: int, stage: int) -> bool:
    # More nuanced conclusion check
    if tension <= 10:
        # Low tension - likely peaceful resolution
        conclusion_probability = 0.8
    elif tension >= 90:
        # High tension - likely breakdown/conflict
        conclusion_probability = 0.9
    elif stage >= 4:
        # Later stages have increasing chance of conclusion
        conclusion_probability = 0.3 * (stage - 3)
    else:
        # Early stages with moderate tension unlikely to conclude
        conclusion_probability = 0.1

    # Random chance to determine if scenario concludes
    return np.random.random() < conclusion_probability


async def calculate_kalki_score(
        chat_history: List[Dict[str, str]],
        user_input: str,
        ai_response: str,
        client: Any,
        faction: str
) -> KalkiScore:
    # Combine all conversation for context
    full_conversation = ""
    for turn in chat_history:
        full_conversation += f"User: {turn['user']}\nAI: {turn['ai']}\n\n"
    full_conversation += f"User: {user_input}\nAI: {ai_response}"

    # Get sentiment of final exchange to influence scoring
    final_sentiment = analyze_sentiment(user_input + " " + ai_response)
    sentiment_modifier = int(final_sentiment * 10)  # -10 to +10 scale

    # Factor in which faction the user was playing
    faction_modifier = 0
    if faction == "neutral":
        faction_modifier = 5  # Bonus for taking neutral role

    # Set up prompt for KALKI scoring evaluation with improved rubric
    evaluation_prompt = (
        "Evaluate the user's conflict resolution approach based on the KALKI scoring system:\n\n"
        "1. EMPATHY (0-30): Did the user consider multiple perspectives? Score higher if they demonstrated understanding of all sides.\n"
        "   - Low (0-10): Showed no understanding of opposing viewpoints\n"
        "   - Medium (11-20): Some acknowledgment of other perspectives\n"
        "   - High (21-30): Deep understanding of multiple viewpoints\n\n"
        "2. DIPLOMATIC SKILL (0-30): Did the user promote peaceful negotiation? Score higher for constructive dialogue and compromise.\n"
        "   - Low (0-10): Confrontational or inflexible approach\n"
        "   - Medium (11-20): Some attempt at negotiation but with limitations\n"
        "   - High (21-30): Skilled diplomacy with concrete proposals\n\n"
        "3. HISTORICAL ACCURACY (0-20): Were the user's decisions informed by real-world lessons? Score higher for realistic approaches.\n"
        "   - Low (0-7): Historically inaccurate or unrealistic\n"
        "   - Medium (8-14): Generally aligned with historical context\n"
        "   - High (15-20): Sophisticated understanding of historical dynamics\n\n"
        "4. ETHICAL BALANCE (0-20): Did the user avoid bias and maintain ethical principles? Score higher for fair solutions.\n"
        "   - Low (0-7): One-sided or ethically questionable approach\n"
        "   - Medium (8-14): Some ethical considerations but with gaps\n"
        "   - High (15-20): Strong ethical framework with consistent principles\n\n"
        "Based on the conversation below, provide numeric scores for each category and a brief explanation of each score.\n\n"
        f"{full_conversation}\n\n"
        "IMPORTANT: Be critical and realistic in your assessment. Not all approaches succeed, and failed attempts should receive appropriate scores.\n"
        "Respond in this exact format (with ONLY the scores and no additional text):\n"
        "EMPATHY: [score]\n"
        "DIPLOMATIC_SKILL: [score]\n"
        "HISTORICAL_ACCURACY: [score]\n"
        "ETHICAL_BALANCE: [score]\n"
    )

    # Get evaluation from LLM
    eval_response = await asyncio.to_thread(
        client.generate,
        model="llama3:latest",
        prompt=evaluation_prompt,
        options={"temperature": 0.3}
    )

    eval_text = eval_response['response'].strip()

    # Extract scores using regex
    empathy_match = re.search(r"EMPATHY: (\d+)", eval_text)
    diplomatic_match = re.search(r"DIPLOMATIC_SKILL: (\d+)", eval_text)
    historical_match = re.search(r"HISTORICAL_ACCURACY: (\d+)", eval_text)
    ethical_match = re.search(r"ETHICAL_BALANCE: (\d+)", eval_text)

    empathy = int(empathy_match.group(1)) if empathy_match else 15
    diplomatic = int(diplomatic_match.group(1)) if diplomatic_match else 15
    historical = int(historical_match.group(1)) if historical_match else 10
    ethical = int(ethical_match.group(1)) if ethical_match else 10

    # Apply sentiment and faction modifiers
    empathy = max(0, min(30, empathy + sentiment_modifier))
    diplomatic = max(0, min(30, diplomatic + sentiment_modifier))
    historical = max(0, min(20, historical + faction_modifier))
    ethical = max(0, min(20, ethical))

    total = empathy + diplomatic + historical + ethical

    return KalkiScore(
        empathy=empathy,
        diplomatic_skill=diplomatic,
        historical_accuracy=historical,
        ethical_balance=ethical,
        total_score=total
    )


def generate_next_actions(tension: int, faction: str, role: str) -> List[str]:
    """Generate contextually appropriate next actions based on tension level and role"""
    common_actions = ["Negotiate", "Make public statement", "Propose solution"]

    # High tension actions (war footing)
    if tension > 80:
        if faction in ["side_a", "side_b"]:
            return common_actions + [
                "Mobilize military forces",
                "Issue ultimatum",
                "Cut diplomatic ties",
                "Seek international support"
            ]
        else:  # neutral
            return common_actions + [
                "Call emergency meeting",
                "Propose sanctions",
                "Threaten to withdraw mediation",
                "Appeal to international community"
            ]

    # High-moderate tension
    elif tension > 60:
        if faction in ["side_a", "side_b"]:
            return common_actions + [
                "Show of force",
                "Recall ambassador",
                "Implement trade restrictions",
                "Appeal to allies"
            ]
        else:
            return common_actions + [
                "Impose deadline",
                "Threaten economic consequences",
                "Propose neutral peacekeeping force"
            ]

    elif tension > 40:
        if faction in ["side_a", "side_b"]:
            return common_actions + [
                "Request international mediation",
                "Hold protest rally",
                "Release intelligence information"
            ]
        else:
            return common_actions + [
                "Hold private talks",
                "Propose confidence-building measures",
                "Offer economic incentives"
            ]

    elif tension > 20:
        if faction in ["side_a", "side_b"]:
            return common_actions + [
                "Propose joint committee",
                "Offer minor concession",
                "Initiate cultural exchange"
            ]
        else:
            return common_actions + [
                "Organize peace conference",
                "Propose step-by-step process",
                "Suggest third-party verification"
            ]

    else:
        if faction in ["side_a", "side_b"]:
            return common_actions + [
                "Sign agreement",
                "Make symbolic gesture",
                "Form joint institution"
            ]
        else:
            return common_actions + [
                "Draft peace treaty",
                "Celebrate progress",
                "Establish monitoring mechanism"
            ]