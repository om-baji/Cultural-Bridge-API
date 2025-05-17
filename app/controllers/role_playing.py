import re
from venv import logger

from fastapi import HTTPException
from typing import Any, List, Dict
import ollama
import asyncio
from dotenv import load_dotenv
import datetime

from app.controllers.conflict_resolution import analyze_sentiment
from app.schemas.schema import RolePlayRequest, StoryResponse, EvaluationRequest, EvaluationResponse
from app.db.singleton import ChromaDBSingleton

load_dotenv()

chroma_client = ChromaDBSingleton()
chroma_collection = chroma_client.get_collection()


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


async def generate_context_aware_actions(scene: str, chat_history: List[Dict[str, str]], client: Any) -> List[str]:
    """Generate contextually relevant suggested actions based on the current scene and chat history."""
    try:
        # Prepare a prompt that asks the model to generate contextually appropriate actions
        full_conversation = ""
        for turn in chat_history:
            full_conversation += f"User: {turn['user']}\n"
            full_conversation += f"AI: {turn['ai']}\n\n"

        prompt = (
            "Based on the following role-playing conversation and the current scene, "
            "suggest 4 specific, contextually relevant actions the user could take next. "
            "These should be clear, concise phrases (5-10 words each) that would make sense given the narrative context.\n\n"
            f"Conversation so far:\n{full_conversation}\n\n"
            f"Current scene:\n{scene}\n\n"
            "Generate 4 contextually relevant actions the user could take next. "
            "Format each action on a new line with no numbering or bullets. "
            "Each action should be a specific, clear phrase that makes sense in the current context."
        )

        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7}
        )

        # Process the response to extract the actions
        action_text = response['message']['content'].strip()
        actions = [action.strip() for action in action_text.split('\n') if action.strip()]

        # Take the first 4 actions, or pad if fewer than 4 were generated
        actions = actions[:4]
        while len(actions) < 4:
            actions.append("Continue the conversation...")

        return actions

    except Exception as e:
        # If something goes wrong, return some generic fallback actions
        return [
            "Ask a follow-up question",
            "Share your perspective",
            "Request more information",
            "Change the subject"
        ]


async def generate_role_play(request: RolePlayRequest, client: Any):
    try:
        system_prompt = (
            f"You are role-playing as a {request.role} from the {request.culture} culture, "
            f"during the {request.era} era. You respond based on that role only. "
            f"Maintain historical and cultural accuracy. Use a {request.tone} tone and write in {request.language}. "
            f"{'Include emotional and reflective thoughts as well.' if request.include_emotion else ''}"
        )

        history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in request.chat_history:
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": turn["ai"]})

        history.append({"role": "user", "content": request.user_input})

        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=history,
            options={"temperature": 0.75, "top_p": 0.9}
        )

        reply = response['message']['content'].strip()
        embeddings = await get_embeddings(reply, client)

        # Generate contextually relevant actions
        suggested_actions = await generate_context_aware_actions(reply, request.chat_history + [
            {"user": request.user_input, "ai": reply}], client)

        metadata = {
            "mode": "role-play",
            "culture": request.culture,
            "role": request.role,
            "era": request.era,
            "tone": request.tone,
            "language": request.language
        }

        chroma_collection.add(
            documents=[reply],
            embeddings=[embeddings],
            metadatas=[metadata],
            ids=[request.role + "-role-" + str(datetime.datetime.now().timestamp())]
        )

        return StoryResponse(
            story=reply,
            character_count=len(reply),
            language=request.language,
            metadata=metadata,
            used_rag=False,
            reference_count=0,
            actions=suggested_actions  # Include the context-aware actions in the response
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in role-play generation: {str(e)}")


async def evaluate_chat_history(request: EvaluationRequest, client: Any):
    try:
        # Combine conversation history into a readable format
        full_conversation = ""
        for turn in request.chat_history:
            full_conversation += f"User: {turn['user']}\n"
            full_conversation += f"AI: {turn['ai']}\n\n"

        # Get sentiment analysis for the entire conversation
        overall_sentiment = 0
        sentiment_modifier = 0

        # If we have sentiment analysis capability, use it
        try:
            combined_text = " ".join([f"{turn['user']} {turn['ai']}" for turn in request.chat_history])
            overall_sentiment = analyze_sentiment(combined_text)
            sentiment_modifier = int(overall_sentiment * 10)  # Convert to -10 to +10 scale
        except Exception as e:
            print(f"Sentiment analysis not available: {str(e)}")

        # Apply faction-specific modifiers if available
        faction_modifier = 0
        if hasattr(request, 'player_faction') and request.player_faction == "neutral":
            faction_modifier = 5  # Bonus for taking neutral role

        # Set up improved evaluation prompt with detailed rubrics
        prompt = (
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
            "Based on the conversation below, provide numeric scores for each category.\n\n"
            f"{full_conversation}\n\n"
            "IMPORTANT: Be critical and realistic in your assessment. Not all approaches succeed, and failed attempts should receive appropriate scores.\n"
            "Respond in this exact format (with ONLY the scores and no additional text):\n"
            "EMPATHY: [score]\n"
            "DIPLOMATIC_SKILL: [score]\n"
            "HISTORICAL_ACCURACY: [score]\n"
            "ETHICAL_BALANCE: [score]\n"
        )

        # Additional context if available
        if hasattr(request, 'conflict_type') and request.conflict_type:
            conflict_context = {
                "india_pakistan": "the 1947 India-Pakistan partition with tension over borders, refugees, and religious differences",
                "israeli_palestinian": "the Israeli-Palestinian conflict with disputes over territory, security, and self-determination",
                "indigenous_rights": "Indigenous rights movements facing challenges of land rights, sovereignty, and cultural preservation",
                "northern_ireland": "the Northern Ireland conflict (The Troubles) with tension between unionists and nationalists",
                "rwanda": "the ethnic tensions in Rwanda leading up to and following the 1994 genocide"
            }
            context = conflict_context.get(request.conflict_type, "a historical conflict")
            prompt = f"For context, this conversation is about {context}.\n\n{prompt}"

        # Get evaluation from LLM with reduced temperature for consistency
        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}  # Reduced temperature for more consistent scoring
        )

        evaluation_text = response['message']['content'].strip()

        # Parse scores using regex for more robust extraction
        scores = {}
        empathy_match = re.search(r"EMPATHY: (\d+)", evaluation_text)
        diplomatic_match = re.search(r"DIPLOMATIC_SKILL: (\d+)", evaluation_text)
        historical_match = re.search(r"HISTORICAL_ACCURACY: (\d+)", evaluation_text)
        ethical_match = re.search(r"ETHICAL_BALANCE: (\d+)", evaluation_text)

        scores["EMPATHY"] = int(empathy_match.group(1)) if empathy_match else 15
        scores["DIPLOMATIC_SKILL"] = int(diplomatic_match.group(1)) if diplomatic_match else 15
        scores["HISTORICAL_ACCURACY"] = int(historical_match.group(1)) if historical_match else 10
        scores["ETHICAL_BALANCE"] = int(ethical_match.group(1)) if ethical_match else 10

        # Apply sentiment and faction modifiers
        scores["EMPATHY"] = max(0, min(30, scores["EMPATHY"] + sentiment_modifier))
        scores["DIPLOMATIC_SKILL"] = max(0, min(30, scores["DIPLOMATIC_SKILL"] + sentiment_modifier))
        scores["HISTORICAL_ACCURACY"] = max(0, min(20, scores["HISTORICAL_ACCURACY"] + faction_modifier))

        # Calculate total score
        total_score = sum(scores.values())

        # If the evaluation contains detailed feedback, extract it
        feedback = {}
        sections = evaluation_text.split("\n\n")
        if len(sections) > 1:
            for section in sections:
                for category in ["EMPATHY", "DIPLOMATIC_SKILL", "HISTORICAL_ACCURACY", "ETHICAL_BALANCE"]:
                    if category in section:
                        # Extract any text after the score as feedback
                        pattern = rf"{category}: \d+(.*?)(?=\n\n|\Z)"
                        match = re.search(pattern, section, re.DOTALL)
                        if match and match.group(1).strip():
                            feedback[category] = match.group(1).strip()

        # Store evaluation results in vector database if available
        try:
            if chroma_collection and request.session_id:
                metadata = {
                    "mode": "evaluation",
                    "session_id": request.session_id,
                    "empathy_score": scores["EMPATHY"],
                    "diplomatic_skill_score": scores["DIPLOMATIC_SKILL"],
                    "historical_accuracy_score": scores["HISTORICAL_ACCURACY"],
                    "ethical_balance_score": scores["ETHICAL_BALANCE"],
                    "total_score": total_score,
                    "sentiment_score": overall_sentiment,
                    "evaluation_timestamp": datetime.datetime.now().isoformat()
                }

                if hasattr(request, 'conflict_type'):
                    metadata["conflict_type"] = request.conflict_type
                if hasattr(request, 'player_faction'):
                    metadata["faction"] = request.player_faction
                if hasattr(request, 'player_role'):
                    metadata["role"] = request.player_role

                # Get embeddings for the conversation
                conversation_embedding = await get_embeddings(full_conversation, client)

                # Store in ChromaDB
                chroma_collection.add(
                    documents=[full_conversation],
                    embeddings=[conversation_embedding],
                    metadatas=[metadata],
                    ids=[f"eval-{request.session_id}-{str(datetime.datetime.now().timestamp())}"]
                )
        except Exception as e:
            print(f"Warning: Could not store evaluation in vector database: {str(e)}")

        return EvaluationResponse(
            empathy_score=scores["EMPATHY"],
            diplomatic_skill_score=scores["DIPLOMATIC_SKILL"],
            historical_accuracy_score=scores["HISTORICAL_ACCURACY"],
            ethical_balance_score=scores["ETHICAL_BALANCE"],
            total_score=total_score,
            feedback=feedback if feedback else None,
            sentiment_score=overall_sentiment if overall_sentiment != 0 else None
        )

    except Exception as e:
        logger.error(f"Something went wrong!: {str(e)}")