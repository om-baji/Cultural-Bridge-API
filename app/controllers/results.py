# First, let's add a schema for the results response
# Add this to your app/schemas/schema.py file

from pydantic import BaseModel
from typing import List, Dict, Optional

# Add this to your existing schema definitions
class KalkiScore(BaseModel):
    empathy: int  # 0-30
    diplomatic_skill: int  # 0-30
    historical_accuracy: int  # 0-20
    ethical_balance: int  # 0-20
    total_score: int
    feedback: Dict[str, str]


class ResultsResponse(BaseModel):
    total_score: int
    individual_scores: KalkiScore
    performance_summary: str
    improvement_suggestions: List[str]


# Then create a new controller file named app/controllers/results.py

from fastapi import HTTPException
from typing import Any, List, Dict
import ollama
import asyncio
import re
from app.schemas.schema import ResultsResponse, KalkiScore
from app.db.singleton import ChromaDBSingleton

chroma_client = ChromaDBSingleton()
chroma_collection = chroma_client.get_collection()


async def analyze_user_responses(user_id: str, client: Any) -> ResultsResponse:
    try:
        # Retrieve all user interactions from the database
        query = f"user_id:{user_id}"
        results = chroma_collection.query(
            query_texts=["user responses"],
            where={"user_id": user_id},
            n_results=50  # Adjust as needed
        )

        if not results or not results['documents']:
            raise HTTPException(status_code=404, detail="No user responses found")

        # Compile all user responses
        user_responses = results['documents'][0]
        metadata_list = results['metadatas'][0]

        # Create a comprehensive prompt for the LLM to analyze
        analysis_prompt = [
            {"role": "system", "content": """
            You are a KALKI scoring system expert. Analyze the user's roleplay responses and provide scores based on:

            1. Empathy (0-30): Did the user consider multiple perspectives?
            2. Diplomatic Skill (0-30): Did the responses promote peaceful negotiation?
            3. Historical Accuracy (0-20): Were the responses based on real-world historical lessons?
            4. Ethical Balance (0-20): Did the user avoid bias and consider ethical implications?

            Provide numerical scores and specific feedback for each category. Be fair and objective.
            """
             },
            {"role": "user",
             "content": f"Here are the user's roleplay responses to analyze:\n\n{user_responses}\n\nProvide KALKI scores and feedback."}
        ]

        # Get analysis from LLM
        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=analysis_prompt
        )

        analysis_text = response['message']['content']

        # Parse the AI's analysis to extract scores
        # This is a simplified parsing logic - you might need something more robust
        empathy_score = _extract_score(analysis_text, "Empathy", 30)
        diplomatic_score = _extract_score(analysis_text, "Diplomatic Skill", 30)
        historical_score = _extract_score(analysis_text, "Historical Accuracy", 20)
        ethical_score = _extract_score(analysis_text, "Ethical Balance", 20)

        total_score = empathy_score + diplomatic_score + historical_score + ethical_score

        # Generate feedback by category
        feedback = {
            "empathy": _extract_feedback(analysis_text, "Empathy"),
            "diplomatic_skill": _extract_feedback(analysis_text, "Diplomatic Skill"),
            "historical_accuracy": _extract_feedback(analysis_text, "Historical Accuracy"),
            "ethical_balance": _extract_feedback(analysis_text, "Ethical Balance")
        }

        # Create improvement suggestions
        improvement_prompt = [
            {"role": "system",
             "content": "Based on the KALKI scores, provide 3-5 specific suggestions for improvement."},
            {"role": "user", "content": f"KALKI Analysis: {analysis_text}\n\nProvide concise improvement suggestions."}
        ]

        improvement_response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=improvement_prompt
        )

        improvement_text = improvement_response['message']['content']
        improvement_suggestions = _extract_suggestions(improvement_text)

        # Create a summary
        performance_summary_prompt = [
            {"role": "system", "content": "Create a brief, encouraging summary of the user's KALKI performance."},
            {"role": "user", "content": f"Total Score: {total_score}/100\nAnalysis: {analysis_text}"}
        ]

        summary_response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=performance_summary_prompt
        )

        performance_summary = summary_response['message']['content']

        # Build the result
        kalki_score = KalkiScore(
            empathy=empathy_score,
            diplomatic_skill=diplomatic_score,
            historical_accuracy=historical_score,
            ethical_balance=ethical_score,
            total_score=total_score,
            feedback=feedback
        )

        return ResultsResponse(
            total_score=total_score,
            individual_scores=kalki_score,
            performance_summary=performance_summary,
            improvement_suggestions=improvement_suggestions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing user responses: {str(e)}")


# Helper functions to parse the LLM response
def _extract_score(text: str, category: str, max_score: int) -> int:
    """Extract a numerical score for a category from text"""
    try:
        # Find the category in the text
        category_index = text.find(category)
        if category_index == -1:
            return 0

        # Look for numerical values after the category
        text_segment = text[category_index:category_index + 200]  # Search within reasonable limit
        import re
        scores = re.findall(r'\b(\d{1,2})/?' + str(max_score) + r'\b', text_segment)

        if scores:
            score = int(scores[0])
            return min(score, max_score)  # Ensure we don't exceed max score

        # Alternative pattern for scores like "Score: 25"
        scores = re.findall(r'(?:score|rating|points):\s*(\d{1,2})', text_segment, re.IGNORECASE)
        if scores:
            score = int(scores[0])
            return min(score, max_score)

        return 0  # Default if no score found
    except:
        return 0


def _extract_feedback(text: str, category: str) -> str:
    """Extract feedback for a specific category"""
    try:
        # Find the category in the text
        category_index = text.find(category)
        if category_index == -1:
            return "No specific feedback available."

        # Extract a reasonable chunk of text after the category
        next_category_index = float('inf')
        for cat in ["Empathy", "Diplomatic Skill", "Historical Accuracy", "Ethical Balance"]:
            if cat == category:
                continue

            idx = text.find(cat, category_index + len(category))
            if idx != -1 and idx < next_category_index:
                next_category_index = idx

        if next_category_index == float('inf'):
            feedback_text = text[category_index + len(category):]
        else:
            feedback_text = text[category_index + len(category):next_category_index]

        # Clean up the feedback
        feedback_text = feedback_text.strip()
        feedback_text = feedback_text.split("\n\n")[0]  # Take first paragraph

        # Remove score numbers
        import re
        feedback_text = re.sub(r'\b\d{1,2}/\d{1,2}\b', '', feedback_text)
        feedback_text = re.sub(r'(?:score|rating|points):\s*\d{1,2}', '', feedback_text, flags=re.IGNORECASE)

        feedback_text = feedback_text.strip()
        if not feedback_text:
            return "No specific feedback available."

        return feedback_text
    except:
        return "Error extracting feedback."


def _extract_suggestions(text: str) -> List[str]:
    import re
    """Extract improvement suggestions from text"""
    try:
        # Split by common list markers
        suggestions = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # Check for numbered or bullet points
            if re.match(r'^\d+\.|\*|\-', line):
                # Remove the marker and add to suggestions
                suggestion = re.sub(r'^\d+\.|\*|\-\s*', '', line).strip()
                if suggestion:
                    suggestions.append(suggestion)

        # If no structured list found, try to extract sentences
        if not suggestions:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            suggestions = [s.strip() for s in sentences if len(s.strip()) > 20][:5]  # Limit to 5

        return suggestions[:5]  # Ensure we return at most 5 suggestions
    except:
        return ["Focus on considering multiple perspectives.",
                "Study historical context more deeply.",
                "Practice diplomatic approaches to conflicts."]

