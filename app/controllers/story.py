from fastapi import HTTPException, Depends
from typing import Any, List, Optional
import ollama
import asyncio
from dotenv import load_dotenv
import datetime

from app.schemas.schema import Response, StoryRequest, StoryResponse, SearchQuery
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


async def add_story(request: StoryRequest, client: Any = Depends(get_model_client)):
    try:
        story_data = {
            "culture": request.culture,
            "theme": request.theme,
            "max_length": request.max_length,
            "language": request.language,
            "tone": request.tone
        }

        story_text = f"{request.culture} {request.theme} {request.tone} {request.language}"
        embeddings = await get_embeddings(story_text, client)

        chroma_collection.add(
            documents=[str(story_data)],
            embeddings=[embeddings],
            metadatas=[story_data],
            ids=[request.culture + "-" + str(datetime.datetime.now().timestamp())]
        )

        return Response(
            success=True,
            message="Added To VectorDB",
            timestamp=str(datetime.datetime.now())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding story: {str(e)}")


async def retrieve_similar_stories(query: str, limit: int = 3, client: Any = None):
    if client is None:
        client = await get_model_client()

    try:
        query_embedding = await get_embeddings(query, client)

        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas"]
        )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving similar stories: {str(e)}")


async def generate_story(request: StoryRequest, client: Any = Depends(get_model_client)):
    query = f"{request.culture} {request.theme} {request.tone}"

    try:
        similar_stories = await retrieve_similar_stories(query, limit=2, client=client)
        retrieved_stories = []

        if similar_stories and len(similar_stories["documents"]) > 0 and len(similar_stories["documents"][0]) > 0:
            for i, doc in enumerate(similar_stories["documents"][0]):
                if "story-" in similar_stories["ids"][0][i]:
                    retrieved_stories.append(doc)

        rag_context = ""
        if retrieved_stories:
            rag_context = "Here are some examples of similar stories for inspiration (DO NOT copy these directly):\n\n"
            for i, story in enumerate(retrieved_stories):
                rag_context += f"Example {i + 1}:\n{story[:500]}...\n\n"

        prompt = (
            f"{rag_context}\n"
            f"Generate an authentic and engaging story from {request.culture} culture. "
            f"{'Theme: ' + request.theme + '.' if request.theme else ''} "
            f"Use a {request.tone} tone. "
            f"Ensure the story is between {max(100, request.max_length // 2)} and {request.max_length} words long. "
            f"Write the story in {request.language}. "
            f"Make this story unique and different from the examples."
        )

        for attempt in range(3):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.chat,
                        model="llama3.2:latest",
                        messages=messages,
                        options={"temperature": 0.7, "top_p": 0.9}
                    ),
                    timeout=60
                )
                story_content = response['message']['content'].strip()

                embeddings = await get_embeddings(story_content, client)

                metadata = {
                    "culture": request.culture,
                    "theme": request.theme,
                    "tone": request.tone,
                    "language": request.language,
                    "model": "llama3.2:latest",
                    "has_rag": len(retrieved_stories) > 0
                }

                chroma_collection.add(
                    documents=[story_content],
                    embeddings=[embeddings],
                    metadatas=[metadata],
                    ids=[request.culture + "-story-" + str(datetime.datetime.now().timestamp())]
                )

                return StoryResponse(
                    story=story_content,
                    character_count=len(story_content),
                    language=request.language,
                    metadata=metadata,
                    used_rag=len(retrieved_stories) > 0,
                    reference_count=len(retrieved_stories)
                )
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise HTTPException(status_code=504, detail="Request timed out")
            except Exception as e:
                if attempt == 2:
                    raise HTTPException(status_code=500, detail=f"Failed to generate story: {str(e)}")

        raise HTTPException(status_code=500, detail="Maximum retries exceeded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in RAG story generation: {str(e)}")


async def search_stories(query: SearchQuery, client: Any = Depends(get_model_client)):
    try:
        search_text = f"{query.text} {query.culture if query.culture else ''} {query.theme if query.theme else ''}"
        query_embedding = await get_embeddings(search_text, client)

        filter_dict = {}
        if query.culture:
            filter_dict["culture"] = query.culture
        if query.theme:
            filter_dict["theme"] = query.theme
        if query.language:
            filter_dict["language"] = query.language

        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=query.limit or 5,
            where=filter_dict if filter_dict else None,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 0
                })

        return {
            "success": True,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching stories: {str(e)}")