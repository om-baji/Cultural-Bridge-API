from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.story_router import router

app = FastAPI(title="Story Generator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["story"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Story Generator API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
