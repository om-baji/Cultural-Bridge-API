from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.routes.results_router import results_router
from app.routes.story_router import router
from app.routes.rpg_router import rpg_router
from app.routes.conflict_router import conflict_router
from app.routes.debate_router import debate_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Story Generator API", version="1.0")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["story"])
app.include_router(rpg_router, prefix="/api/v1", tags=["Role Playing"])
app.include_router(conflict_router, prefix="/api/v1", tags=["Conflict Resolution"])
app.include_router(debate_router, prefix="/api/v1", tags=["Debate Mode"])
app.include_router(results_router, prefix="/api", tags=["Results"])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
