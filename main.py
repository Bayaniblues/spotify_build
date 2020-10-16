from typing import Optional
from graphs import create_plot, search_id

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    bar = create_plot()
    return templates.TemplateResponse("home.html", {
        "plot": bar,
        "request": request,
        "id": id})


@app.get("/search/id", response_class=HTMLResponse)
async def ask_id(request: Request):
    return templates.TemplateResponse('idsearch.html',{"request": request,})

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/v1/search/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    output = search_id(item_id)
    return {"item_id": item_id, "q": output}
