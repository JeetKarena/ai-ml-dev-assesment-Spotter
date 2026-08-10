"""FastAPI application entrypoint."""

from fastapi import FastAPI

from api.lifespan import lifespan

app = FastAPI(title="Spotter API", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Spotter API is running"}
