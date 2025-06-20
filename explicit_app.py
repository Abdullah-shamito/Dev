from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Resolve API"}

@app.get("/healthz")
def health():
    return {"status": "ok"}
