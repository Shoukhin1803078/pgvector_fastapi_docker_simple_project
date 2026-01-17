from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, String, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text
from pgvector.sqlalchemy import Vector
from langchain_ollama import OllamaEmbeddings
import os

# ==================================================
# DB CONFIG
# ==================================================
# DATABASE_URL = "postgresql+psycopg2://ai:ai@db:5432/ai"

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

print(f"POSTGRES_DB==={POSTGRES_DB} POSTGRES_PASSWORD==={POSTGRES_PASSWORD} POSTGRES_USER==={POSTGRES_USER} POSTGRES_HOST==={POSTGRES_HOST} POSTGRES_PORT={POSTGRES_PORT}")


DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ==================================================
# OLLAMA EMBEDDINGS
# ==================================================
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
    #  base_url="http://ollama:11434" # if ollama is install in docker pc
    )

# ==================================================
# 1️⃣ NORMAL CRUD MODEL
# ==================================================
class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    description = Column(Text)

# ==================================================
# 2️⃣ VECTOR MODEL
# ==================================================
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Vector(768))

# ==================================================
# SCHEMAS
# ==================================================
class ItemCreate(BaseModel):
    name: str
    description: str

class DocCreate(BaseModel):
    content: str

class SearchQuery(BaseModel):
    query: str

# ==================================================
# FASTAPI APP
# ==================================================
app = FastAPI(title="CRUD + pgvector + Ollama")

# ==================================================
# STARTUP
# ==================================================
@app.on_event("startup")
def startup():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)

# ==================================================
# 🔹 CRUD ENDPOINTS
# ==================================================
@app.post("/items")
def create_item(item: ItemCreate):
    db = SessionLocal()
    obj = Item(name=item.name, description=item.description)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    db.close()
    return obj

@app.get("/items")
def get_items():
    db = SessionLocal()
    items = db.query(Item).all()
    db.close()
    return items

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    db = SessionLocal()
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        return {"error": "Item not found"}
    db.delete(item)
    db.commit()
    db.close()
    return {"message": "Item deleted ✅"}

# ==================================================
# 🔹 VECTOR DB ENDPOINTS
# ==================================================
@app.post("/vector/add")
def add_document(doc: DocCreate):
    db = SessionLocal()
    vector = embeddings.embed_query(doc.content)

    d = Document(content=doc.content, embedding=vector)
    db.add(d)
    db.commit()
    db.refresh(d)
    db.close()

    return {"id": d.id, "message": "Stored with embedding 🔥"}

@app.post("/vector/search")
def search_document(payload: SearchQuery):
    db = SessionLocal()
    q_vector = embeddings.embed_query(payload.query)

    results = (
        db.query(
            Document,
            Document.embedding.cosine_distance(q_vector).label("distance")
        )
        .order_by("distance")
        .limit(3)
        .all()
    )

    db.close()

    return [
        {
            "id": doc.id,
            "content": doc.content,
            "distance": float(distance)
        }
        for doc, distance in results
    ]
