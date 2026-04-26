import os
import json
import secrets
from datetime import datetime
from typing import Optional

import anthropic
import sqlite3
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
APP_PASSWORD      = os.environ.get("APP_PASSWORD", "brain2024")
DB_PATH           = os.environ.get("DB_PATH", "/tmp/brain.db")

app = FastAPI(title="Brain")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

active_sessions = set()

# ── Database ──────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_input   TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            summary     TEXT,
            category    TEXT    NOT NULL DEFAULT 'general',
            subcategory TEXT,
            tags        TEXT    DEFAULT '[]',
            entities    TEXT    DEFAULT '[]',
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup():
    try:
        init_db()
    except Exception as e:
        print(f"DB init error: {e}")

# ── Tool helpers ──────────────────────────────────────────────────────────────
def db_save_note(raw_input: str, content: str, summary: str,
                 category: str, subcategory: Optional[str],
                 tags: list, entities: list) -> dict:
    conn = get_db()
    conn.execute(
        """INSERT INTO notes (raw_input, content, summary, category, subcategory, tags, entities)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (raw_input, content, summary, category, subcategory,
         json.dumps(tags), json.dumps(entities))
    )
    conn.commit()
    conn.close()
    return {"status": "saved", "category": category, "summary": summary}

def db_search_notes(query: str, category: str = "all", limit: int = 10) -> list:
    conn = get_db()
    like = f"%{query}%"
    if category == "all":
        rows = conn.execute(
            """SELECT id, content, summary, category, subcategory, tags, entities, created_at
               FROM notes
               WHERE content LIKE ? OR summary LIKE ? OR tags LIKE ? OR entities LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (like, like, like, like, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, content, summary, category, subcategory, tags, entities, created_at
               FROM notes
               WHERE (content LIKE ? OR summary LIKE ? OR tags LIKE ? OR entities LIKE ?)
               AND category = ?
               ORDER BY created_at DESC LIMIT ?""",
            (like, like, like, like, category, limit)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_person(name: str) -> list:
    conn = get_db()
    rows = conn.execute(
        """SELECT id, content, summary, category, tags, created_at
           FROM notes
           WHERE entities LIKE ? OR content LIKE ?
           ORDER BY created_at DESC""",
        (f'%{name}%', f'%{name}%')
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_recent(limit: int = 10, category: str = "all") -> list:
    conn = get_db()
    if category == "all":
        rows = conn.execute(
            "SELECT id, content, summary, category, tags, created_at FROM notes ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, content, summary, category, tags, created_at FROM notes WHERE category=? ORDER BY created_at DESC LIMIT ?",
            (category, limit)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_history(limit: int = 20) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE content != '' ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def db_clear_messages():
    conn = get_db()
    conn.execute("DELETE FROM messages")
    conn.commit()
    conn.close()

def db_add_message(role: str, content: str):
    conn = get_db()
    conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

# ── AI Tools definition ───────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "save_note",
        "description": "Save a note to the database. Use this whenever the user shares information they want to remember.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content":     {"type": "string", "description": "Cleaned, well-structured version of the note"},
                "summary":     {"type": "string", "description": "One-sentence summary"},
                "category":    {"type": "string", "enum": ["personal", "clinical", "business", "study", "resources"],
                                "description": "personal=personal life, clinical=conditions/meds/treatments, business=clinic building, study=board prep, resources=contacts/URLs/tools"},
                "subcategory": {"type": "string", "description": "Optional subcategory e.g. medications, conditions, contacts, urls, licensing"},
                "tags":        {"type": "array", "items": {"type": "string"}, "description": "Keywords for retrieval"},
                "entities":    {"type": "array", "items": {"type": "string"}, "description": "Named entities: people, medications, conditions, organizations"}
            },
            "required": ["content", "summary", "category", "tags", "entities"]
        }
    },
    {
        "name": "search_notes",
        "description": "Search saved notes by keyword or phrase. Use for any retrieval request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string", "description": "Search keywords"},
                "category": {"type": "string", "enum": ["personal", "clinical", "business", "study", "resources", "all"], "default": "all"},
                "limit":    {"type": "integer", "default": 10}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_person",
        "description": "Get all notes mentioning a specific person.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_recent_notes",
        "description": "Get the most recently saved notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit":    {"type": "integer", "default": 10},
                "category": {"type": "string", "enum": ["personal", "clinical", "business", "study", "resources", "all"], "default": "all"}
            }
        }
    }
]

SYSTEM_PROMPT = """You are Brain, a personal AI assistant and note-taking agent.

Your user is a PMHNP (Psychiatric Mental Health Nurse Practitioner) who just finished her program. She is:
- Studying for the PMHNP board exam (ANCC)
- Building an online psychiatric telehealth clinic
- Tracking clinical knowledge: conditions, medications, treatments
- Managing contacts, resources, and her personal life

CATEGORIES:
- personal    → personal life, feelings, events (handle sensitively)
- clinical    → psychiatric conditions, medications, DSM criteria, pharmacology, assessment tools, treatment protocols
- business    → telehealth clinic, licensing, credentialing, billing, insurance, platforms, legal, marketing
- study       → board exam prep, mnemonics, practice questions, key concepts
- resources   → contacts/networking, URLs, books, courses, tools, recommendations

RULES:
1. When user shares info → always call save_note. Never skip saving.
2. When user asks a question → call search_notes or get_person, then give a clear synthesized answer.
3. After saving, tell the user what category you saved it under and why.
4. For clinical notes, structure them: drug class / mechanism / indications / dosing / side effects.
5. For people, always include their name in entities[].
6. Be warm, concise, and encouraging. She is working hard.
7. If you are unsure of category, pick the best fit and mention it."""

# ── Agent loop ────────────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict, raw: str) -> dict:
    if name == "save_note":
        return db_save_note(raw, args["content"], args["summary"], args["category"],
                            args.get("subcategory"), args.get("tags", []), args.get("entities", []))
    elif name == "search_notes":
        return db_search_notes(args["query"], args.get("category", "all"), args.get("limit", 10))
    elif name == "get_person":
        return db_get_person(args["name"])
    elif name == "get_recent_notes":
        return db_get_recent(args.get("limit", 10), args.get("category", "all"))
    return {"error": "unknown tool"}

def content_to_dict(block) -> dict:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {}

def run_agent(user_message: str) -> str:
    db_add_message("user", user_message)
    messages = db_get_history(20)
    print(f"Sending {len(messages)} messages to Claude")

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    print(f"Got response: stop_reason={response.stop_reason}, blocks={len(response.content)}")

    final_text = ""
    for block in response.content:
        if block.type == "text":
            final_text += block.text
            print(f"Text block: {block.text[:50]}")

    if not final_text:
        final_text = "I'm here but had trouble responding — please try again."

    db_add_message("assistant", final_text)
    return final_text

# ── Auth ──────────────────────────────────────────────────────────────────────
def is_authenticated(request: Request) -> bool:
    token = request.cookies.get("session")
    return token in active_sessions

# ── Routes ────────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return {"status": "ok"}

class LoginRequest(BaseModel):
    password: str

@app.post("/login")
async def login(body: LoginRequest, response: Response):
    if body.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password")
    token = secrets.token_hex(32)
    active_sessions.add(token)
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=60*60*24*30)
    return {"ok": True}

@app.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    active_sessions.discard(token)
    response.delete_cookie("session")
    return {"ok": True}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        reply = run_agent(body.message.strip())
        return {"reply": reply}
    except Exception as e:
        print(f"Chat error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reset")
async def reset():
    db_clear_messages()
    return {"ok": True, "message": "Chat history cleared. Go back to Brain and try again!"}

@app.get("/notes")
async def list_notes(request: Request, category: str = "all", limit: int = 20):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return db_get_recent(limit, category)
