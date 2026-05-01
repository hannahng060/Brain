import os
import json
import secrets
import base64
from datetime import datetime
from typing import Optional

import anthropic
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
APP_PASSWORD      = os.environ.get("APP_PASSWORD", "brain2024")
DATABASE_URL      = os.environ.get("DATABASE_URL", "")

app = FastAPI(title="Brain")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

active_sessions = set()

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id          SERIAL PRIMARY KEY,
            raw_input   TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            summary     TEXT,
            category    TEXT    NOT NULL DEFAULT 'general',
            subcategory TEXT,
            tags        TEXT    DEFAULT '[]',
            entities    TEXT    DEFAULT '[]',
            created_at  TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         SERIAL PRIMARY KEY,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            section    TEXT PRIMARY KEY,
            content    TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    # Seed default empty sections
    sections = ['about','health','nutrition','fitness','career','personal','routines']
    for s in sections:
        cur.execute("INSERT INTO profile (section, content) VALUES (%s, '') ON CONFLICT (section) DO NOTHING", (s,))
    # Migrate old category names to new ones
    cur.execute("UPDATE notes SET category = 'psychiatry' WHERE category IN ('clinical', 'study')")
    conn.commit()
    cur.close()
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
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO notes (raw_input, content, summary, category, subcategory, tags, entities)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (raw_input, content, summary, category, subcategory,
         json.dumps(tags), json.dumps(entities))
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "saved", "category": category, "summary": summary}

def db_search_notes(query: str, category: str = "all", limit: int = 10) -> list:
    conn = get_db()
    cur = conn.cursor()
    # Search for each word independently so "happiness quote" finds notes with either word
    words = [w.strip() for w in query.split() if w.strip()]
    if not words:
        words = [query]
    conditions = " OR ".join(
        ["(content ILIKE %s OR summary ILIKE %s OR tags ILIKE %s OR entities ILIKE %s)"] * len(words)
    )
    params = []
    for w in words:
        like = f"%{w}%"
        params.extend([like, like, like, like])
    if category != "all":
        conditions = f"({conditions}) AND category = %s"
        params.append(category)
    params.append(limit)
    cur.execute(
        f"""SELECT id, content, summary, category, subcategory, tags, entities, created_at
            FROM notes WHERE {conditions}
            ORDER BY created_at DESC LIMIT %s""",
        params
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

def db_get_person(name: str) -> list:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, content, summary, category, tags, created_at
           FROM notes WHERE entities ILIKE %s OR content ILIKE %s
           ORDER BY created_at DESC""",
        (f'%{name}%', f'%{name}%')
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

def db_get_today_logs(category: str, subcategory: str) -> list:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, content, summary, category, subcategory, created_at
           FROM notes
           WHERE category = %s AND subcategory ILIKE %s AND created_at >= NOW() - INTERVAL '30 hours'
           ORDER BY created_at ASC""",
        (category, f"%{subcategory}%")
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

def db_get_recent(limit: int = 10, category: str = "all") -> list:
    conn = get_db()
    cur = conn.cursor()
    if category == "all":
        cur.execute("SELECT id, content, summary, category, subcategory, tags, created_at FROM notes ORDER BY created_at DESC LIMIT %s", (limit,))
    else:
        cur.execute("SELECT id, content, summary, category, subcategory, tags, created_at FROM notes WHERE category=%s ORDER BY created_at DESC LIMIT %s", (category, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

def db_get_history(limit: int = 20) -> list:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE content != '' ORDER BY created_at DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def db_update_note(note_id: int, fields: dict) -> dict:
    conn = get_db()
    cur = conn.cursor()
    allowed = ["subcategory", "category", "summary", "content"]
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return {"status": "nothing to update"}
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    cur.execute(f"UPDATE notes SET {set_clause} WHERE id = %s", list(updates.values()) + [note_id])
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "updated", "note_id": note_id, "fields": list(updates.keys())}

def db_get_profile() -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT section, content FROM profile ORDER BY section")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {r["section"]: r["content"] for r in rows}

def db_update_profile_section(section: str, content: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE profile SET content = %s, updated_at = NOW() WHERE section = %s", (content, section))
    conn.commit()
    cur.close()
    conn.close()

def build_profile_context() -> str:
    profile = db_get_profile()
    filled = {k: v for k, v in profile.items() if v.strip()}
    if not filled:
        return ""
    labels = {
        "about": "About Me",
        "health": "Health & Medical",
        "nutrition": "Nutrition Goals",
        "fitness": "Fitness & Activity",
        "career": "Career & Business",
        "personal": "Personal Values & Goals",
        "routines": "Daily Routines"
    }
    lines = ["USER PROFILE (always use this context when responding):"]
    for k, v in filled.items():
        lines.append(f"[{labels.get(k, k)}]: {v}")
    return "\n".join(lines)

def db_clear_messages():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages")
    conn.commit()
    cur.close()
    conn.close()

def db_add_message(role: str, content: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (role, content) VALUES (%s, %s)", (role, content))
    conn.commit()
    cur.close()
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
                "summary":     {"type": "string", "description": "Short heading, 3-6 words max, like a headline. Examples: 'Strattera for Adult ADHD', 'Hooding Ceremony Day', 'Korean BBQ Lunch', 'Morning Oatmeal Recipe'"},
                "category":    {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "business", "resources", "lifestyle"],
                                "description": "personal=inner world/feelings/journal, psychiatry=psychiatric conditions/meds/assessments/treatments, psychotherapy=therapy modalities (CBT/DBT/ACT etc), icu=ICU nursing/medical knowledge, business=clinic building, resources=contacts/URLs/tools/future ideas, lifestyle=outer world/diet/health/fitness/closet/travel/finance/home"},
                "subcategory": {"type": "string",
                                "enum": ["DSM-5","Medications","Assessments","Treatments","Lab Values","Neuroscience","Ethics & Law","Board Prep",
                                         "CBT","DBT","ACT","Psychodynamic","Motivational Interviewing","Trauma-Focused","Family & Couples","Group Therapy","Theory & Foundations",
                                         "Neuro","Respiratory","Cardiac","GI","Renal","Hematology","Pharmacology","Procedures","Protocols & Guidelines",
                                         "Licensing","Credentialing","Billing & Insurance","Marketing","Platforms","Legal",
                                         "Contacts","URLs & Links","Books","Courses","Tools","Future Ideas",
                                         "Reflections","Goals","Mental Health","Gratitude","Relationships",
                                         "Diet","Health","Fitness","Closet","Travel","Finance","Home","Gardening"],
                                "description": "Pick the subcategory. psychiatry→DSM-5/Medications/Assessments/Treatments/Lab Values/Neuroscience/Ethics & Law/Board Prep. psychotherapy→CBT/DBT/ACT/Psychodynamic/Motivational Interviewing/Trauma-Focused/Family & Couples/Group Therapy/Theory & Foundations. icu→Neuro/Respiratory/Cardiac/GI/Renal/Hematology/Pharmacology/Procedures/Protocols & Guidelines. business→Licensing/Credentialing/Billing & Insurance/Marketing/Platforms/Legal. resources→Contacts/URLs & Links/Books/Courses/Tools/Future Ideas. personal→Reflections/Goals/Mental Health/Gratitude/Journal/Relationships. lifestyle→Diet/Health/Fitness/Closet/Travel/Finance/Home/Gardening"},
                "tags":        {"type": "array", "items": {"type": "string"}, "description": "Keywords for retrieval"},
                "entities":    {"type": "array", "items": {"type": "string"}, "description": "Named entities: people, medications, conditions, organizations"}
            },
            "required": ["content", "summary", "category", "subcategory", "tags", "entities"]
        }
    },
    {
        "name": "search_notes",
        "description": "Search saved notes by keyword or phrase. Use for any retrieval request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string", "description": "Search keywords"},
                "category": {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "business", "resources", "lifestyle", "all"], "default": "all"},
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
                "category": {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "business", "resources", "lifestyle", "all"], "default": "all"}
            }
        }
    },
    {
        "name": "get_today_logs",
        "description": "Get all notes logged today for a specific category and subcategory. Use this to find today's diet log, health log, fitness log, etc. before updating them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category":    {"type": "string", "enum": ["lifestyle","personal","psychiatry","psychotherapy","icu","business","resources"]},
                "subcategory": {"type": "string", "description": "e.g. Diet, Health, Fitness"}
            },
            "required": ["category", "subcategory"]
        }
    },
    {
        "name": "update_note",
        "description": "Update an existing note's subcategory, category, content, or summary. Use this to reorganize or correct existing notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id":     {"type": "integer", "description": "The id of the note to update"},
                "subcategory": {"type": "string",
                                "enum": ["DSM-5","Medications","Assessments","Treatments","Lab Values","Neuroscience","Ethics & Law","Board Prep",
                                         "CBT","DBT","ACT","Psychodynamic","Motivational Interviewing","Trauma-Focused","Family & Couples","Group Therapy","Theory & Foundations",
                                         "Neuro","Respiratory","Cardiac","GI","Renal","Hematology","Pharmacology","Procedures","Protocols & Guidelines",
                                         "Licensing","Credentialing","Billing & Insurance","Marketing","Platforms","Legal",
                                         "Contacts","URLs & Links","Books","Courses","Tools","Future Ideas",
                                         "Reflections","Goals","Mental Health","Gratitude","Relationships",
                                         "Daily Log","Diet","Health","Fitness","Closet","Travel","Finance","Home","Gardening"]},
                "category":    {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "business", "resources", "lifestyle"]},
                "summary":     {"type": "string"},
                "content":     {"type": "string"}
            },
            "required": ["note_id"]
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
- personal      → inner world: feelings, reflections, journal, mental health, relationships, gratitude (handle sensitively)
- lifestyle     → outer world: Daily Log (daily tracking), diet knowledge/recipes, health, fitness, closet, travel, finance, home, gardening
- psychiatry    → psychiatric conditions, medications, DSM criteria, pharmacology, assessment tools, treatment protocols, neuroscience, ethics & law, board prep
- psychotherapy → therapy modalities: CBT, DBT, ACT, Psychodynamic, Motivational Interviewing, Trauma-Focused, Family & Couples, Group Therapy, theory & foundations
- icu           → ICU nursing knowledge: Neuro, Respiratory, Cardiac, GI, Renal, Hematology, Pharmacology, Procedures, Protocols & Guidelines
- business      → telehealth clinic, licensing, credentialing, billing, insurance, platforms, legal, marketing
- resources     → contacts/networking, URLs, books, courses, tools, recommendations, future ideas

RULES:
1. When user shares info → always call save_note. Never skip saving.
2. When a message contains MULTIPLE types of content (e.g. journal story + food log, or event + people + meal) → call save_note MULTIPLE TIMES, once per content type. Never combine different life areas into one note.
3. For diet/food logs → ALWAYS call search_notes first to check if a recipe or meal already exists. If found, use its saved nutrition data. Always include estimated calories, protein, carbs, fat in diet notes.
4. For journal entries → capture people present in entities[], emotions, milestones, events separately from food.
5. When user asks a question → call search_notes or get_person FIRST, then give a clear synthesized answer based on what you find. Always show what you retrieved before asking follow-up questions.
6. After saving, tell the user what you saved and where (category → subcategory). Keep it brief.
7. For clinical notes, structure them: drug class / mechanism / indications / dosing / side effects.
8. For people, always include their name in entities[].
9. Be warm and concise. After retrieving notes, show the summary and end with "Want to add anything?" at most — nothing more. You are a note assistant, not a therapist or journal coach. No bullet-point questions, no prompts about feelings.
10. If you are unsure of category, pick the best fit and mention it.
11. DIET LOG UPDATES (only when user explicitly asks to update a standalone diet log):
    a. Call get_today_logs with category=lifestyle, subcategory=Diet to find today's note.
    b. If found: append the meal, recalculate totals, update_note. If not found: save_note under Diet.
    c. NOTE: If the user is logging their day generally (daily log), do NOT use this rule — use rule 13 instead. Meals belong in the Daily Log, not a separate Diet note.
13. DAILY LOG: When user logs anything about their day (Oura metrics, medications, meals, activities, energy, mood, routine, anything that happened today):
    a. Call get_today_logs with category=lifestyle, subcategory=Daily Log to find today's note.
    b. If found: update the relevant sections with the new information. Call update_note with the complete updated content.
    c. If not found: create a new note with save_note under lifestyle → Daily Log.
       Heading format: "M.DD.YY - DayOfWeek - [Type of Day]" where Type of Day is inferred from context (e.g. Workday, Rest Day, Day Off, Travel Day). Example: "4.30.26 - Thursday - Workday"
    d. Always use this consistent section structure. Fill in what the user reported, put "—" for sections not mentioned. Use HTML bold+underline for every section header exactly as shown:

<strong><u>OURA RING METRICS:</u></strong>
[data or —]

<strong><u>MEDICATIONS & SUPPLEMENTS:</u></strong>
[data or —]

<strong><u>MORNING ROUTINE:</u></strong>
[data or —]

<strong><u>MOOD:</u></strong>
[data or —]

<strong><u>ENERGY:</u></strong>
[data or —]

<strong><u>MEALS:</u></strong>
[all meals with estimated calories — breakfast, lunch, dinner, snacks]

<strong><u>ACTIVITIES:</u></strong>
[data or —]

<strong><u>SPIRITUAL / LEARNING:</u></strong>
[data or —]

<strong><u>EVENING ROUTINE:</u></strong>
[data or —]

<strong><u>ANALYSIS:</u></strong>
[brief summary connecting the day's data]

    e. Meals always go in the MEALS section of Daily Log. NEVER create a separate Diet note for today's meals.
12. QUIZ MODE: When user says "quiz me", "quiz me on [topic]", or "test me":
    a. Call get_recent_notes with category="clinical" and limit=20 to get all clinical notes. If a specific topic is mentioned, also call search_notes with that topic and category="clinical".
    b. If notes are found: immediately ask the FIRST question. Do not explain what you're doing, do not ask what they want to study, just start the quiz.
    c. If truly no notes found: say ONE sentence only — "No clinical notes saved yet — add some and I'll quiz you." Nothing more.
    d. Ask ONE question at a time. Question types: definition, mechanism, indication, side effect, dosing, DSM criteria, or clinical scenario.
    e. Wait for the user's answer. Do NOT give the answer before they respond.
    f. After their answer:
       - If correct: confirm it, then add one key clinical pearl or real-world application from the notes to deepen understanding.
       - If incorrect or incomplete: gently correct them, explain the right answer thoroughly using their notes, highlight the 1-2 most important things to remember, and offer a memory trick if helpful.
       - Either way: end with the teaching, not a question. Let the user ask for the next one when ready.
    g. Keep going if user asks for another question, says "next", or "keep going".
    h. At the end give a short score (e.g. "4/5 — strong on mechanisms, review side effects")."""

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
    elif name == "get_today_logs":
        return db_get_today_logs(args.get("category","lifestyle"), args.get("subcategory","Diet"))
    elif name == "update_note":
        note_id = args.get("note_id")
        fields = {k: args.get(k) for k in ["subcategory", "category", "summary", "content"]}
        return db_update_note(note_id, fields)
    return {"error": "unknown tool"}

def content_to_dict(block) -> dict:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {}

def run_agent_loop(messages: list, raw: str) -> tuple:
    saves_made = []
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages
    )
    while response.stop_reason == "tool_use":
        assistant_content = [content_to_dict(b) for b in response.content]
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input, raw)
            if block.name in ("save_note", "update_note") and "id" in result:
                saves_made.append({"id": result["id"], "tool": block.name})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str)
            })
        messages = messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user",      "content": tool_results}
        ]
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return text, saves_made

def run_agent(user_message: str) -> dict:
    db_add_message("user", user_message)
    messages = db_get_history(20)
    profile_context = build_profile_context()
    if profile_context:
        messages = [
            {"role": "user", "content": profile_context},
            {"role": "assistant", "content": "Got it, I have your profile and will use it as context for all responses."}
        ] + messages
    final_text, saves_made = run_agent_loop(messages, user_message)
    if not final_text:
        final_text = "I'm here but had trouble responding — please try again."
    db_add_message("assistant", final_text)
    return {"reply": final_text, "saves": saves_made}

def run_upload_agent(file_label: str, extracted: str, user_note: str) -> str:
    # Truncate very long documents to avoid token issues
    max_chars = 6000
    truncated = extracted[:max_chars] + ("\n\n[Document truncated — first portion saved]" if len(extracted) > max_chars else "")

    # Ask Claude ONLY for metadata as JSON — bypass tool-use uncertainty entirely
    meta_prompt = (
        f"A file was uploaded: {file_label}\n"
        + (f"User note: {user_note}\n" if user_note else "")
        + f"\nContent:\n{truncated}\n\n"
        "Return ONLY a JSON object with these fields:\n"
        '{"summary": "one sentence", "category": "personal|psychiatry|psychotherapy|icu|business|resources|lifestyle", '
        '"subcategory": "exact subcategory name", "tags": ["tag1","tag2"], "entities": ["name1"]}\n'
        "Categories: personal=inner world, lifestyle=outer world/diet/health/fitness, "
        "psychiatry=psychiatric conditions/meds/assessments/board prep, psychotherapy=therapy modalities, "
        "icu=ICU nursing/medical, business=clinic building, resources=contacts/URLs/tools/future ideas.\n"
        "Subcategories — psychiatry: Conditions/Medications/Assessments/Treatments/Lab Values/Neuroscience/Ethics & Law/Board Prep. "
        "psychotherapy: CBT/DBT/ACT/Psychodynamic/Motivational Interviewing/Trauma-Focused/Family & Couples/Group Therapy/Theory & Foundations. "
        "icu: Neuro/Respiratory/Cardiac/GI/Renal/Hematology/Pharmacology/Procedures/Protocols & Guidelines. "
        "business: Licensing/Credentialing/Billing & Insurance/Marketing/Platforms/Legal. "
        "resources: Contacts/URLs & Links/Books/Courses/Tools/Future Ideas. "
        "personal: Reflections/Goals/Mental Health/Gratitude/Relationships. "
        "lifestyle: Daily Log/Diet/Health/Fitness/Closet/Travel/Finance/Home/Gardening.\n"
        "Return ONLY the JSON, no other text."
    )

    meta_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": meta_prompt}]
    )
    meta_text = meta_response.content[0].text.strip() if meta_response.content else ""

    try:
        # Strip markdown code fences if present
        if meta_text.startswith("```"):
            meta_text = meta_text.split("```")[1]
            if meta_text.startswith("json"):
                meta_text = meta_text[4:]
        meta = json.loads(meta_text)
        summary     = meta.get("summary", file_label)
        category    = meta.get("category", "resources")
        subcategory = meta.get("subcategory")
        tags        = meta.get("tags", [])
        entities    = meta.get("entities", [])
    except Exception:
        summary, category, subcategory, tags, entities = file_label, "resources", None, [], []

    # Save directly — no tool-use loop, guaranteed to save
    db_save_note(f"[Uploaded {file_label}]", truncated, summary, category, subcategory, tags, entities)

    reply = (
        f"Saved! Filed under **{category}**"
        + (f" → {subcategory}" if subcategory else "")
        + f".\n\n**{summary}**"
        + (f"\n\nTags: {', '.join(tags)}" if tags else "")
    )
    db_add_message("assistant", reply)
    return reply

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
    local_date: Optional[str] = None

@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        msg = body.message.strip()
        if body.local_date:
            msg = f"[Today's date: {body.local_date}]\n{msg}"
        result = run_agent(msg)
        return result
    except Exception as e:
        print(f"Chat error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test():
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello in one sentence."}]
        )
        text = response.content[0].text if response.content else "empty response"
        return {"ok": True, "reply": text}
    except Exception as e:
        return {"ok": False, "error": str(e), "type": type(e).__name__}

@app.get("/reset")
async def reset():
    db_clear_messages()
    return {"ok": True, "message": "Chat history cleared. Go back to Brain and try again!"}

@app.get("/stats")
async def get_stats(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT category, COUNT(*) as count FROM notes GROUP BY category ORDER BY count DESC")
    rows = cur.fetchall()
    cur.execute("SELECT COUNT(*) as total FROM notes")
    total = cur.fetchone()["total"]
    cur.close()
    conn.close()
    return {"total": total, "by_category": [dict(r) for r in rows]}

@app.get("/profile")
async def get_profile(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return db_get_profile()

class ProfileUpdate(BaseModel):
    section: str
    content: str

@app.put("/profile")
async def update_profile(body: ProfileUpdate, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    db_update_profile_section(body.section, body.content)
    return {"ok": True}

@app.get("/notes")
async def list_notes(request: Request, category: str = "all", limit: int = 20):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return db_get_recent(limit, category)

class NoteUpdate(BaseModel):
    summary: Optional[str] = None
    content: str
    category: Optional[str] = None
    subcategory: Optional[str] = None

@app.put("/notes/{note_id}")
async def update_note(note_id: int, body: NoteUpdate, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE notes SET content = %s, summary = %s, category = %s, subcategory = %s WHERE id = %s",
        (body.content, body.summary, body.category, body.subcategory, note_id)
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}

class QuizRequest(BaseModel):
    topic: Optional[str] = None

@app.post("/quiz")
async def start_quiz(body: QuizRequest, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Fetch notes directly — psychiatry, psychotherapy, or icu
    conn = get_db()
    cur = conn.cursor()
    topic_lower = (body.topic or "").lower()
    if topic_lower == "icu":
        cats = "('icu')"
    elif topic_lower == "psychotherapy":
        cats = "('psychotherapy')"
    else:
        cats = "('psychiatry','clinical','study')"  # include old category names for backwards compat

    if body.topic and topic_lower not in ("icu", "psychiatry", "psychotherapy"):
        words = [w.strip() for w in body.topic.split() if w.strip()]
        conditions = " OR ".join(["(content ILIKE %s OR summary ILIKE %s OR subcategory ILIKE %s)"] * len(words))
        params = []
        for w in words:
            params.extend([f"%{w}%", f"%{w}%", f"%{w}%"])
        params.append(30)
        cur.execute(
            f"SELECT content, summary, subcategory, category FROM notes WHERE category IN {cats} AND ({conditions}) ORDER BY created_at DESC LIMIT %s",
            params
        )
    else:
        cur.execute(
            f"SELECT content, summary, subcategory, category FROM notes WHERE category IN {cats} ORDER BY created_at DESC LIMIT 30"
        )
    notes = cur.fetchall()
    cur.close()
    conn.close()

    if not notes:
        msg = "No notes found on that topic yet — add some and I'll quiz you."
        db_add_message("assistant", msg)
        return {"reply": msg}

    # Build context from notes
    notes_text = "\n\n---\n\n".join(
        f"[{n['subcategory'] or 'Clinical'}] {n['summary']}\n{n['content']}"
        for n in notes
    )
    topic_label = body.topic or "clinical knowledge"

    quiz_prompt = (
        f"You are quizzing a PMHNP student on her own saved notes. "
        f"Topic requested: {topic_label}.\n\n"
        f"Here are her notes:\n{notes_text}\n\n"
        "Ask ONE quiz question based on these notes. "
        "Question types vary by topic — for medications: mechanism, indication, side effect, dosing; "
        "for conditions: DSM criteria, symptoms, differential; for therapy: technique, indication, theory; "
        "for any topic: definition or application scenario. "
        "Ask the question only — do not give the answer, do not give options, do not explain. "
        "End with 'Take your time!' on a new line."
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": quiz_prompt}]
    )
    question = response.content[0].text.strip() if response.content else "Could not generate a question."

    # Inject into chat history so follow-up answers work naturally
    db_add_message("user", f"Quiz me on {topic_label}")
    db_add_message("assistant", question)
    return {"reply": question}

@app.delete("/notes/{note_id}")
async def delete_note(note_id: int, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM notes WHERE id = %s", (note_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}

def extract_pdf_text(data: bytes) -> str:
    import pdfplumber, io
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)

def extract_excel_text(data: bytes) -> str:
    import openpyxl, io
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    parts = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        parts.append(f"Sheet: {sheet}")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts)

def extract_docx_text(data: bytes) -> str:
    import docx, io
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_pptx_text(data: bytes) -> str:
    from pptx import Presentation
    import io
    prs = Presentation(io.BytesIO(data))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        parts.append(f"Slide {i}:")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        parts.append(t)
    return "\n".join(parts)

def extract_csv_text(data: bytes) -> str:
    import csv, io
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    return "\n".join("\t".join(row) for row in reader if any(row))

def extract_image_text(data: bytes, media_type: str) -> str:
    b64 = base64.standard_b64encode(data).decode("utf-8")
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": "Extract all text and key information from this image. Format it clearly with structure. Include everything visible."}
            ]
        }]
    )
    return response.content[0].text if response.content else ""

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), note: str = Form(default="")):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    data = await file.read()
    filename = file.filename or ""
    content_type = file.content_type or ""

    try:
        if filename.endswith(".pdf") or content_type == "application/pdf":
            extracted = extract_pdf_text(data)
            file_label = f"PDF: {filename}"
        elif filename.endswith((".xlsx", ".xls")) or "spreadsheet" in content_type or "excel" in content_type:
            extracted = extract_excel_text(data)
            file_label = f"Excel: {filename}"
        elif filename.endswith(".docx") or "wordprocessingml" in content_type or "msword" in content_type:
            extracted = extract_docx_text(data)
            file_label = f"Word doc: {filename}"
        elif filename.endswith(".pptx") or "presentationml" in content_type or "powerpoint" in content_type:
            extracted = extract_pptx_text(data)
            file_label = f"PowerPoint: {filename}"
        elif filename.endswith(".csv") or "text/csv" in content_type:
            extracted = extract_csv_text(data)
            file_label = f"CSV: {filename}"
        elif content_type.startswith("image/"):
            extracted = extract_image_text(data, content_type)
            file_label = f"Image: {filename or 'screenshot'}"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, Word, PowerPoint, Excel, CSV, or image file.")

        if not extracted.strip():
            raise HTTPException(status_code=400, detail="Could not extract any content from this file.")

        # Save the file content as its own note
        file_reply = run_upload_agent(file_label, extracted, "")

        # If user also typed a message, process it separately through the full agent
        # so Brain can split it into multiple notes (journal, diet, etc.)
        if note.strip():
            text_reply = run_agent(note.strip())
            reply = text_reply + f"\n\n📎 *Also saved {file_label}.*"
        else:
            reply = file_reply

        return {"reply": reply}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
