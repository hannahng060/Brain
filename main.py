import os
import re
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

def db_search_notes(query: str, category: str = "all", limit: int = 30) -> list:
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

def db_get_log_by_date(date_str: str) -> list:
    """Find a Daily Log note by date. Normalizes separators so 5-1, 5/1, 5.1 all match."""
    # Normalize separators → try multiple variants so 5-26, 5/26, 5.26 all find "5.01.26"
    normalized = re.sub(r'[-/]', '.', date_str.strip())
    variants = list({date_str.strip(), normalized,
                     re.sub(r'[-/.]', '-', date_str.strip()),
                     re.sub(r'[-/.]', '/', date_str.strip())})
    conn = get_db()
    cur = conn.cursor()
    conditions = " OR ".join(
        ["(summary ILIKE %s OR content ILIKE %s)"] * len(variants)
    )
    params = []
    for v in variants:
        params.extend([f"%{v}%", f"%{v}%"])
    cur.execute(
        f"""SELECT id, content, summary, category, subcategory, created_at
           FROM notes
           WHERE category = 'lifestyle' AND subcategory ILIKE '%daily log%'
           AND ({conditions})
           ORDER BY created_at DESC LIMIT 3""",
        params
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
           WHERE category = %s AND subcategory ILIKE %s AND created_at >= NOW() - INTERVAL '48 hours'
           ORDER BY created_at ASC""",
        (category, f"%{subcategory}%")
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

def db_get_recent(limit: int = 30, category: str = "all") -> list:
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

def db_update_section_by_id(note_id: int, section: str, text: str) -> dict:
    """Replace a section's content in a note by its ID (used for analysis overwrites)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT content FROM notes WHERE id = %s", (note_id,))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return {"status": "error", "message": f"Note {note_id} not found"}
    content = row["content"]
    section_upper = section.upper().rstrip(':') + ':'
    header_pattern = re.compile(
        r'(<strong><u>' + re.escape(section_upper) + r'<\/u><\/strong>)(.*?)(?=<strong><u>|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    match = header_pattern.search(content)
    if not match:
        # Section not found — append it to the end of the note
        new_section = f'\n\n<strong><u>{section_upper}</u></strong>\n{text}\n'
        new_content = content.rstrip() + new_section
    else:
        # Replace existing section content
        new_content = content[:match.start(2)] + '\n' + text + '\n\n' + content[match.end(2):]
    cur.execute("UPDATE notes SET content = %s WHERE id = %s", (new_content, note_id))
    conn.commit()
    cur.close(); conn.close()
    return {"status": "updated", "id": note_id, "section": section}

def db_update_daily_log_section(date_ref: str, section: str, text: str) -> dict:
    """Find a daily log by date reference and append text to a section — all in one step."""
    conn = get_db()
    cur = conn.cursor()

    # Find the note: "today"/"" = last 48h most recent, "yesterday" = last 48h oldest,
    # anything else = search by date string across all daily logs
    date_ref_lower = date_ref.lower().strip()
    if date_ref_lower in ("today", ""):
        cur.execute(
            """SELECT id, content FROM notes
               WHERE category='lifestyle' AND subcategory ILIKE '%daily log%'
               AND created_at >= NOW() - INTERVAL '48 hours'
               ORDER BY created_at DESC LIMIT 1""")
    elif date_ref_lower == "yesterday":
        cur.execute(
            """SELECT id, content FROM notes
               WHERE category='lifestyle' AND subcategory ILIKE '%daily log%'
               AND created_at >= NOW() - INTERVAL '48 hours'
               ORDER BY created_at ASC LIMIT 1""")
    else:
        # Normalize separators and search in summary/content
        normalized = re.sub(r'[-/]', '.', date_ref.strip())
        variants = list({date_ref.strip(), normalized,
                         re.sub(r'[-/.]', '-', date_ref.strip()),
                         re.sub(r'[-/.]', '/', date_ref.strip())})
        conditions = " OR ".join(["(summary ILIKE %s OR content ILIKE %s)"] * len(variants))
        params = []
        for v in variants:
            params.extend([f"%{v}%", f"%{v}%"])
        cur.execute(
            f"""SELECT id, content FROM notes
               WHERE category='lifestyle' AND subcategory ILIKE '%daily log%'
               AND ({conditions})
               ORDER BY created_at DESC LIMIT 1""", params)

    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return {"status": "error", "message": f"No daily log found for '{date_ref}'"}

    note_id = row["id"]
    content  = row["content"]

    # Find the section and append
    section_upper = section.upper().rstrip(':') + ':'
    header_pattern = re.compile(
        r'(<strong><u>' + re.escape(section_upper) + r'<\/u><\/strong>)(.*?)(?=<strong><u>|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    match = header_pattern.search(content)
    if not match:
        cur.close(); conn.close()
        return {"status": "error", "message": f"Section '{section}' not found in note {note_id}"}

    existing = match.group(2).strip()
    if existing and existing not in ('—', '-', ''):
        new_body = existing + '; ' + text
    else:
        new_body = text

    new_content = content[:match.start(2)] + '\n' + new_body + '\n\n' + content[match.end(2):]
    cur.execute("UPDATE notes SET content = %s WHERE id = %s", (new_content, note_id))
    conn.commit()
    cur.close(); conn.close()
    return {"status": "updated", "id": note_id, "section": section, "date_ref": date_ref}

def db_get_history(limit: int = 20) -> list:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE content != '' ORDER BY created_at DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    # Sanitize: ensure messages alternate user/assistant (no consecutive same-role)
    # Drop orphaned messages that would break the Anthropic API
    sanitized = []
    for msg in msgs:
        if sanitized and sanitized[-1]["role"] == msg["role"]:
            # Same role back-to-back — drop the earlier one
            sanitized.pop()
        sanitized.append(msg)
    # Must start with user message
    while sanitized and sanitized[0]["role"] != "user":
        sanitized.pop(0)
    return sanitized

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
                "limit":    {"type": "integer", "default": 30}
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
                "limit":    {"type": "integer", "default": 30},
                "category": {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "business", "resources", "lifestyle", "all"], "default": "all"}
            }
        }
    },
    {
        "name": "get_today_logs",
        "description": "Get notes logged in the last 48 hours for a specific category and subcategory. Use this to find today's or yesterday's daily log before updating.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category":    {"type": "string", "enum": ["lifestyle","personal","psychiatry","psychotherapy","icu","business","resources"]},
                "subcategory": {"type": "string", "description": "e.g. Diet, Health, Fitness, Daily Log"}
            },
            "required": ["category", "subcategory"]
        }
    },
    {
        "name": "get_log_by_date",
        "description": "Find a Daily Log note by a specific date. Use when the user mentions a specific date like '4/28', 'April 28', '4.28.26', or 'last Tuesday'. Returns matching daily log notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_str": {"type": "string", "description": "The date to search for, e.g. '4/28', '4.28', 'April 28', 'May 1'"}
            },
            "required": ["date_str"]
        }
    },
    {
        "name": "update_daily_log",
        "description": "Add new text to a specific section of a Daily Log in ONE step — finds the log and appends the text automatically. Use this for ALL daily log section updates. Never use update_note for daily logs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_ref": {"type": "string", "description": "When is the log? Use 'today', 'yesterday', or a date like '5/1', '5.1.26', 'May 1'"},
                "section":  {"type": "string", "description": "Section to add to: ACTIVITIES, REFLECTIONS, MEALS, MEDICATIONS & SUPPLEMENTS, MOOD, ENERGY, MORNING ROUTINE, EVENING ROUTINE, SPIRITUAL / LEARNING, OURA RING METRICS, ANALYSIS"},
                "text":     {"type": "string", "description": "The new text to add to that section"}
            },
            "required": ["date_ref", "section", "text"]
        }
    },
    {
        "name": "update_note",
        "description": "Update an existing note's subcategory, category, content, or summary. Use this to reorganize or correct existing notes. For Daily Log section updates, prefer append_to_section instead.",
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
    },
    {
        "name": "no_save",
        "description": "Use ONLY as your very first action when the message is purely conversational with zero new information — e.g. 'thanks', 'ok', 'got it'. NEVER call this after already calling save_note or update_note. Do NOT use this if the user mentions any fact, experience, plan, or knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
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

QUICK CAPTURE RULE:
If the message starts with [QUICK CAPTURE — MUST SAVE], the user captured a quick thought on the go. You MUST save or update immediately — never call no_save. Strip the [QUICK CAPTURE] prefix from the content before saving. Apply all routing rules (personal/today/I/me → correct Daily Log section per Rule 4; clinical knowledge → new note; etc).

CRITICAL RULE — READ THIS FIRST:
You MUST call a tool on EVERY single message. No exceptions. Choose exactly one:
- save_note → if the message contains ANY new information (facts, plans, experiences, knowledge — anything)
- search_notes or get_recent_notes → if the message is asking a question about saved notes
- no_save → ONLY if the message is pure conversation with zero information (e.g. "thanks", "ok", "got it")
When in doubt: SAVE IT. It is always better to save than to skip.

When writing the content of a note, always end it with a brief, meaningful sentence that ties the note together — a reflection, clinical pearl, or contextual insight. Write it as a standalone statement, not addressed to the user. No "you" or "your". Example: a Bible verse note ends with "A reminder that strength is found in surrender." A medication note ends with "Key pearl: monitor QTc when combining serotonergic agents."

After calling save_note or update_note, you MUST follow up with a warm, brief text response: confirm what was saved, where it was filed. Do NOT call no_save or any other tool after saving — go straight to your text reply.

RULES:
1. SAVE EVERY MESSAGE THAT CONTAINS INFO. Call save_note immediately. Never skip. Never assume it was already saved.
2. When a message contains MULTIPLE types of content (e.g. journal story + food log, or event + people + meal) → call save_note MULTIPLE TIMES, once per content type. Never combine different life areas into one note. EXCEPTION: if the user explicitly says "add to my daily log" or "update my log for [date]", ALL described details go into that one Daily Log entry — do not split into separate notes.
3. For diet/food logs → ALWAYS call search_notes first to check if a recipe or meal already exists. If found, use its saved nutrition data. Always include estimated calories, protein, carbs, fat in diet notes.
4. Personal messages about the day → ALWAYS update today's Daily Log using update_daily_log. NEVER create a separate Personal note. NEVER use update_note or get_today_logs for this — just call update_daily_log directly with date_ref, section, and text. It handles finding the note automatically. Route to the correct section:
   - REFLECTIONS: feelings, emotions, gratitude, mental/spiritual thoughts (e.g. "I feel blessed", "I'm anxious about...")
   - ACTIVITIES: tasks done, errands, chores, actions taken (e.g. "I cut David's hair", "I went to the store", "I cleaned the house") — APPEND to existing activities, do not replace them
   - MEDICATIONS & SUPPLEMENTS: any medication or supplement taken with time (e.g. "I took Vyvanse 10mg at 9:30am") — APPEND to existing entries
   - MEALS: anything eaten or drank — APPEND to existing meals
   - MOOD: overall emotional tone
   Strong signals it belongs in Daily Log: (a) uses "I" or "me"; (b) mentions people by name in personal context; (c) time words: "today", "tonight", "this morning", "yesterday", "tomorrow".
5. For journal entries → capture people present in entities[], emotions, milestones, events separately from food.
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
13. DAILY LOG: When user logs anything about their day (Oura metrics, medications, meals, activities, energy, mood, routine, anything that happened):
    a. Find the right daily log:
       - If user mentions a specific date (e.g. "4/28", "April 28", "May 1st") → call get_log_by_date with that date string.
       - If user says "today" or no time reference → call get_today_logs (category=lifestyle, subcategory=Daily Log), use the most recent result.
       - If user says "yesterday" → call get_today_logs, use the older result (or match heading date).
       - If no note found → create one with save_note.
    b. Call update_daily_log with date_ref="today" (or "yesterday" or a specific date), the correct section name, and only the new text. This handles finding and updating in one step. NEVER call update_note or get_today_logs for daily log section updates.
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

<strong><u>REFLECTIONS:</u></strong>
[personal thoughts, feelings, gratitude, moments, anything on her mind today]

<strong><u>ANALYSIS:</u></strong>
[brief summary connecting the day's data]

    e. Meals always go in the MEALS section of Daily Log. NEVER create a separate Diet note for today's meals.
    f. Personal thoughts, feelings, reflections, gratitude, or anything about how the day feels → always go in the REFLECTIONS section of the Daily Log. NEVER create a separate Personal note for these. If no Daily Log exists yet for today, create one first, then add the reflection.
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
        return db_search_notes(args["query"], args.get("category", "all"), args.get("limit", 30))
    elif name == "get_person":
        return db_get_person(args["name"])
    elif name == "get_recent_notes":
        return db_get_recent(args.get("limit", 30), args.get("category", "all"))
    elif name == "no_save":
        return {"status": "ok"}
    elif name == "get_today_logs":
        return db_get_today_logs(args.get("category","lifestyle"), args.get("subcategory","Diet"))
    elif name == "get_log_by_date":
        return db_get_log_by_date(args.get("date_str",""))
    elif name == "update_daily_log":
        return db_update_daily_log_section(
            args.get("date_ref", "today"),
            args.get("section", "REFLECTIONS"),
            args.get("text", "")
        )
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
    infer_messages = list(messages)
    # Force tool_choice=any on first call — Brain MUST call a tool (save_note, search, or no_save)
    # This makes saving impossible to skip; Brain can no longer "forget" to call a tool
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        tool_choice={"type": "any"},
        messages=infer_messages
    )
    while response.stop_reason == "tool_use":
        assistant_content = [content_to_dict(b) for b in response.content]
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input, raw)
            if block.name in ("save_note", "update_note", "update_daily_log") and "id" in result:
                saves_made.append({"id": result["id"], "tool": block.name})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str)
            })
        infer_messages = infer_messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user",      "content": tool_results}
        ]
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=infer_messages
        )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return text, saves_made

def run_agent(user_message: str) -> dict:
    db_add_message("user", user_message)
    messages = db_get_history(10)
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
async def list_notes(request: Request, category: str = "all", limit: int = 500):
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

def strip_html(html: str) -> str:
    """Convert HTML to readable plain text for AI analysis."""
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(p|div|li|tr|h[1-6]|strong|u)>', ' ', html, flags=re.IGNORECASE)
    html = re.sub(r'<[^>]+>', '', html)
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') \
               .replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"')
    html = re.sub(r'[ \t]{2,}', ' ', html)
    html = re.sub(r'\n{3,}', '\n\n', html)
    return html.strip()

class LogAppendRequest(BaseModel):
    date_ref: str
    section: str
    text: str

@app.post("/log-append")
async def log_append(body: LogAppendRequest, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not body.date_ref.strip():
        raise HTTPException(status_code=400, detail="date_ref is required")
    if not body.section.strip():
        raise HTTPException(status_code=400, detail="section is required")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    result = db_update_daily_log_section(body.date_ref.strip(), body.section.strip(), body.text.strip())
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

class LogAnalyzeRequest(BaseModel):
    note_id: int

@app.post("/log-analyze")
async def log_analyze(body: LogAnalyzeRequest, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, content, summary, created_at FROM notes WHERE id = %s", (body.note_id,))
    note = cur.fetchone()
    cur.close()
    conn.close()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    profile_text = build_profile_context()
    log_plain = strip_html(note["content"])
    log_date  = note["summary"] or "today"

    analyze_prompt = f"""You are Brain, Hannah's personal AI assistant. She has asked you to analyze her daily log and give honest, grounded feedback.

{profile_text}

Daily log — {log_date}:
{log_plain}

Reflect on her day with these lenses:
1. Overall alignment — based on her goals and values, was this a good day? Be honest, not just encouraging.
2. What she did well — specific things that moved her forward or reflect who she's becoming.
3. What drifted — anything that was off-track, low-value, or inconsistent with her goals. Be direct but kind.
4. One thing to carry forward — a single concrete focus for tomorrow based on today's data.

Write like a thoughtful coach who knows her deeply. Be specific to what she actually logged — never generic. 3-4 short paragraphs, no bullet points. Warm but real."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        messages=[{"role": "user", "content": analyze_prompt}]
    )
    analysis = response.content[0].text.strip() if response.content else "Could not generate analysis."

    # Convert markdown formatting to HTML before saving into the note
    def md_to_html(text: str) -> str:
        # Bold: **text** → <strong>text</strong>
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italic: *text* → <em>text</em>
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Paragraphs separated by blank lines
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        return '<br><br>'.join(p.replace('\n', '<br>') for p in paras)

    analyzed_at = datetime.now().strftime("%-m/%-d/%y %-I:%M %p")
    analysis_html = f"<em style='font-size:12px;color:#888'>{analyzed_at}</em><br><br>{md_to_html(analysis)}"
    db_update_section_by_id(note["id"], "ANALYSIS", analysis_html)

    return {"analysis": analysis, "summary": log_date, "saved": True}

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
        params.append(50)
        cur.execute(
            f"SELECT content, summary, subcategory, category FROM notes WHERE category IN {cats} AND ({conditions}) ORDER BY RANDOM() LIMIT %s",
            params
        )
    else:
        cur.execute(
            f"SELECT content, summary, subcategory, category FROM notes WHERE category IN {cats} ORDER BY RANDOM() LIMIT 50"
        )
    notes = cur.fetchall()
    cur.close()
    conn.close()

    if not notes:
        msg = "No notes found on that topic yet — add some and I'll quiz you."
        db_add_message("assistant", msg)
        return {"reply": msg}

    import random as _random
    notes_list = list(notes)
    _random.shuffle(notes_list)
    # Pick a random starting note to anchor the question — forces variety
    anchor = notes_list[0]
    anchor_topic = anchor['subcategory'] or anchor['summary'] or 'this topic'

    # Build context from notes (shuffled)
    notes_text = "\n\n---\n\n".join(
        f"[{n['subcategory'] or 'Clinical'}] {n['summary']}\n{n['content']}"
        for n in notes_list
    )
    topic_label = body.topic or "clinical knowledge"

    quiz_prompt = (
        f"You are quizzing a PMHNP student on her own saved notes. "
        f"Topic requested: {topic_label}.\n\n"
        f"Here are her notes (in random order):\n{notes_text}\n\n"
        f"IMPORTANT: Focus your question on '{anchor_topic}' — do NOT ask about the same topic as last time. "
        "Vary the question type each time: mechanism, indication, side effect, dosing, DSM criteria, "
        "differential, therapy technique, or clinical scenario. "
        "Ask ONE question only — no answer, no options, no explanation. "
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
