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
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
APP_PASSWORD      = os.environ.get("APP_PASSWORD", "brain2024")
DATABASE_URL      = os.environ.get("DATABASE_URL", "")

app = FastAPI(title="Brain")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

active_sessions = set()
_last_error: dict = {}   # stores most recent chat error for debugging

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
    sections = ['about','health','nutrition','fitness','career','personal','routines','weekly_plan','daily_focus']
    for s in sections:
        cur.execute("INSERT INTO profile (section, content) VALUES (%s, '') ON CONFLICT (section) DO NOTHING", (s,))
    cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id          SERIAL PRIMARY KEY,
            topic       TEXT NOT NULL,
            question    TEXT NOT NULL,
            result      TEXT NOT NULL CHECK (result IN ('right','partial','wrong')),
            created_at  TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("ALTER TABLE quiz_results ADD COLUMN IF NOT EXISTS note_id INTEGER")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    # Migrate old category names to new ones
    cur.execute("UPDATE notes SET category = 'psychiatry' WHERE category IN ('clinical', 'study')")
    # Move Board Prep notes from psychiatry to boards
    cur.execute("UPDATE notes SET category = 'boards', subcategory = 'Board Prep' WHERE category = 'psychiatry' AND subcategory = 'Board Prep'")
    # Merge Exam Structure into Board Prep
    cur.execute("UPDATE notes SET subcategory = 'Board Prep' WHERE category = 'boards' AND subcategory = 'Exam Structure'")
    # Remap quiz_results topics to official 6 ANCC board categories
    cur.execute("""
        UPDATE quiz_results SET topic = 'Assessment & Diagnosis'
        WHERE topic IN ('DSM-5','Assessments','Assessment','Diagnosis','Mental Status','Neuroscience','Neuro')
    """)
    cur.execute("""
        UPDATE quiz_results SET topic = 'Psychopharmacology'
        WHERE topic IN ('Medications','Pharmacology','Medication','Lab Values','Neuroscience','Biochemistry','Side Effects')
    """)
    cur.execute("""
        UPDATE quiz_results SET topic = 'Psychotherapy'
        WHERE topic IN ('CBT','DBT','ACT','Psychodynamic','Motivational Interviewing','Trauma-Focused',
                        'Family & Couples','Group Therapy','Theory & Foundations','Therapy','Therapies')
    """)
    cur.execute("""
        UPDATE quiz_results SET topic = 'Medical Management'
        WHERE topic IN ('Medical','Lab','Labs','Procedures','Protocols','ICU','Cardiac','Respiratory','Renal')
    """)
    cur.execute("""
        UPDATE quiz_results SET topic = 'Special Populations'
        WHERE topic IN ('Child','Geriatric','Pediatric','Pregnancy','Forensic','Older Adults','Adolescent')
    """)
    cur.execute("""
        UPDATE quiz_results SET topic = 'Professional & Ethics'
        WHERE topic IN ('Ethics','Law','Legal','Licensing','Professional','Credentialing')
    """)
    conn.commit()
    cur.close()
    conn.close()

@app.on_event("startup")
async def startup():
    try:
        init_db()
    except Exception as e:
        print(f"DB init error: {e}")

SARAH_GUIDE_PATH = "/Users/hannahnguyen/Documents/01_NP-School/Clinical-References/Board-Exam-Prep/2026 Sarah StudyGuide.pdf"

@app.get("/sarah-guide/pdf")
def serve_sarah_guide():
    """Serve Sarah's study guide PDF for the in-app viewer."""
    if not os.path.exists(SARAH_GUIDE_PATH):
        raise HTTPException(status_code=404, detail="Study guide PDF not found.")
    return FileResponse(SARAH_GUIDE_PATH, media_type="application/pdf",
                        headers={"Content-Disposition": "inline; filename=sarah-study-guide.pdf"})

@app.get("/api/admin/fix-board-prep")
def fix_board_prep():
    """One-time migration: move Board Prep notes from psychiatry to boards."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE notes SET category='boards', subcategory='Board Prep' WHERE category='psychiatry' AND subcategory='Board Prep'")
    moved = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return {"moved": moved, "status": "done"}

# ── Tool helpers ──────────────────────────────────────────────────────────────
def db_save_note(raw_input: str, content: str, summary: str,
                 category: str, subcategory: Optional[str],
                 tags: list, entities: list) -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO notes (raw_input, content, summary, category, subcategory, tags, entities)
           VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (raw_input, content, summary, category, subcategory,
         json.dumps(tags), json.dumps(entities))
    )
    note_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "saved", "id": note_id, "category": category, "subcategory": subcategory, "summary": summary}

def _fix_ts(row: dict) -> dict:
    """Append Z to created_at so JavaScript treats it as UTC (not local time)."""
    r = dict(row)
    if r.get("created_at") and hasattr(r["created_at"], "isoformat"):
        r["created_at"] = r["created_at"].isoformat() + "Z"
    return r

_SPIRITUAL_KEYWORDS = {
    "devotion", "devotional", "scripture", "psalm", "proverbs", "gospel",
    "bible", "verse", "prayer", "sermon", "worship", "faith", "god", "jesus",
    "christ", "holy spirit", "lord", "blessed", "salvation", "grace", "amen",
    "matthew", "john", "luke", "mark", "romans", "corinthians", "ephesians",
    "philippians", "genesis", "exodus", "isaiah", "jeremiah", "hebrews",
    "revelation", "church", "ministry", "spiritual", "hymn", "righteousness"
}

def _is_spiritual(text: str) -> bool:
    """Return True if the text looks like spiritual/faith content."""
    lower = text.lower()
    return sum(1 for kw in _SPIRITUAL_KEYWORDS if kw in lower) >= 2

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
    return [_fix_ts(r) for r in rows]

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
    return [_fix_ts(r) for r in rows]

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
           WHERE category = 'lifestyle' AND subcategory ILIKE '%%daily log%%'
           AND ({conditions})
           ORDER BY created_at DESC LIMIT 3""",
        params
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [_fix_ts(r) for r in rows]

def db_get_today_logs(category: str, subcategory: str) -> list:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, content, summary, category, subcategory, created_at
           FROM notes
           WHERE category = %s AND subcategory ILIKE %s AND created_at >= NOW() - INTERVAL '20 hours'
           ORDER BY created_at ASC""",
        (category, f"%{subcategory}%")
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [_fix_ts(r) for r in rows]

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
    return [_fix_ts(r) for r in rows]

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

def _merge_meals(existing_html: str, new_html: str) -> str:
    """Merge meal table rows by meal type — new rows replace same type, new types are appended."""
    MEAL_ORDER = ['breakfast', 'lunch', 'dinner', 'snack', 'dessert', 'desert']
    TABLE_STYLE = 'style="border-collapse:collapse;font-size:14px;margin:4px 0"'

    def extract_rows(html):
        return re.findall(r'<tr>.*?</tr>', html, re.DOTALL | re.IGNORECASE)

    def row_label(row):
        for label in MEAL_ORDER:
            if label in row.lower():
                return label
        return None

    existing_rows = extract_rows(existing_html) if existing_html and existing_html not in ('—', '-', '') else []
    new_rows = extract_rows(new_html)

    if not new_rows:
        return existing_html or new_html  # Nothing parseable in new — keep existing

    # Build ordered dict: label → row (existing first)
    merged = {}
    for row in existing_rows:
        lbl = row_label(row) or ('_x_' + str(len(merged)))
        merged[lbl] = row
    # New rows replace same label or add new
    for row in new_rows:
        lbl = row_label(row) or ('_n_' + str(len(merged)))
        merged[lbl] = row

    # Output in canonical meal order, then any extras
    result = []
    for lbl in MEAL_ORDER:
        if lbl in merged:
            result.append(merged.pop(lbl))
    result.extend(v for k, v in merged.items() if not k.startswith('_x_') or True)

    return f'<table {TABLE_STYLE}>{"".join(result)}</table>'

def db_update_daily_log_section(date_ref: str, section: str, text: str) -> dict:
    """Find a daily log by date reference and append text to a section — all in one step."""
    conn = get_db()
    cur = conn.cursor()

    # Find the note: "today"/"" = last 48h most recent, "yesterday" = last 48h oldest,
    # anything else = search by date string across all daily logs
    date_ref_lower = date_ref.lower().strip()
    if date_ref_lower in ("today", ""):
        # Use 20-hour window so early-morning logs don't find yesterday's note
        cur.execute(
            """SELECT id, content FROM notes
               WHERE category='lifestyle' AND subcategory ILIKE '%daily log%'
               AND created_at >= NOW() - INTERVAL '20 hours'
               ORDER BY created_at DESC LIMIT 1""")
    elif date_ref_lower == "yesterday":
        # Yesterday = between 20 and 48 hours ago
        cur.execute(
            """SELECT id, content FROM notes
               WHERE category='lifestyle' AND subcategory ILIKE '%daily log%'
               AND created_at >= NOW() - INTERVAL '48 hours'
               AND created_at < NOW() - INTERVAL '20 hours'
               ORDER BY created_at DESC LIMIT 1""")
    else:
        # Build search variants — handle both full format ("Wednesday, May 6, 2026")
        # and short numeric formats ("5/6/26", "5.6.26")
        base = date_ref.strip()
        normalized = re.sub(r'[-/]', '.', base)
        variants = list({base, normalized,
                         re.sub(r'[-/.]', '-', base),
                         re.sub(r'[-/.]', '/', base)})
        # Also extract just "Month Day, Year" portion if full weekday format given
        # e.g. "Wednesday, May 6, 2026" → also search "May 6, 2026" and "May 6"
        month_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:,?\s*\d{4})?)', base, re.IGNORECASE)
        if month_match:
            variants.append(month_match.group(1))
            variants.append(month_match.group(1).split(',')[0].strip())  # "May 6"
        variants = list(set(variants))
        conditions = " OR ".join(["(summary ILIKE %s OR content ILIKE %s)"] * len(variants))
        params = []
        for v in variants:
            params.extend([f"%{v}%", f"%{v}%"])
        cur.execute(
            f"""SELECT id, content FROM notes
               WHERE category='lifestyle' AND subcategory ILIKE '%%daily log%%'
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
    # Backward-compat: old logs have "SPIRITUAL / LEARNING" as a single section.
    # If SPIRITUAL or LEARNING not found, fall back to the combined header.
    if not match and section.upper() in ("SPIRITUAL", "LEARNING"):
        fallback_pattern = re.compile(
            r'(<strong><u>SPIRITUAL\s*/\s*LEARNING:<\/u><\/strong>)(.*?)(?=<strong><u>|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        match = fallback_pattern.search(content)
    if not match:
        cur.close(); conn.close()
        return {"status": "error", "message": f"Section '{section}' not found in note {note_id}"}

    existing = match.group(2).strip()
    # These sections always replace (never append) to prevent duplicate tables
    replace_sections = {'OURARINGMETRICS', 'OURA RING METRICS', 'OURA', 'MOOD', 'ENERGY', 'DAILY ROUTINE', 'DAILYROUTINE', 'MORNING ROUTINE', 'MORNINGROUTINE', 'EVENING ROUTINE', 'EVENINGROUTINE', 'MOOD & ENERGY', 'MOODANDENERGY'}
    if section.upper() == 'MEALS':
        # Smart merge: keep existing meal rows, add/replace new ones by meal type
        new_body = _merge_meals(existing, text)
    elif section.upper().replace(' ', '') in replace_sections or section.upper() in replace_sections:
        new_body = text
    elif existing and existing not in ('—', '-', ''):
        new_body = existing + '\n\n' + text
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
    # Only load messages from the last 16 hours — yesterday's lecture dumps
    # can be enormous and blow the token limit on a fresh session
    cur.execute("""SELECT role, content FROM messages
                   WHERE content != '' AND created_at >= NOW() - INTERVAL '16 hours'
                   ORDER BY created_at DESC LIMIT %s""", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # Cap each message at 3000 chars — lecture dumps can be massive and blow token limits
    MAX_MSG = 3000
    msgs = [{"role": r["role"], "content": r["content"][:MAX_MSG] + ("…" if len(r["content"]) > MAX_MSG else "")} for r in reversed(rows)]
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
    import json as _json
    profile = db_get_profile()
    filled = {k: v for k, v in profile.items() if v and str(v).strip()}
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
        if k in ("weekly_plan", "daily_focus"):
            continue  # handled separately below
        lines.append(f"[{labels.get(k, k)}]: {v}")

    # Inject weekly plan with today's events highlighted
    wp_raw = profile.get("weekly_plan", "")
    if wp_raw and wp_raw.strip():
        try:
            wp = _json.loads(wp_raw) if isinstance(wp_raw, str) else wp_raw
            today = datetime.now().strftime("%A")  # e.g. "Monday"
            work_days = wp.get("work_days", [])
            events = wp.get("events", [])
            week_of = wp.get("week_of", "")
            plan_lines = [f"[This Week's Plan{' (week of ' + week_of + ')' if week_of else ''}]:"]
            plan_lines.append(f"  Work days: {', '.join(work_days) if work_days else 'not set'}")
            today_events = [e for e in events if e.get("day", "").capitalize() == today]
            if today_events:
                plan_lines.append(f"  ⚡ TODAY ({today}) events: " + "; ".join(e.get("note", "") for e in today_events))
            other_events = [e for e in events if e.get("day", "").capitalize() != today]
            if other_events:
                plan_lines.append("  Other events: " + "; ".join(f"{e.get('day')}: {e.get('note','')}" for e in other_events))
            lines.append("\n".join(plan_lines))
        except Exception:
            lines.append(f"[Weekly Plan]: {wp_raw}")

    return "\n".join(lines)

def db_save_quiz_result(topic: str, question: str, result: str, note_id: int = None) -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO quiz_results (topic, question, result, note_id) VALUES (%s, %s, %s, %s) RETURNING id",
        (topic, question[:200], result, note_id)
    )
    row = cur.fetchone()
    conn.commit(); cur.close(); conn.close()
    return {"status": "saved", "id": row["id"]}

def db_get_weak_areas(limit: int = 8) -> list:
    """Return topics sorted by lowest score (most wrong answers first)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT topic,
               COUNT(*) as total,
               SUM(CASE WHEN result='right'   THEN 1 ELSE 0 END) as rights,
               SUM(CASE WHEN result='partial' THEN 1 ELSE 0 END) as partials,
               SUM(CASE WHEN result='wrong'   THEN 1 ELSE 0 END) as wrongs,
               ROUND(100.0 * (SUM(CASE WHEN result='right' THEN 1 ELSE 0 END) +
                              SUM(CASE WHEN result='partial' THEN 1 ELSE 0 END) * 0.5) / COUNT(*)) AS score_pct
        FROM quiz_results
        GROUP BY topic
        HAVING COUNT(*) >= 1
        ORDER BY score_pct ASC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [dict(r) for r in rows]

def db_save_daily_focus(priorities: list, study_focus: str, date_str: str) -> dict:
    data = json.dumps({"date": date_str, "priorities": priorities, "study_focus": study_focus})
    db_update_profile_section("daily_focus", data)
    return {"status": "saved", "date": date_str, "priorities": priorities, "study_focus": study_focus}

def db_get_daily_focus() -> dict:
    profile = db_get_profile()
    raw = profile.get("daily_focus", "")
    if not raw:
        return {"date": "", "priorities": [], "study_focus": ""}
    try:
        return json.loads(raw)
    except Exception:
        return {"date": "", "priorities": [], "study_focus": ""}

def db_save_weekly_plan(week_of: str, work_days: list, events: list, notes: str) -> dict:
    plan = json.dumps({
        "week_of": week_of,
        "work_days": [d.strip().capitalize() for d in work_days],
        "events": events,
        "notes": notes,
        "saved_at": datetime.now().isoformat()
    })
    db_update_profile_section("weekly_plan", plan)
    return {"status": "saved", "work_days": work_days, "week_of": week_of}

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
                "category":    {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "np_fellowship", "business", "resources", "lifestyle", "mom", "garden", "boards"],
                                "description": "personal=inner world/feelings/journal (subcategories: Reflections, Goals, Mental Health, Gratitude), mom=everything related to Hannah's mother — benefits, healthcare, calls, travel (subcategories: Quick Reference, IEHP, Medi-Cal, Medicare, Social Security, Primary Doc, Eye Care, Pharmacy, Cash Benefits, Vietnam Travel), garden=plant tracker and gardening notes (subcategories: Orchids, House Plants, Outdoor Flowers, Notes & Learning), boards=ANCC PMHNP-BC board exam prep — practice questions organized by topic (subcategories: Assessment & Diagnosis, Psychopharmacology, Psychotherapy, Medical Management, Special Populations, Professional & Ethics), psychiatry=psychiatric conditions/meds/assessments/treatments, psychotherapy=therapy modalities (CBT/DBT/ACT etc), icu=ICU nursing/medical knowledge, business=clinic building, resources=contacts/URLs/tools/future ideas, lifestyle=outer world/diet/health/fitness/closet/travel/finance/home"},
                "subcategory": {"type": "string",
                                "enum": ["DSM-5","Medications","Assessments","Treatments","Lab Values","Neuroscience","Ethics & Law",
                                         "CBT","DBT","ACT","Psychodynamic","Motivational Interviewing","Trauma-Focused","Family & Couples","Group Therapy","Theory & Foundations",
                                         "Neuro","Respiratory","Cardiac","GI","Renal","Hematology","Pharmacology","Procedures","Protocols & Guidelines",
                                         "Bootcamp","Case Consults","Weekly Calls","Practice Building","Community Notes","Clinical Pearls",
                                         "Licensing","Credentialing","Billing & Insurance","Marketing","Social Media","Platforms","Legal",
                                         "Contacts","URLs & Links","Books","Courses","Tools","Future Ideas",
                                         "Reflections","Goals","Mental Health","Gratitude","Journal",
                                         "Daily Log","Diet","Health","Fitness","Closet","Travel","Finance","Home","Gardening","Social Media",
                                         "Assessment & Diagnosis","Psychopharmacology","Psychotherapy","Medical Management","Special Populations","Professional & Ethics","Board Prep"],
                                "description": "Pick the subcategory. psychiatry→DSM-5/Medications/Assessments/Treatments/Lab Values/Neuroscience/Ethics & Law. psychotherapy→CBT/DBT/ACT/Psychodynamic/Motivational Interviewing/Trauma-Focused/Family & Couples/Group Therapy/Theory & Foundations. icu→Neuro/Respiratory/Cardiac/GI/Renal/Hematology/Pharmacology/Procedures/Protocols & Guidelines. np_fellowship→Bootcamp/Case Consults/Weekly Calls/Practice Building/Community Notes/Clinical Pearls. business→Licensing/Credentialing/Billing & Insurance/Marketing/Social Media/Platforms/Legal. resources→Contacts/URLs & Links/Books/Courses/Tools/Future Ideas. personal→Reflections/Goals/Mental Health/Gratitude. NOTE: Reflections is also where meaningful spiritual phrases/quotes go when Hannah wants them to appear on her inspiration banner — short, memorable lines worth seeing daily. lifestyle→Daily Log/Diet/Health/Fitness/Closet/Travel/Finance/Home/Gardening/Social Media. boards→Assessment & Diagnosis/Psychopharmacology/Psychotherapy/Medical Management/Special Populations/Professional & Ethics/Board Prep (application, eligibility, ATT letter, scheduling, registration, exam breakdown, question counts, test structure, test-taking strategy — anything about the exam journey that is NOT a practice question)"},
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
                "category": {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "np_fellowship", "business", "resources", "lifestyle", "mom", "garden", "boards", "all"], "default": "all"},
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
                "category": {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "business", "resources", "lifestyle", "mom", "garden", "boards", "all"], "default": "all"}
            }
        }
    },
    {
        "name": "get_today_logs",
        "description": "Get notes logged in the last 48 hours for a specific category and subcategory. Use this to find today's or yesterday's daily log before updating.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category":    {"type": "string", "enum": ["lifestyle","personal","psychiatry","psychotherapy","icu","np_fellowship","business","resources","mom","garden","boards"]},
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
                "section":  {"type": "string", "description": "Section to add to: ACTIVITIES, REFLECTIONS, MEDICATIONS & SUPPLEMENTS, MOOD, SPIRITUAL, OURA RING METRICS"},
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
                                "enum": ["DSM-5","Medications","Assessments","Treatments","Lab Values","Neuroscience","Ethics & Law",
                                         "CBT","DBT","ACT","Psychodynamic","Motivational Interviewing","Trauma-Focused","Family & Couples","Group Therapy","Theory & Foundations",
                                         "Neuro","Respiratory","Cardiac","GI","Renal","Hematology","Pharmacology","Procedures","Protocols & Guidelines",
                                         "Licensing","Credentialing","Billing & Insurance","Marketing","Platforms","Legal",
                                         "Contacts","URLs & Links","Books","Courses","Tools","Future Ideas",
                                         "Reflections","Goals","Mental Health","Gratitude",
                                         "Daily Log","Diet","Health","Fitness","Closet","Travel","Finance","Home","Gardening","Social Media",
                                         "Assessment & Diagnosis","Psychopharmacology","Psychotherapy","Medical Management","Special Populations","Professional & Ethics","Board Prep"]},
                "category":    {"type": "string", "enum": ["personal", "psychiatry", "psychotherapy", "icu", "business", "resources", "lifestyle", "boards"]},
                "summary":     {"type": "string"},
                "content":     {"type": "string"}
            },
            "required": ["note_id"]
        }
    },
    {
        "name": "save_quiz_result",
        "description": "Call this immediately after evaluating the user's quiz answer — BEFORE giving feedback. Records whether she got it right, partial, or wrong so Brain can track weak areas and focus future quizzes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic":    {"type": "string", "enum": ["Assessment & Diagnosis","Psychopharmacology","Psychotherapy","Medical Management","Special Populations","Professional & Ethics"], "description": "MUST be one of the 6 official ANCC board categories. Map the question's subject: DSM-5/assessments/diagnosis → 'Assessment & Diagnosis'; medications/pharmacology/neuroscience/lab values → 'Psychopharmacology'; CBT/DBT/ACT/psychodynamic/any therapy modality → 'Psychotherapy'; medical conditions/procedures/ICU → 'Medical Management'; child/geriatric/pregnancy/forensic → 'Special Populations'; ethics/law/licensing/professional → 'Professional & Ethics'"},
                "question": {"type": "string", "description": "First 150 characters of the question asked"},
                "result":   {"type": "string", "enum": ["right","partial","wrong"],
                             "description": "right=fully correct and complete, partial=correct concept but missing key details, wrong=incorrect or didn't know"},
                "note_id":  {"type": "integer", "description": "The note ID of the board question that was answered (from NOTE_ID in the quiz message)"}
            },
            "required": ["topic", "question", "result"]
        }
    },
    {
        "name": "save_daily_focus",
        "description": "Save today's focus: up to 3 daily priorities (any life area) + one study topic. Call this when the user sets their intentions for the day.",
        "input_schema": {
            "type": "object",
            "properties": {
                "priorities":   {"type": "array", "items": {"type": "string"}, "description": "Up to 3 priorities for today — anything: tasks, goals, errands, self-care"},
                "study_focus":  {"type": "string", "description": "One clinical topic to focus studying on today e.g. 'Medications', 'DSM-5', 'CBT'"},
                "date_str":     {"type": "string", "description": "Today's date e.g. '2026-05-07'"}
            },
            "required": ["priorities"]
        }
    },
    {
        "name": "save_weekly_plan",
        "description": "Save the user's weekly schedule when she tells you her work days and plans for the week. This is NOT a note — it's a live setting used for smart reminders. Overwrites previous week's plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "week_of":    {"type": "string", "description": "Week range e.g. 'May 4-10, 2026'"},
                "work_days":  {"type": "array", "items": {"type": "string"}, "description": "Work day names e.g. ['Monday', 'Wednesday', 'Thursday']"},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day":  {"type": "string", "description": "Day name e.g. 'Monday'"},
                            "note": {"type": "string", "description": "What's happening e.g. 'PT appointment', 'Date night with David'"}
                        }
                    },
                    "description": "Special events or appointments by day"
                },
                "notes": {"type": "string", "description": "Any other notes about the week", "default": ""}
            },
            "required": ["work_days"]
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
    },
    {
        "name": "get_weak_areas",
        "description": "Returns Hannah's board exam weak areas sorted by lowest score. Use this during morning briefings to personalize the board study card with her actual weakest topic.",
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
- boards        → ANCC PMHNP-BC board exam prep (subcategories: Assessment & Diagnosis, Psychopharmacology, Psychotherapy, Medical Management, Special Populations, Professional & Ethics = practice questions by ANCC topic; Board Prep = everything about the exam journey — application, eligibility, ATT letter, scheduling, registration, ANCC account, exam breakdown, question counts, test structure, test-taking strategy)

QUICK CAPTURE RULE:
If the message starts with [QUICK CAPTURE — MUST SAVE], the user captured a quick thought on the go. You MUST save or update immediately — never call no_save. Strip the [QUICK CAPTURE] prefix from the content before saving. Apply all routing rules (personal/today/I/me → correct Daily Log section per Rule 4; clinical knowledge → new note; etc).

LECTURE DUMP RULE — READ THIS BEFORE SAVING ANY CLINICAL NOTE:
When Hannah is in a class, review session, or rapid study session, she pastes content in bursts. Before creating a NEW psychiatry/psychotherapy/boards note, ALWAYS call search_notes first with the topic keywords to check if a note on the same subject was saved in the last 2 hours. If a match is found:
- UPDATE the existing note by appending the new content — do NOT create a duplicate
- Signal: "Added to your existing [topic] note."
If no match is found, create a new note as usual.
This applies especially to: prevention levels, neurotransmitter pathways, DSM criteria, medication classes, screening tools, or any topic that clearly continues a previous paste.
Exception: if the new content is a completely different subtopic (e.g. previous note was about dopamine, new content is about eating disorders), create a new note.

CLINICAL TABLE & IMAGE ROUTING — apply when content contains a table, chart, or structured reference data:
- Lab values, reference ranges, metabolic panels → psychiatry → Lab Values
- BMI tables, weight/height charts, dosing tables, treatment algorithms → psychiatry → Treatments
- Brain regions, neural circuits, neuroanatomy → psychiatry → Neuroscience
- Neurotransmitter pathways (dopamine, serotonin, GABA, glutamate, norepinephrine, receptor types) → psychiatry → Neuroscience
- Screening tools, rating scales (PHQ-9, GAD-7, MMSE, MoCA, Columbia, AUDIT, PCL-5) → psychiatry → Assessments
- Prevention levels (primary/secondary/tertiary), public health frameworks → psychiatry → Treatments
- Exam breakdown, ANCC question counts, test structure, time limits, passing scores → boards → Board Prep
- ⛔ NEVER route clinical reference tables to resources. Resources is for contacts, URLs, books, and tools only.

NOTE COUNT RULE:
When Hannah asks "how many notes did you save?", "what did you save today?", "did you save everything?", or any question about today's note count — ALWAYS call get_recent_notes with category="all" and limit=50, then count and list what was saved. Give her an accurate count grouped by category. Never say you don't know. Never say "I've been saving as we go" without actually retrieving and counting.

⛔ PEOPLE CRM OVERRIDE — CHECK THIS BEFORE ANY ROUTING DECISION:
If the message mentions a specific named person (other than Hannah herself) with ANY facts about them — their health, family, job, events, what they said, what they're going through — ALWAYS save to category="people". NEVER route to personal→Relationships, resources→Contacts, or lifestyle. The people category is the ONLY home for information about other people. This takes priority over all other routing rules.

PEOPLE DEDUPLICATION RULE:
Before creating a NEW people note, always call search_notes first to check if a note for that person already exists (search by first name). If a match is found:
- Update the existing note instead of creating a new one
- If unsure whether it's the same person (e.g. same first name, different last name or context), flag it: "I found an existing note for [Name] — is this the same person, or someone different?"
- If Hannah confirms same person: update the existing note and delete or ignore the duplicate
- If two notes exist for the same person (e.g. "Sarah" and "Sarah Johnson"), flag it proactively: "It looks like you may have two cards for Sarah — want me to merge them into one?"
Never silently create a duplicate people card.

CRITICAL RULE — READ THIS FIRST:
You MUST call a tool on EVERY single message. No exceptions. Choose exactly one:
- save_note → if the message contains ANY new information (facts, plans, experiences, knowledge — anything)
- search_notes or get_recent_notes → if the message is asking a question about saved notes
- no_save → ONLY if the message is pure conversation with zero information (e.g. "thanks", "ok", "got it")
When in doubt: SAVE IT. It is always better to save than to skip.

UNDERSTANDING INSTRUCTIONS — READ CAREFULLY:
1. CORRECTIONS: If the user says "no", "that's not right", "you misunderstood", "I meant...", "not that" — STOP, acknowledge the mistake in ONE sentence, then redo it correctly. Never defend the wrong action.
2. AMBIGUOUS REFERENCES: If the user says "update it", "change that", "add to it", "fix it" without specifying which note — call search_notes FIRST to find the most recent relevant note, then act on it. Never say "which note?" without trying to find it first.
3. VAGUE BUT CLEAR INTENT: If the instruction is short or informal ("save this", "remember that", "note that down") — just save it. Don't ask for clarification.
4. MULTI-STEP REQUESTS: If the user asks you to do several things in one message ("save X and also update Y and quiz me on Z") — do ALL of them in sequence. Don't pick just one.
5. IMPLICIT UPDATES: If the user shares new info about something already saved (e.g. new details about a person, a follow-up on a health topic) — search for the existing note first and UPDATE it rather than creating a duplicate.
6. TONE MATCHING: If the user is frustrated, brief, or correcting you — respond with one direct sentence, fix the issue, and don't over-explain. Don't be defensive.

FIRST-PERSON VOICE RULE (applies to ALL personal, lifestyle, health, daily log, and people notes):
When saving notes about Hannah's own life, experiences, decisions, appointments, health, thoughts, or plans involving other people — always write in FIRST PERSON as if Hannah herself wrote it. Use "I", "me", "my" — never "Hannah", "she", "her". Fix grammar and sentence structure, fill in context if needed, but keep her voice natural and authentic.
✅ "Dr. offered a steroid injection. I said no for now — I want to finish all my PT sessions first and reassess after."
✅ "I want to go on a walk with Laura soon."
❌ "Hannah declined the steroid injection and will consider it after completing PT."
❌ "Hannah wants to walk with Laura."

⛔ NEVER ADD YOUR OWN THOUGHTS TO HER NOTES — CRITICAL:
Notes must contain ONLY what Hannah actually said. Never add Brain's own suggestions, next steps, recommendations, or action items into the saved note content. If you think something is worth suggesting, say it in your RESPONSE TEXT — never embed it in the note itself.
❌ WRONG: saving "Need to ask Dr. Sandi what options exist" when Hannah never said that
❌ WRONG: adding "Consider following up with..." or "Next step:" to a personal reflection
✅ RIGHT: save only her words, then suggest in your reply: "Want me to note a follow-up with Dr. Sandi about this?"
This applies to ALL note types: daily log, reflections, spiritual, people, clinical. Her notes = her words only.

BOLDING RULE FOR PLAIN TEXT NOTES:
When saving plain text clinical notes, you MAY use <strong> for: headings, drug names, DSM criteria labels, key terms, and section headers — this helps readability.
⛔ But NEVER bold random words mid-sentence just because you think they sound important. Bold = structure and labels, not your own emphasis judgment on content words.
This applies to personal reflections, health updates, appointments, daily log entries, plans, intentions, and people/CRM notes where Hannah's actions or wishes are mentioned. Clinical/knowledge notes (psychiatry, boards, psychotherapy) are written as reference material and are exempt from this rule.
Also: strip out any meta-instructions directed at Brain from the note content. Phrases like "remind me", "save this", "note this", "don't forget" are instructions — they belong in Brain's response, not in the saved note.

When writing the content of a note, always end it with a brief, meaningful sentence that ties the note together — a reflection, clinical pearl, or contextual insight. Write it as a standalone statement, not addressed to the user. No "you" or "your". Example: a Bible verse note ends with "A reminder that strength is found in surrender." A medication note ends with "Key pearl: monitor QTc when combining serotonergic agents."

After calling save_note or update_note, you MUST follow up with a warm, brief text response: confirm what was saved, where it was filed. Do NOT mention note IDs or numbers in any response — they are internal database values that mean nothing to Hannah. Never say "saved as note #47" or "updated note ID 82". Just say where it was filed and what it contains.

⛔ FORMATTING: ALL responses must use clean HTML only — NO markdown, NO asterisks, NO pound signs (#), NO backticks for formatting. Use <strong> for bold, <br> for line breaks, <ul>/<li> for lists, <em> for italics. This applies to every single response including conversational replies, teaching content, and confirmations.

RULES:
1. SAVE EVERY MESSAGE THAT CONTAINS INFO. Call save_note immediately. Never skip. Never assume it was already saved.
2. When a message contains MULTIPLE types of content (e.g. journal story + event + people update) → call save_note MULTIPLE TIMES, once per content type. Never combine different life areas into one note. EXCEPTION: if the user explicitly says "add to my daily log" or "update my log for [date]", ALL described details go into that one Daily Log entry — do not split into separate notes.
3. FOOD & DIET — Hannah does NOT log meals. She uses Brain as a diet coach, not a food tracker. If she mentions food:
   - Give HONEST feedback on whether it aligns with anti-inflammatory eating (do NOT default to "great choice!" — be truthful)
   - Anti-inflammatory principles: emphasize vegetables, berries, fatty fish, olive oil, nuts, turmeric, ginger, whole grains; avoid seed oils, refined sugar, processed foods, excessive red meat
   - If she asks about a recipe or food idea → give practical guidance based on anti-inflammatory principles
   - Do NOT track calories, macros, or log meals to the daily log
4. Personal messages about the day → ALWAYS update today's Daily Log using update_daily_log. NEVER create a separate Personal note. NEVER use update_note or get_today_logs for this — just call update_daily_log directly with date_ref, section, and text. It handles finding the note automatically. Route to the correct section:
   - SPIRITUAL: ANY faith-related content — devotions, scripture, Bible verses, sermons, spiritual thoughts, prayers, faith reflections, church notes, anything God/faith/worship related — whether typed, spoken, or from an attached image. ⛔ ALWAYS log to SPIRITUAL in today's daily log. NEVER create a separate note for spiritual content.
     → For devotion images: save ONLY the scripture reference + one key insight/takeaway to SPIRITUAL. Do NOT transcribe the full devotion text. Then engage warmly — teach, explain, and discuss the passage with the user.
     → For quick spiritual thoughts typed by the user (e.g. "God is good", "I prayed today"): log briefly to SPIRITUAL, then respond warmly.
     → The goal is DISCUSSION and LEARNING, not archiving. Keep the saved entry short.
     ⛔ PRAYER DUPLICATE RULE: When a prayer is typed, call update_daily_log ONCE to the SPIRITUAL section. Do NOT also call save_note for the same prayer content. One save only — never both.
   - REFLECTIONS: feelings, emotions, gratitude, mental/spiritual thoughts (e.g. "I feel blessed", "I'm anxious about...")
   - ACTIVITIES: tasks done, errands, chores, actions taken (e.g. "I cut David's hair", "I went to the store", "I cleaned the house") — APPEND to existing activities, do not replace them
   - MEDICATIONS & SUPPLEMENTS: any medication or supplement taken with time (e.g. "I took Vyvanse 10mg at 9:30am") — APPEND to existing entries
   - MEALS: anything eaten or drank — REPLACE with complete updated table including ALL meals logged today
   - MOOD: overall emotional tone, energy level, stress, resilience, HR, BP — Energy is always logged here inside the MOOD table, never as a separate section
   Strong signals it belongs in Daily Log: (a) uses "I" or "me"; (b) mentions people by name in personal context; (c) time words: "today", "tonight", "this morning", "yesterday", "tomorrow".
5. For journal entries → capture people present in entities[], emotions, milestones, events separately from food.
5. When user asks a question → call search_notes or get_person FIRST, then give a clear synthesized answer based on what you find. Always show what you retrieved before asking follow-up questions.
6. After saving, tell the user what you saved and where (category → subcategory). Keep it brief.
7. For clinical notes, structure them: drug class / mechanism / indications / dosing / side effects.
8. For people, always include their name in entities[].
9. Be warm and concise. After retrieving notes, show the summary and end with "Want to add anything?" at most — nothing more. You are a note assistant, not a therapist or journal coach. No bullet-point questions, no prompts about feelings.
10. If you are unsure of category, pick the best fit and mention it.
14. NP FELLOWSHIP ROUTING: Save to np_fellowship (not psychiatry) when the note includes ANY of: real patient case context, advice from an experienced NP or mentor, wisdom from Lyndsay Hills' program, takeaways from weekly calls or Skool community, practice-building insights, or anything the user says came from "the fellowship" or "the program." Use these subcategories: Bootcamp (program materials/frameworks), Case Consults (real case discussions), Weekly Calls (call notes), Practice Building (running a private practice), Community Notes (Skool/group chat gems), Clinical Pearls (real-world clinical wisdom with context). If the note is a standalone clinical fact with no fellowship context → save to psychiatry/psychotherapy/icu instead.
13. DAILY LOG: When user logs anything about their day (Oura metrics, medications, activities, energy, mood, routine, anything that happened):
    a. You always know today's exact date from [Today's date: ...] at the top of the message. Use it.
    b. Call update_daily_log using the FULL date string from [Today's date:] as date_ref — for example "Wednesday, May 6, 2026". Never use "today" or short formats like "5/6/26". The note heading and the search both use this exact format so they always match.
    c. If the update returns "not found" → ALWAYS auto-create today's log immediately with save_note under lifestyle → Daily Log, then call update_daily_log again to add the data. Never skip the auto-create step. Never modify a note from a different date. This applies to ALL message types: sleep check-in, mood check-in, supplement log, routine update, activity — if there is no log yet for today, create it first, THEN update it.
    d. Heading format: "Wednesday, May 6, 2026 - Workday" — use the full date from [Today's date:] plus the type of day (Workday or Day Off based on the day of week — Mon–Fri = Workday, Sat–Sun = Day Off).
    e. ⛔ NEVER WIPE EXISTING DATA: For any section that replaces (MOOD, OURA RING METRICS), ALWAYS call get_today_logs first to read what is already there. Hannah logs via panels and chat throughout the day — Brain does not see panel-logged entries in its conversation history. Read the existing section content, then write the complete merged result. Never write a section with only the new entry and lose what was there before.
    d. Always use this consistent section structure. Fill in what the user reported, put "—" for sections not mentioned. Use HTML bold+underline for every section header exactly as shown:

<strong><u>OURA RING METRICS:</u></strong>
[ONLY written from the morning sleep check-in. Format as a two-column table: left column = Readiness, Sleep Score, Hours Slept; right column = Deep Sleep, REM Sleep, Sleep Debt.
IMPORTANT: Sleep time values are entered in decimal shorthand where the digits after the decimal = minutes, NOT fractions. Convert before displaying:
- 1.4 → 1h 4m (not 1h 24m)
- 1.22 → 1h 22m
- 1.45 → 1h 45m
- 6.22 → 6h 22m
- 0.45 → 45m
Always display converted human-readable format (e.g. 1h 22m) in the table, never the raw decimal. Example:
<table style="border-collapse:collapse;font-size:14px;margin:4px 0"><tr><td style="padding:2px 16px 2px 0;color:#888;white-space:nowrap">Readiness</td><td style="padding:2px 40px 2px 0"><strong>72</strong></td><td style="padding:2px 16px 2px 0;color:#888;white-space:nowrap">Deep Sleep</td><td><strong>1h 45m</strong></td></tr><tr><td style="padding:2px 16px 2px 0;color:#888;white-space:nowrap">Sleep Score</td><td style="padding:2px 40px 2px 0"><strong>65</strong></td><td style="padding:2px 16px 2px 0;color:#888;white-space:nowrap">REM Sleep</td><td><strong>1h 22m</strong></td></tr><tr><td style="padding:2px 16px 2px 0;color:#888;white-space:nowrap">Hours Slept</td><td style="padding:2px 40px 2px 0"><strong>6h 22m</strong></td><td style="padding:2px 16px 2px 0;color:#888;white-space:nowrap">Sleep Debt</td><td><strong>6h 30m</strong></td></tr></table>
NEVER write to this section from a mood check-in. Daytime Stress, Resilience, and HR from mood check-ins go in the MOOD section only. Use — if no sleep data logged at all.]

<strong><u>MEDICATIONS & SUPPLEMENTS:</u></strong>
[IMPORTANT: When a "Supplement log:" message arrives → ALWAYS call update_daily_log with section=MEDICATIONS & SUPPLEMENTS. APPEND the item with the time provided. Example format: "✅ 💊 Vitamin — 8:30 AM". Do not replace existing entries, only append. Never ignore a "Supplement log:" message.]


<strong><u>MOOD:</u></strong>
[IMPORTANT: This section is always REPLACED (not appended). Every time you update MOOD, FIRST call get_today_logs (category=lifestyle, subcategory=Daily Log) to read the existing MOOD section from today's note — there may be earlier check-ins logged via the panel that are NOT in this conversation. Then write ONE complete table containing ALL rows: the existing ones from the note PLUS the new one being added now.
Format as a single combined table with header row + one data row per check-in. Include only columns where at least one check-in has data. Example with all columns:
<table style="border-collapse:collapse;font-size:14px;margin:4px 0"><tr><td style="padding:2px 20px 2px 0;color:#888;white-space:nowrap">Time</td><td style="padding:2px 20px 2px 0;color:#888">Mood</td><td style="padding:2px 20px 2px 0;color:#888">Energy</td><td style="padding:2px 20px 2px 0;color:#888">Stress</td><td style="padding:2px 20px 2px 0;color:#888">Resilience</td><td style="padding:2px 20px 2px 0;color:#888">HR</td><td style="color:#888">BP</td></tr><tr><td style="padding:2px 20px 2px 0;white-space:nowrap">9:30 AM</td><td style="padding:2px 20px 2px 0"><strong>😊 Happy</strong></td><td style="padding:2px 20px 2px 0"><strong>4/5</strong></td><td style="padding:2px 20px 2px 0"><strong>Relaxed</strong></td><td style="padding:2px 20px 2px 0"><strong>🟢 Solid</strong></td><td style="padding:2px 20px 2px 0"><strong>66 bpm</strong></td><td><strong>118/76</strong></td></tr><tr><td style="padding:2px 20px 2px 0;white-space:nowrap">3:00 PM</td><td style="padding:2px 20px 2px 0"><strong>😤 Frustrated</strong> — work stress</td><td style="padding:2px 20px 2px 0"><strong>2/5</strong></td><td style="padding:2px 20px 2px 0"><strong>Stressed</strong></td><td style="padding:2px 20px 2px 0"><strong>🟢 Solid</strong></td><td style="padding:2px 20px 2px 0"><strong>72 bpm</strong></td><td>—</td></tr></table>
Daytime Stress, Resilience, HR, and BP always go HERE — never in OURA RING METRICS.
If Vyvanse dose is logged → also update MEDICATIONS & SUPPLEMENTS section (e.g. "Vyvanse (Brand) 30mg at 9:00 AM").
If the mood context mentions mom, family caregiving, Social Security, Medi-Cal, Medicare, IEHP, or any situation involving Hannah's mother → also save a separate note under mom → the relevant subcategory (e.g. IEHP, Social Security, Medi-Cal) capturing what happened, the emotional impact, and any relevant details. Use mom → Quick Reference for phone numbers and account info.]

<strong><u>ACTIVITIES:</u></strong>
[data or —]

<strong><u>SPIRITUAL:</u></strong>
[faith, prayer, scripture, gratitude to God, faith moments — preserve prayer: and my words: verbatim; summarize quoted text from others]

⛔ DO NOT CREATE these sections — they have been removed: MORNING ROUTINE, EVENING ROUTINE, ENERGY, DAILY ROUTINE, LEARNING, MEALS, ANALYSIS. Never write to them, never include them as placeholders with —, never create them. They do not exist.

<strong><u>REFLECTIONS:</u></strong>
[personal feelings, self-observations, emotional processing, relationship moments, gratitude for people/experiences — about her inner world and sense of self, NOT faith/God]


    e. If Hannah mentions food, give honest anti-inflammatory feedback but do NOT log it to the daily log.
    f. Personal thoughts, feelings, reflections, gratitude, or anything about how the day feels → always go in the REFLECTIONS section of the Daily Log. NEVER create a separate Personal note for these. If no Daily Log exists yet for today, create one first, then add the reflection.
    g. SECTION ROUTING — SPIRITUAL vs REFLECTIONS:
       - SPIRITUAL = her relationship with God/faith: prayers, scripture, devotionals, faith moments, blessings she attributes to God, gratitude TO God. Route here if she says "I prayed", "I read the Bible", "God...", "blessed", "scripture", "devotion", or anything faith-related.
       - REFLECTIONS = her relationship with herself: personal feelings, self-awareness, emotional processing, observations about her day, gratitude FOR experiences/people, moments that mattered. Route here if she says "I feel", "I realized", "I'm proud", "I'm grateful for [person/event]", "I noticed", or any inner-world observation not directed at God.
    h. QUOTE HANDLING in SPIRITUAL section:
       - If text starts with "prayer:" or "my words:" → preserve it VERBATIM, word for word, exactly as she wrote it.
       - If text is enclosed in quotation marks ("...") → this is someone else's words; summarize or paraphrase it rather than quoting it in full.
    i. NO REPETITION: Never state the same fact in two different sections. Each piece of information belongs in exactly one section.
    j. FORMATTING RULES FOR ALL SECTIONS:
       - When a section has 2+ distinct items, activities, or events → use an HTML unordered list: <ul style="margin:4px 0;padding-left:18px"><li>item one</li><li>item two</li></ul>
       - NEVER chain multiple items with semicolons, commas, or run-on sentences.
       - Single flowing sentences (e.g. "Feeling grateful today") stay as plain prose — no bullet needed.
       - For data/metrics sections (Oura, medications, supplements) → use the two-column table format shown above.
       - For REFLECTIONS and SPIRITUAL → prose paragraphs are fine, but if listing multiple distinct thoughts, use the bullet list.
       BAD:  "Red light therapy, stretch; reviewed notes at 4am"
       GOOD: <ul style="margin:4px 0;padding-left:18px"><li>Red light therapy</li><li>Stretch</li><li>Reviewed notes at 4 am</li></ul>
       BAD:  "ICU shift until 8pm; napped at lunch; charted notes"
       GOOD: <ul style="margin:4px 0;padding-left:18px"><li>ICU shift until 8pm</li><li>Napped at lunch</li><li>Charted notes</li></ul>
20. FINANCE SUBSCRIPTIONS: When a message starts with "Finance subscription added:" →
    a. Search for an existing note with summary containing "Subscriptions" or "Memberships" under lifestyle → Finance.
    b. If found: update it by adding or updating a row. Keep all existing rows and all existing flags/callouts at the bottom.
    c. If not found: create a new note under lifestyle → Finance titled "Subscriptions & Memberships".
    d. FORMAT — use clean HTML only, NO markdown, NO asterisks:
       - Section heading: <div style="font-weight:700;font-size:14px;margin-bottom:6px">📋 Subscriptions & Memberships</div>
       - Table with styled header row and data rows:
         <table style="border-collapse:collapse;font-size:14px;width:100%">
           <tr style="border-bottom:2px solid #e0e0e0">
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Service</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Cost</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Renewal</td>
             <td style="padding:4px 0;font-weight:700;color:#555">Card</td>
           </tr>
           <tr style="border-bottom:1px solid #f0f0f0">
             <td style="padding:5px 16px 5px 0">Netflix</td>
             <td style="padding:5px 16px 5px 0">$15.99/mo</td>
             <td style="padding:5px 16px 5px 0">Monthly</td>
             <td style="padding:5px 0">MC 8600</td>
           </tr>
         </table>
       - Parse item text to extract service name, cost, renewal date/frequency, and card if provided.
    e. FLAGS — if the charge looks unusual (amount seems high, unexpected, or user signals concern with words like "why", "weird", "not sure"):
       Add a callout BELOW the table:
       <div style="background:#fff8e1;border-left:4px solid #f59e0b;border-radius:6px;padding:8px 12px;margin-top:10px;font-size:13px">⚠️ <strong>Review:</strong> [specific flag note — e.g. "Spotify $24.98 — higher than usual, may have auto-upgraded to Duo/Family. Check plan at spotify.com/account."]</div>
       If multiple flags exist, stack them. Keep all existing flags when updating.
    f. Never create duplicate rows for the same service — update existing row instead.
    g. Do not add to the daily log. This is a standalone reference note.

22. DEBTS & LOANS: When a message starts with "Debt/loan added:" →
    a. Search for an existing note with summary containing "Debt" or "Loan" under lifestyle → Finance.
    b. If found: update it by adding or updating a row. Keep all existing rows and flags.
    c. If not found: create a new note under lifestyle → Finance titled "Debts & Loans". FORMAT — clean HTML only, NO markdown:
       - Heading: <div style="font-weight:700;font-size:14px;margin-bottom:6px">💳 Debts & Loans</div>
       - Table: <table style="border-collapse:collapse;font-size:14px;width:100%">
           <tr style="border-bottom:2px solid #e0e0e0">
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Account</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Type</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Balance</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Min/mo</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">APR</td>
             <td style="padding:4px 0;font-weight:700;color:#555">Notes</td>
           </tr>...
         </table>
    d. For high-interest debts (APR > 20%) or large balances, add a flag callout:
       <div style="background:#fef2f2;border-left:4px solid #ef4444;border-radius:6px;padding:8px 12px;margin-top:10px;font-size:13px">🔴 <strong>Priority:</strong> [e.g. "Chase Sapphire at 24% APR — pay above minimum when possible to reduce interest."]</div>
    e. Never duplicate rows — update existing row instead.
    f. Do not add to the daily log. This is a standalone reference note.

21. BILLS & ACCOUNTS: When a message starts with "Bill/account added:" →
    a. Search for an existing note with summary containing "Bills" or "Accounts" under lifestyle → Finance.
    b. If found: update it by adding or updating a row. Keep all existing rows and flags.
    c. If not found: create a new note under lifestyle → Finance titled "Bills & Accounts". FORMAT — clean HTML only, NO markdown:
       - Heading: <div style="font-weight:700;font-size:14px;margin-bottom:6px">🏦 Bills & Accounts</div>
       - Table: <table style="border-collapse:collapse;font-size:14px;width:100%">
           <tr style="border-bottom:2px solid #e0e0e0">
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Account</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Type</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Policy / Acct #</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Amount</td>
             <td style="padding:4px 16px 4px 0;font-weight:700;color:#555">Website</td>
             <td style="padding:4px 0;font-weight:700;color:#555">Notes</td>
           </tr>...
         </table>
    d. Never duplicate rows — update existing row instead.
    e. Do not add to the daily log. This is a standalone reference note.

23. WEEKLY REVIEW: When a message starts with "WEEKLY REVIEW REQUEST:" →
    a. Call get_today_logs for the past 7 days to gather data.
    b. Respond using clean HTML only — NO markdown, NO asterisks. Format:

       <div style="font-size:15px;line-height:1.6">

       <div style="font-weight:700;font-size:15px;color:var(--primary);margin-bottom:6px">📊 Last Week at a Glance</div>
       <ul style="margin:0 0 14px;padding-left:20px;font-size:14px">
         <li>Sleep: [observation, e.g. "averaged 6h — dipped mid-week"]</li>
         <li>Mood: [pattern]</li>
         <li>Habits: [e.g. "checklist completed 4/7 days"]</li>
         <li>Study: [e.g. "2 quiz sessions, strong on psychopharm"]</li>
         <li>Meals / Activity: [brief]</li>
       </ul>

       <div style="font-weight:700;font-size:15px;color:var(--primary);margin-bottom:6px">💡 Insights</div>
       <ul style="margin:0 0 14px;padding-left:20px;font-size:14px">
         <li>[specific honest observation 1]</li>
         <li>[specific honest observation 2]</li>
       </ul>

       <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px">
         <div style="flex:1;min-width:140px;background:#e8f5e9;border-left:4px solid #4caf50;border-radius:8px;padding:10px 12px;font-size:14px">✅ <strong>Win:</strong> [one specific thing she did well]</div>
         <div style="flex:1;min-width:140px;background:#fff3e0;border-left:4px solid #ff9800;border-radius:8px;padding:10px 12px;font-size:14px">🔧 <strong>Next week:</strong> [one specific thing to improve]</div>
       </div>

       <div style="font-size:13px;color:#888">Ready to plan next week? Hit 📆 Plan Next Week on the dashboard.</div>
       </div>

    c. Be specific — use real numbers and real patterns from the logs, not generic statements.
    d. Tone: warm, direct, honest. Hannah has ADHD — scannable format matters more than length.
    e. Do NOT save to daily log.

24. MORNING BRIEFING:
    Triggers: message starts with "Morning check-in (Oura):" OR first message of the day is a greeting (hi, hello, good morning, good afternoon, good evening, hey, morning).

    a. If sleep data is included (Oura check-in) → FIRST save it to OURA RING METRICS as usual (Rule 4).
    b. ALWAYS call search_notes with category="people" and query="birthday important date surgery event" to scan People notes for anything time-sensitive today or in the next 7 days.
    c. ALWAYS call get_weak_areas to find her weakest board topic to highlight in the study card.
    d. ⛔ ALWAYS respond using the EXACT HTML card format below — NEVER plain text, NEVER markdown, NEVER skip the colored cards. Every section must be a colored div card exactly as shown:

    <div style="font-size:14px;line-height:1.7">
    <div style="font-weight:700;font-size:16px;margin-bottom:10px">[use the correct greeting based on local_time provided: before 12pm → ☀️ Good morning; 12–5pm → 🌤 Good afternoon; after 5pm → 🌙 Good evening], Hannah!</div>

    <div style="background:#f0f4ff;border-left:4px solid #667eea;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:14px">
      📅 <strong>Today — [full day, date]</strong><br>
      [Work day 💼 / Day off 🌿] · [Today's events from weekly plan if any — if none, skip]
    </div>

    [If sleep data provided:]
    <div style="background:#f0fdf4;border-left:4px solid #4caf50;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:14px">
      😴 <strong>Sleep:</strong> Readiness [X] · [X]h slept · [one honest line: "solid recovery" / "below average — take it easy" / "sleep debt building"]
    </div>

    [If daily focus is set for today:]
    <div style="background:#fff8e1;border-left:4px solid #f59e0b;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:14px">
      🎯 <strong>Today's focus:</strong> [Brain / Home / World items if set]
    </div>

    [Always include if board notes exist — use get_weak_areas result:]
    <div style="background:#f3e5f5;border-left:4px solid #9c27b0;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:14px">
      📚 <strong>Board focus today:</strong> [Weakest topic] ([score]%) — [one honest, specific study tip for this topic, not generic]
    </div>

    [If any People notes have birthdays TODAY or within 7 days, OR upcoming events (surgery, travel, big life event) within 7 days — add this card:]
    <div style="background:#fce4ec;border-left:4px solid #e91e63;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:14px">
      💛 <strong>People on your mind</strong><br>
      [Bullet per person: "• 🎂 [Name]'s birthday is [today/tomorrow/in X days] — [month day]"
                          "• 🏥 [Name] has surgery on [date] — [days away]"
                          "• ✈️ [Name] is traveling on [date]"
                          — keep each line short and actionable]
    </div>

    <div style="font-size:13px;color:#888;margin-top:6px">[One short encouraging sentence — specific to what she has going on today]</div>
    </div>

    e. Keep it SHORT — skimmable in 10 seconds. Fill in real data, skip cards with no data (except Today and Board focus — always show those two).
    f. Only show the People card if there is actually something relevant in the next 7 days.
    g. ⛔ Even if no Oura data, no weekly plan, no people events — STILL use the colored card format. Never fall back to plain text.
    h. TONE BY TIME OF DAY — this is important, be intentional:
       - MORNING (before 12pm): Energizing and forward-looking. Show the day ahead. Frame it as "here's what you have today." Encouraging but not over the top.
       - AFTERNOON (12–5pm): Warm and grounding. Call get_today_logs to see what she's already done. Acknowledge what she accomplished. If it's been a full day, validate that. Give a gentle nudge toward the remaining focus or study goal. One honest sentence about momentum — e.g. "You've got a solid half-day in — keep the streak going" or "It's been a heavy morning, give yourself grace this afternoon."
       - EVENING (after 5pm): Calm and wind-down focused. Call get_today_logs to review the full day. Reflect on what happened — wins, hard moments, what she did well. Help her close the day. The board study card in the evening should frame it as "tomorrow's focus" not "today's." End with something that helps her transition to rest, not push harder. Never pressure her to do more at night.

16. DAILY FOCUS: When the user sets intentions for the day — priorities, goals, what they want to accomplish, what to study today — call save_daily_focus. Extract up to 3 priorities (any life area: work tasks, errands, self-care, personal) and one study topic. Examples: "today I want to focus on finishing my notes and studying CBT", "my top 3 today: call insurance, submit credentialing, study medications", "I want to get through my inbox and review DSM-5 today".
15. WEEKLY PLAN: When the user describes their upcoming week (work days, appointments, events) — call save_weekly_plan immediately. This is NOT a note, it is a live setting Brain uses all week for smart reminders. Extract: which days are work days, any appointments or events per day. Overwrite the previous week every time. Examples that trigger this rule: "this week I work Mon, Wed, Thu", "next week my schedule is...", "I'm on modified schedule: Monday and Friday", "I have PT on Monday".
12. QUIZ MODE: When user says "quiz me", "quiz me on [topic]", or "test me":
    a. Call get_recent_notes with category="clinical" and limit=20 to get all clinical notes. If a specific topic is mentioned, also call search_notes with that topic and category="clinical".
    b. If notes are found: immediately ask the FIRST question. Do not explain what you're doing, do not ask what they want to study, just start the quiz.
    c. If truly no notes found: say ONE sentence only — "No clinical notes saved yet — add some and I'll quiz you." Nothing more.
    d. Ask ONE question at a time. Question style rules:
       - Simple, direct clinical stem — one sentence maximum
       - High yield only: mechanism of action, first-line treatment, key DSM criteria, black box warning, contraindication, side effect profile, or a brief clinical scenario
       - Open ended — never multiple choice
       - Expected answer: 1–3 sentences
       - End with "Take your time!" on a new line
    e. Wait for the user's answer. Do NOT give the answer before they respond.
    f. After their answer — do this IN ORDER:
       1. FIRST call save_quiz_result with the topic and result (right/partial/wrong) — do this before giving any feedback
          - right = fully correct and complete
          - partial = right idea but missing key details (e.g. knew the drug class but not the mechanism)
          - wrong = incorrect or said they didn't know
       2. THEN give feedback using colored HTML cards — NO markdown, NO asterisks:

          - right:
            <div style="background:#e8f5e9;border-left:4px solid #4caf50;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:14px">✅ <strong>Correct!</strong> [one sentence confirmation + ONE clinical pearl to deepen understanding]</div>

          - partial:
            <div style="background:#fff8e1;border-left:4px solid #f59e0b;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:14px">🟡 <strong>Partially right.</strong> [affirm what she got right]. Missing: [the key piece she missed]. <br><em>Memory hook: [trick if helpful]</em></div>

          - wrong:
            <div style="background:#fef2f2;border-left:4px solid #ef4444;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:14px">❌ <strong>Not quite.</strong> The answer is: [correct answer]. [explain in 2-3 plain sentences]. <br><em>Remember: [one memory hook]</em></div>

       3. End with the teaching point — not another question. Let her ask for the next one.
    g. Keep going if user says "next", "another", or "keep going".
    h. At the end give a short score using a card:
       <div style="background:#f3f4f6;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:14px">📊 <strong>Score: [X/Y]</strong> — [one line: strong on X, review Y]</div>

16. CALL LOG: When a message starts with "Call log (...):":
    a. Save ONE note under mom → the matching subcategory (Social Security, IEHP, Medi-Cal, Medicare, Primary Doc, Eye Care, Pharmacy, Cash Benefits). If no match, use mom → Quick Reference.
    b. Format the note clearly:
       📞 [Date & Time]
       Called: [who] — [phone if provided]
       What happened: [summary]
       Next step: [next step if provided]
    c. Do NOT create multiple notes. ONE note per call log message.
    d. If a phone number was used → also update mom → Quick Reference with that number if not already there.
18. BOARD QUESTION ENTRY: When a message starts with "Board question entry:" → parse the structured data and save ONE note under boards → [topic subcategory]. Format the note as:

📋 **[TOPIC]**

**Q:** [question text]

[A) choice]
[B) choice]
[C) choice]
[D) choice]

✅ **Correct: [letter]) [correct choice text]**

**Rationale:** [rationale if provided]

*Source: [source if provided]*

Tag the note with the topic name as a tag.

19. BOARD QUIZ: When a message starts with "[BOARD QUIZ - SAVED QUESTION]" →
    a. The message contains the question, choices, LOCKED CORRECT ANSWER, and rationale from Hannah's saved notes.
    b. Present it in this EXACT format — do NOT reveal the answer yet:

    📋 **Board Question** *(topic name)*

    [Question text]

    A) [choice]
    B) [choice]
    C) [choice]
    D) [choice]

    *Type A, B, C, or D to answer.*

    c. When she responds with a letter (A/B/C/D), grade her answer ONLY against the LOCKED CORRECT ANSWER in the message.
       ⛔ NEVER override the locked answer with your own clinical reasoning. The stored answer IS correct for this question — period.
       ⛔ Even if you believe a different answer is more clinically accurate, you MUST honor the locked answer. Hannah submitted this question herself and the locked answer is what she is studying.
       - If her answer matches LOCKED CORRECT ANSWER:
         <div style="background:#e8f5e9;border-left:4px solid #4caf50;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:14px">✅ <strong>Correct!</strong> [brief encouragement]</div>
         Then show the rationale in plain text below.
       - If her answer does NOT match:
         <div style="background:#fef2f2;border-left:4px solid #ef4444;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:14px">❌ <strong>The correct answer is [locked letter]) [locked text].</strong><br>[show the stored rationale]</div>
    d. After revealing the answer, call save_quiz_result with the topic, and right/wrong as appropriate.
    e. Then ask: 'Want another board question? (say next or pick a topic)'

    When a message starts with "Board quiz:" (no saved question) →
    a. Generate your own board-style question on a random or specified topic.
    b. Present in the same A/B/C/D format. Grade using your own reasoning for these Brain-generated questions only.

    ⛔ GLOBAL RULE FOR ALL QUIZZES: If you retrieve a question from Hannah's saved notes (boards category) and the note contains a ✅ Correct: answer, that stored answer IS the correct answer — always. Never override it with your own clinical reasoning, regardless of context. This applies whether the quiz was started from the board tab or from regular chat.

25. PERSONAL CRM — People notes:
    Triggers — NO explicit command needed. Brain detects these naturally:
    - Any mention of a fact about a specific named person: kids, spouse, pets, job, birthday, surgery, travel, school, where they live, what they said, what they're going through
    - "I talked to [name] today", "I ran into [name]", "[name] called me", "[name] told me..."
    - "Update [name]", "Remember about [name]", "Add to [name]'s note" (explicit commands)
    - "What do I know about [name]?" or "Pull up [name]" or "Tell me about [name]"

    a. FOR UPDATES — creating or updating a person's note:
       1. Identify ALL named people in the message — both the person Hannah is talking about AND any other people mentioned by name who have their own facts stated about them.
          Example: "I talked to Christina and she said Charlie goes to Cornell" → two people: Christina and Charlie.
          - Christina's card: log in Notes & Conversations: "• [Month Year] — said Charlie goes to Cornell"
          - Charlie's card: log under Key Facts: "School: Cornell University, Ithaca NY"
          Each person gets their own card updated with the facts that are ABOUT THEM specifically.
       2. For each person identified: call search_notes with their name and category="people" to check if they already have a note.
       3. If a note exists → MERGE the new info into the existing note content and re-save using save_note with the same summary (person's name). Never lose existing info — only add to it.
       4. If no note exists → CREATE a new note using the full template below. Fill in only what is known; leave other fields as "—" so they can be filled later.
       5. ⛔ ALWAYS use category="people" — NEVER use category="resources" or subcategory="Contacts". People cards belong ONLY in the people category.
          subcategory = Family / Friends / Work / Community (infer from context; default to Friends)
       6. ⛔ The note summary (title) MUST be the person's name — first name only if that's all that's known, or full name. NEVER use any other text as the summary. Not a description, not a topic, not a sentence — just the name.
       7. Use the full template — even if only one field is known:

       <div style="font-size:14px;line-height:1.8">
       <div style="font-weight:700;font-size:17px;margin-bottom:2px">[Full Name]</div>
       <div style="font-size:12px;color:#888;margin-bottom:14px">[Subcategory] · Last updated [Month Year]</div>

       <div style="background:#fff3e0;border-left:4px solid #d45d00;border-radius:8px;padding:10px 14px;margin-bottom:10px">
       <strong>👤 Key Facts</strong><br>
       How we know each other: [relationship / context]<br>
       Job / Role: [or —]<br>
       Location: [city/state or —]<br>
       Birthday: [Month Day, Year or just Month Day if year unknown — or —]<br>
       Spouse / Partner: [name or —]
       </div>

       <div style="background:#f3f4f6;border-radius:8px;padding:10px 14px;margin-bottom:10px">
       <strong>👨‍👩‍👧 Family & Pets</strong><br>
       Kids: [names, ages, schools if known — or —]<br>
       Pets: [names, type — or —]
       </div>

       <div style="background:#fce4ec;border-left:4px solid #e91e63;border-radius:8px;padding:10px 14px;margin-bottom:10px">
       <strong>📅 Important Dates & Events</strong><br>
       [Bullet list — include ALL upcoming or notable dates: birthdays (repeat annually), surgeries, travel, big milestones, events they mentioned]<br>
       Format each as: "• [Month Day, Year] — [what it is, e.g. 'Surgery at Loma Linda' / 'Flying to Ithaca' / 'Birthday 🎂']"<br>
       [If nothing known yet: —]
       </div>

       <div style="background:#f0f4ff;border-left:4px solid #667eea;border-radius:8px;padding:10px 14px;margin-bottom:10px">
       <strong>📝 Notes & Conversations</strong><br>
       [Bullet list — things they've shared, memorable moments, things Hannah wants to remember. Most recent first.]<br>
       Format: "• [Month Year] — [detail]"<br>
       [If nothing yet: —]
       </div>
       </div>

       6. Use the person's name as the note summary/title.
       7. Confirm briefly: "Got it — saved to [name]'s card. [one line on what was added]"

    b. FOR LOOKUPS — "What do I know about [name]?":
       1. Call search_notes with the person's name and category="people".
       2. If found → summarize in 3-5 conversational bullet points. Highlight anything upcoming or time-sensitive first.
       3. If not found → "I don't have a note for [name] yet. Want me to create one?"

    e. IMPORTANT DATES awareness:
       - Birthdays should always be stored in the Important Dates section with exact month/day so Brain can surface them annually.
       - Surgeries, travel, and major events should include the full date (Month Day, Year).
       - Brain uses these during morning briefings (Rule 24) to remind Hannah when something is coming up within 7 days.

    g. TONE: Quick and natural on confirms. No over-explaining."""

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
    elif name == "get_weak_areas":
        return db_get_weak_areas(limit=6)
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
    elif name == "save_quiz_result":
        return db_save_quiz_result(
            args.get("topic", "General"),
            args.get("question", ""),
            args.get("result", "wrong"),
            note_id=args.get("note_id")
        )
    elif name == "save_daily_focus":
        return db_save_daily_focus(
            args.get("priorities", []),
            args.get("study_focus", ""),
            args.get("date_str", datetime.now().strftime("%Y-%m-%d"))
        )
    elif name == "save_weekly_plan":
        return db_save_weekly_plan(
            args.get("week_of", ""),
            args.get("work_days", []),
            args.get("events", []),
            args.get("notes", "")
        )
    return {"error": "unknown tool"}

def content_to_dict(block) -> dict:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {}  # unknown block type — filtered out below

def run_agent_loop(messages: list, raw: str) -> tuple:
    saves_made = []
    infer_messages = list(messages)
    if not infer_messages:
        return "I'm here but had trouble responding — please try again.", saves_made
    # Force tool_choice=any on first call — Brain MUST call a tool (save_note, search, or no_save)
    # This makes saving impossible to skip; Brain can no longer "forget" to call a tool
    # max_tokens=4096 prevents truncation mid-tool-call (daily logs can be long)
    # Try Sonnet first (smarter), fall back to Haiku if unavailable
    for _model in ["claude-sonnet-4-5-20251001", "claude-3-5-sonnet-20241022", "claude-haiku-4-5-20251001"]:
        try:
            response = client.messages.create(
                model=_model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                tool_choice={"type": "any"},
                messages=infer_messages
            )
            break
        except Exception as _e:
            if "haiku" in _model:
                raise
            continue
    while response.stop_reason == "tool_use":
        # Filter out empty/unknown content blocks — prevents "invalid content" API errors
        assistant_content = [d for d in (content_to_dict(b) for b in response.content) if d]
        if not assistant_content:
            break  # no valid content to continue with
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input, raw)
            if block.name in ("save_note", "update_note", "update_daily_log") and "id" in result:
                saves_made.append({
                    "id": result["id"],
                    "tool": block.name,
                    "category": result.get("category", ""),
                    "subcategory": result.get("subcategory", ""),
                    "summary": result.get("summary", "")
                })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str)
            })
        if not tool_results:
            break  # no tool results to send — exit gracefully
        infer_messages = infer_messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user",      "content": tool_results}
        ]
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=infer_messages
        )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return text, saves_made

def run_agent(user_message: str) -> dict:
    db_add_message("user", user_message)
    profile_context = build_profile_context()

    def _build_messages(limit: int) -> list:
        msgs = db_get_history(limit)
        if profile_context:
            msgs = [
                {"role": "user", "content": profile_context},
                {"role": "assistant", "content": "Got it, I have your profile and will use it as context for all responses."}
            ] + msgs
        return msgs

    # Try with progressively fewer messages if we hit the token limit
    final_text, saves_made = None, []
    for history_limit in [10, 6, 4, 2]:
        try:
            messages = _build_messages(history_limit)
            final_text, saves_made = run_agent_loop(messages, user_message)
            break
        except Exception as e:
            err = str(e)
            if "prompt is too long" in err or "too many tokens" in err.lower() or "invalid_request_error" in err:
                if history_limit == 2:
                    final_text = "I'm having trouble with a very long conversation. Please try refreshing and starting fresh!"
                continue
            raise

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
        "IMPORTANT: If this content is ANY spiritual content — devotion, Bible passage, scripture, sermon, prayer, spiritual thought, worship — → set category=lifestyle, subcategory=Daily Log, and summary should be the passage/sermon title or first line.\n"
        "Return ONLY a JSON object with these fields:\n"
        '{"summary": "one sentence", "category": "personal|psychiatry|psychotherapy|icu|np_fellowship|business|resources|lifestyle|people|mom|garden|boards", '
        '"subcategory": "exact subcategory name", "tags": ["tag1","tag2"], "entities": ["name1"]}\n'
        "Categories: personal=inner world, lifestyle=outer world/diet/health/fitness, "
        "psychiatry=psychiatric conditions/meds/assessments/board prep, psychotherapy=therapy modalities, "
        "icu=ICU nursing/medical, business=clinic building, resources=URLs/tools/future ideas, people=people CRM cards.\n"
        "Subcategories — lifestyle: Daily Log/Diet/Health/Fitness/Closet/Travel/Finance/Home/Gardening. "
        "personal: Reflections/Goals/Mental Health/Gratitude. "
        "psychiatry: Conditions/Medications/Assessments/Treatments/Lab Values/Neuroscience/Ethics & Law. "
        "boards: Assessment & Diagnosis/Psychopharmacology/Psychotherapy/Medical Management/Special Populations/Professional & Ethics/Board Prep. "
        "psychotherapy: CBT/DBT/ACT/Psychodynamic/Motivational Interviewing/Trauma-Focused/Family & Couples/Group Therapy/Theory & Foundations. "
        "icu: Neuro/Respiratory/Cardiac/GI/Renal/Hematology/Pharmacology/Procedures/Protocols & Guidelines. "
        "business: Licensing/Credentialing/Billing & Insurance/Marketing/Platforms/Legal. "
        "resources: URLs & Links/Books/Courses/Tools/Future Ideas.\n"
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
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app")

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    with open("static/index.html") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

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
    local_time: Optional[str] = None

@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        msg = body.message.strip()
        if body.local_date:
            time_str = f" · {body.local_time}" if body.local_time else ""
            msg = f"[Today's date: {body.local_date}{time_str}]\n{msg}"
        result = run_agent(msg)
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Chat error: {type(e).__name__}: {e}\n{tb}")
        _last_error["type"] = type(e).__name__
        _last_error["msg"] = str(e)
        _last_error["tb"] = tb
        _last_error["time"] = datetime.now().isoformat()
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

@app.get("/chat-history")
async def get_chat_history(request: Request, limit: int = 40):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    msgs = db_get_history(limit)
    # db_get_history returns newest-first; reverse for display
    return list(reversed(msgs))

@app.get("/reset")
async def reset():
    db_clear_messages()
    return {"ok": True, "message": "Chat history cleared. Go back to Brain and try again!"}

@app.get("/last-error")
async def last_error():
    """Returns the last chat error traceback for debugging."""
    if not _last_error:
        return {"ok": True, "message": "No errors recorded since last deploy."}
    return _last_error

@app.get("/board-quiz/random")
async def board_quiz_random(request: Request, topic: str = None, exclude: str = None):
    """Return a spaced-repetition-weighted board question.
    exclude: comma-separated note IDs to skip (avoids duplicates in a drill set).
    """
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_db()
    cur = conn.cursor()
    # Parse exclude list
    excluded_ids = []
    if exclude:
        try:
            excluded_ids = [int(x) for x in exclude.split(",") if x.strip().isdigit()]
        except Exception:
            pass

    # Spaced repetition: weight questions by last result + recency
    # Never seen=100, wrong=80, partial=60, right>7d=40, right>3d=20, right recently=5
    topic_filter = "AND n.subcategory ILIKE %s" if topic else ""
    exclude_filter = f"AND n.id != ALL(%s)" if excluded_ids else ""
    params = []
    if topic:
        params.append(f"%{topic}%")
    if excluded_ids:
        params.append(excluded_ids)
    cur.execute(f"""
        WITH last_results AS (
            SELECT DISTINCT ON (note_id) note_id, result, created_at
            FROM quiz_results
            WHERE note_id IS NOT NULL
            ORDER BY note_id, created_at DESC
        ),
        scored AS (
            SELECT n.id, n.content, n.subcategory,
                CASE
                    WHEN lr.note_id IS NULL THEN 100
                    WHEN lr.result = 'wrong'   THEN 80
                    WHEN lr.result = 'partial' THEN 60
                    WHEN lr.result = 'right' AND lr.created_at < NOW() - INTERVAL '7 days' THEN 40
                    WHEN lr.result = 'right' AND lr.created_at < NOW() - INTERVAL '3 days' THEN 20
                    ELSE 5
                END AS weight
            FROM notes n
            LEFT JOIN last_results lr ON lr.note_id = n.id
            WHERE n.category = 'boards' {topic_filter} {exclude_filter}
        )
        SELECT id, content, subcategory FROM scored
        ORDER BY RANDOM() * weight DESC
        LIMIT 1
    """, params)
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return {"found": False}

    content = row["content"]
    subcategory = row["subcategory"] or ""

    # Parse question
    q_match = re.search(r'\*\*Q:\*\*\s*(.+?)(?=\n\s*[A-D]\))', content, re.DOTALL)
    question = q_match.group(1).strip() if q_match else ""

    # Parse choices
    choices = {}
    for letter in ['A','B','C','D']:
        m = re.search(rf'\n\s*{letter}\)\s*(.+?)(?=\n\s*[B-D]\)|\n\s*✅|\Z)', content, re.DOTALL)
        if m:
            choices[letter] = m.group(1).strip()

    # Parse correct answer  e.g. "✅ **Correct: B) Some text**"
    correct_match = re.search(r'✅[^\n]*Correct:\s*([A-D])\)\s*(.+?)(?:\*\*|$)', content, re.IGNORECASE)
    correct_letter = correct_match.group(1).upper() if correct_match else ""
    correct_text   = correct_match.group(2).strip() if correct_match else choices.get(correct_letter, "")

    # Parse rationale
    rat_match = re.search(r'\*\*Rationale:\*\*\s*(.+?)(?=\n\*Source|\*Source|\Z)', content, re.DOTALL)
    rationale = rat_match.group(1).strip() if rat_match else ""

    # Parse source
    src_match = re.search(r'\*Source:\s*(.+?)\*', content)
    source = src_match.group(1).strip() if src_match else ""

    return {
        "found": True,
        "note_id": row["id"],
        "topic": subcategory,
        "question": question,
        "choices": choices,
        "correct_letter": correct_letter,
        "correct_text": correct_text,
        "rationale": rationale,
        "source": source
    }

@app.post("/quiz/save-result")
async def save_quiz_result_direct(request: Request):
    """Direct result save for drill mode — bypasses Brain AI."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    body = await request.json()
    result = db_save_quiz_result(
        body.get("topic", ""),
        body.get("question", ""),
        body.get("result", "wrong"),
        body.get("note_id")
    )
    return result

@app.get("/weekly-plan")
async def get_weekly_plan(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    profile = db_get_profile()
    plan_raw = profile.get("weekly_plan", "")
    if not plan_raw or not plan_raw.strip():
        return {"work_days": [], "events": [], "week_of": "", "notes": ""}
    try:
        return json.loads(plan_raw)
    except Exception:
        return {"work_days": [], "events": [], "week_of": "", "notes": ""}

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
    local_time: Optional[str] = None  # e.g. "5/5/26 11:26 PM" from browser

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

    analyze_prompt = f"""You are Brain, Hannah's personal AI assistant. She has asked you to analyze her daily log.

{profile_text}

Daily log — {log_date}:
{log_plain}

Write a coaching reflection in exactly 3 paragraphs. Each paragraph covers one distinct lens — do NOT repeat any theme, insight, or recommendation across paragraphs:

Paragraph 1 — What went well: 2-3 specific things from today that reflect her values, goals, or growth. Reference actual details from the log.
Paragraph 2 — What drifted: 1-2 honest observations about what was off-track or inconsistent. Be direct but kind. Do not repeat anything from paragraph 1.
Paragraph 3 — One thing to carry forward: A single concrete, actionable focus for tomorrow. Must be fresh — not a repeat of anything already said above.

Rules: No bullet points. No headers. No generic statements. No repeating the same theme in multiple paragraphs. Specific to what she actually logged. Warm but real. 3 paragraphs only."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        messages=[{"role": "user", "content": analyze_prompt}]
    )
    analysis = response.content[0].text.strip() if response.content else "Could not generate analysis."

    # Convert markdown formatting to HTML before saving into the note
    def md_to_html(text: str) -> str:
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        return '<br><br>'.join(p.replace('\n', '<br>') for p in paras)

    # Use browser local time if provided, otherwise fall back to server time
    analyzed_at = body.local_time if body.local_time else datetime.now().strftime("%-m/%-d/%y %-I:%M %p")
    analysis_html = f"<em style='font-size:12px;color:#888'>{analyzed_at}</em><br><br>{md_to_html(analysis)}"
    db_update_section_by_id(note["id"], "ANALYSIS", analysis_html)

    return {"analysis": analysis, "summary": log_date, "saved": True}

# ── Affirmation (fresh each dashboard load, varied focus) ─────────────────────
import random as _random_aff

_AFFIRMATION_FOCUSES = [
    ("her courage to pursue a demanding career while managing her own health",
     ["courage", "career", "health", "strength"]),
    ("her identity as a future PMHNP and what that means for the patients she'll serve",
     ["identity", "purpose", "calling", "empathy"]),
    ("her daily consistency — small habits compounding into something great",
     ["consistency", "habits", "discipline", "routine"]),
    ("releasing the fear of failure and trusting her preparation",
     ["failure", "fear", "confidence", "trust", "preparation"]),
    ("her worth being independent of her productivity or performance",
     ["worth", "value", "self-worth", "enough", "acceptance"]),
    ("the gap between where she is and where she's going — and why that gap is okay",
     ["growth", "progress", "journey", "patience", "learning"]),
    ("her resilience in balancing study, work, health, and personal growth",
     ["resilience", "balance", "stress", "self-management", "endurance"]),
    ("the courage it takes to show up even on hard or tired days",
     ["showing up", "tired", "burnout", "perseverance", "motivation"]),
    ("her values and what grounds her when things feel uncertain",
     ["values", "faith", "grounded", "belief", "purpose", "spiritual"]),
    ("the progress she has already made that she might be underselling",
     ["progress", "achievement", "confidence", "recognition", "growth"]),
]

@app.get("/affirmation")
async def get_affirmation(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    profile_text = build_profile_context()
    if not profile_text:
        return {"affirmation": ""}

    # Pull most recent daily log for extra context
    conn = get_db(); cur = conn.cursor()
    cur.execute(
        "SELECT content FROM notes WHERE category = 'lifestyle' AND subcategory ILIKE '%daily log%' ORDER BY created_at DESC LIMIT 1"
    )
    row = cur.fetchone(); cur.close(); conn.close()
    recent_log = strip_html(row["content"])[:600] if row else ""
    log_snippet = f"\n\nHer most recent daily log entry:\n{recent_log}" if recent_log else ""

    focus_text, keywords = _random_aff.choice(_AFFIRMATION_FOCUSES)

    prompt = (
        f"{profile_text}{log_snippet}\n\n"
        f"Write ONE short, deeply personal affirmation for this person. Focus specifically on: {focus_text}. "
        "It must feel written *for her specifically* — not generic. "
        "Be warm, grounded, and empowering. No cheesy slogans. "
        "1–2 sentences only. No quotes, no labels, just the affirmation."
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip() if response.content else ""
    return {"affirmation": text, "keywords": keywords}


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
    elif topic_lower in ("all_clinical", "all clinical", "clinical"):
        cats = "('psychiatry','psychotherapy','icu','clinical','study')"
    else:
        cats = "('psychiatry','clinical','study')"  # include old category names for backwards compat

    if body.topic and topic_lower not in ("icu", "psychiatry", "psychotherapy", "all_clinical", "all clinical", "clinical"):
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

    # ── Weight toward weak areas ──────────────────────────────────────────
    weak_areas = db_get_weak_areas(limit=5)
    weak_topics = [w["topic"] for w in weak_areas]  # sorted weakest first

    # Separate notes into weak and non-weak buckets
    weak_notes   = [n for n in notes_list if (n["subcategory"] or "") in weak_topics]
    other_notes  = [n for n in notes_list if (n["subcategory"] or "") not in weak_topics]
    _random.shuffle(weak_notes)
    _random.shuffle(other_notes)

    # 70% chance to pick from weak bucket if it has notes, otherwise random
    if weak_notes and _random.random() < 0.70:
        anchor = weak_notes[0]
        ordered_notes = weak_notes + other_notes
    else:
        _random.shuffle(notes_list)
        anchor = notes_list[0]
        ordered_notes = notes_list

    anchor_topic = anchor["subcategory"] or anchor["summary"] or "clinical knowledge"

    # Build context (up to 40 notes, weak-first)
    context_notes = ordered_notes[:40]
    notes_text = "\n\n---\n\n".join(
        f"[{n['subcategory'] or 'Clinical'}] {n['summary']}\n{strip_html(n['content'])[:600]}"
        for n in context_notes
    )
    topic_label = "clinical knowledge" if topic_lower in ("all_clinical", "all clinical", "clinical", "") else (body.topic or "clinical knowledge")

    # Weak area hint for the prompt
    weak_hint = ""
    if weak_areas:
        weak_list = ", ".join(f"{w['topic']} ({w['score_pct']}%)" for w in weak_areas[:3])
        weak_hint = f"\nHer current weak areas (lowest scores): {weak_list}. Prioritize these topics."

    quiz_prompt = (
        f"You are quizzing Hannah, a PMHNP student preparing for her board exam, using her own saved notes.\n"
        f"Topic: {topic_label}.{weak_hint}\n\n"
        f"Notes:\n{notes_text}\n\n"
        f"Focus this question on: {anchor_topic}\n\n"
        "Write ONE board-style question. Rules:\n"
        "- One clear sentence — no preamble, no 'which of the following'\n"
        "- High yield: mechanism of action, first-line treatment, key DSM criteria, black box warning, contraindication, or brief clinical scenario\n"
        "- Open ended — expected answer is 1–3 sentences\n"
        "- Simple language, clinical precision\n"
        "- Do NOT give the answer or any hints\n"
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

@app.get("/daily-focus")
async def get_daily_focus(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return db_get_daily_focus()

class DailyFocusRequest(BaseModel):
    priorities: list
    study_focus: Optional[str] = ""
    date_str: Optional[str] = ""

@app.post("/daily-focus")
async def set_daily_focus(body: DailyFocusRequest, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    date_str = body.date_str or datetime.now().strftime("%Y-%m-%d")
    return db_save_daily_focus(body.priorities, body.study_focus or "", date_str)

@app.get("/quiz/quick-win")
async def quiz_quick_win(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    # Find the weak topic closest to mastery (highest score among weak areas)
    weak = db_get_weak_areas(limit=10)
    if weak:
        # Sort by score descending — pick the "almost there" topic
        best = sorted(weak, key=lambda x: x["score_pct"], reverse=True)[0]
        target_topic = best["topic"]
    else:
        target_topic = None
    # Fetch notes for that topic
    conn = get_db(); cur = conn.cursor()
    if target_topic:
        cur.execute(
            "SELECT content, summary, subcategory FROM notes WHERE subcategory = %s ORDER BY RANDOM() LIMIT 20",
            (target_topic,)
        )
    else:
        cur.execute(
            "SELECT content, summary, subcategory FROM notes WHERE category IN ('psychiatry','psychotherapy','icu','np_fellowship') ORDER BY RANDOM() LIMIT 20"
        )
    notes = cur.fetchall(); cur.close(); conn.close()
    if not notes:
        return {"reply": "Add some clinical notes first and I'll find you a quick win! 💪"}
    import random as _r; _r.shuffle(list(notes))
    notes_text = "\n\n---\n\n".join(
        f"[{n['subcategory']}] {n['summary']}\n{strip_html(n['content'])[:400]}"
        for n in notes
    )
    prompt = (
        f"You are quizzing a PMHNP student. She needs a confidence boost — pick the most straightforward, achievable question from this topic: {target_topic or 'clinical knowledge'}.\n\n"
        f"Notes:\n{notes_text}\n\n"
        "Write ONE easy but meaningful question — something she likely knows or is close to knowing. "
        "A clear definition, a well-known first-line treatment, or a basic DSM criterion. "
        "One sentence. Open ended. End with 'You got this! 💪' on a new line."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    question = response.content[0].text.strip() if response.content else "Could not generate a question."
    db_add_message("user", f"Quick win quiz on {target_topic or 'clinical knowledge'}")
    db_add_message("assistant", question)
    topic_label = f"your best chance right now: **{target_topic}**" if target_topic else "clinical knowledge"
    return {"reply": f"⚡ Quick win — {topic_label}\n\n{question}"}

@app.get("/quiz/weak-areas")
async def get_weak_areas(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return db_get_weak_areas(limit=6)

@app.patch("/note/{note_id}")
async def patch_note(note_id: int, request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    body = await request.json()
    fields = {k: v for k, v in body.items() if k in ("content", "summary", "category", "subcategory")}
    if not fields:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    result = db_update_note(note_id, fields)
    return result

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
                {"type": "text", "text": """Extract all text and information from this image. Follow these rules carefully:

1. HIGHLIGHTED or COLORED text — any text that appears highlighted (yellow, blue, etc.), bold, underlined, or in a different color than the surrounding text: wrap it in <strong> tags so it stands out in the note. These are key terms the instructor emphasized.
2. TABLES — preserve the table structure using HTML <table> tags with borders.
3. GRAPHS/DIAGRAMS — describe what the graph shows, label axes, note key values.
4. BULLET POINTS/LISTS — use <ul><li> tags.
5. HEADINGS — use <strong> tags.
6. URLs — write in full format starting with https://.
7. Format everything as clean HTML — no markdown asterisks or pound signs.
8. Extract EVERYTHING visible — do not skip any text even if small or in margins."""}
            ]
        }]
    )
    raw = response.content[0].text if response.content else ""
    # Strip any markdown code fences the model may have wrapped the HTML in
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw.strip())
    raw = re.sub(r"```\s*$", "", raw.strip())
    return raw.strip()

def save_image_note(data: bytes, media_type: str, filename: str, description: str, user_note: str, text_only: bool = False) -> str:
    """Save image as a note. If text_only=True, saves extracted text only (no embedded image)."""
    if text_only:
        # Text-only: skip embedding the image, just save the extracted description
        content = (
            f'<div style="font-size:14px;line-height:1.7">'
            f'<div>{description}</div>'
            + (f'<div style="margin-top:10px;font-style:italic;color:#888">{user_note}</div>' if user_note else '')
            + '</div>'
        )
    else:
        b64 = base64.standard_b64encode(data).decode("utf-8")
        img_tag = f'<img src="data:{media_type};base64,{b64}" style="max-width:40%;border-radius:10px;margin-bottom:12px;display:block">'
        # Build note content: image on top, description below
        content = (
            f'<div style="font-size:14px;line-height:1.7">'
            f'{img_tag}'
            f'<div style="margin-top:10px">{description}</div>'
            + (f'<div style="margin-top:10px;font-style:italic;color:#888">{user_note}</div>' if user_note else '')
            + '</div>'
        )

    # Get metadata from description
    meta_prompt = (
        f"An image was saved with this description: {description[:500]}\n"
        + (f"User note: {user_note}\n" if user_note else "")
        + "ROUTING RULES:\n"
        + "⛔ If this image contains ANY spiritual/faith content — devotions, Bible verses, scripture, sermons, prayers, worship, faith reflections — set category=lifestyle AND subcategory=Daily Log.\n"
        + "✅ category=boards ONLY if the image is explicitly a board exam practice question (has A/B/C/D answer choices) or is labeled as ANCC/board prep material. Subcategory: Assessment & Diagnosis | Psychopharmacology | Psychotherapy | Medical Management | Special Populations | Professional & Ethics | Board Prep.\n"
        + "✅ Lecture slides, class notes, pharmacology slides, DSM content, clinical assessments, medication info, neuroscience → category=psychiatry. Pick the best subcategory: DSM-5 | Medications | Assessments | Treatments | Lab Values | Neuroscience | Ethics & Law.\n"
        + "✅ Psychotherapy models, therapy techniques → category=psychotherapy.\n"
        + "Return ONLY a JSON object: "
        '{"summary": "short title for this image", "category": "personal|lifestyle|people|psychiatry|psychotherapy|icu|np_fellowship|business|resources|mom|garden|boards", '
        '"subcategory": "subcategory or null", "tags": ["tag1"]}\n'
        "Return ONLY the JSON."
    )
    meta_response = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=256,
        messages=[{"role": "user", "content": meta_prompt}]
    )
    meta_text = meta_response.content[0].text.strip() if meta_response.content else ""
    try:
        if meta_text.startswith("```"):
            meta_text = meta_text.split("```")[1]
            if meta_text.startswith("json"): meta_text = meta_text[4:]
        meta = json.loads(meta_text)
        summary     = meta.get("summary", filename or "Image")
        category    = meta.get("category", "personal")
        subcategory = meta.get("subcategory")
        tags        = meta.get("tags", [])
    except Exception:
        summary, category, subcategory, tags = filename or "Image", "personal", None, []

    db_save_note(f"[Image: {filename}]", content, summary, category, subcategory, tags, [])
    loc = f"<strong>{category}</strong>" + (f" → {subcategory}" if subcategory else "")
    if text_only:
        reply = (
            f"📝 Text saved (no image) in {loc}.<br><br>"
            f"<strong>{summary}</strong><br><br>"
            + description[:300]
            + ("..." if len(description) > 300 else "")
        )
    else:
        reply = (
            f"📸 Saved! Photo stored in {loc}.<br><br>"
            f"<strong>{summary}</strong><br><br>"
            + description[:300]
            + ("..." if len(description) > 300 else "")
        )
    db_add_message("assistant", reply)
    return reply

_BOARD_TOPIC_MAP = {
    "assessment": "Assessment & Diagnosis",
    "diagnosis": "Assessment & Diagnosis",
    "psychopharmacology": "Psychopharmacology",
    "pharmacology": "Psychopharmacology",
    "medication": "Psychopharmacology",
    "psychotherapy": "Psychotherapy",
    "therapy": "Psychotherapy",
    "cbt": "Psychotherapy",
    "dbt": "Psychotherapy",
    "medical": "Medical Management",
    "special population": "Special Populations",
    "child": "Special Populations",
    "geriatric": "Special Populations",
    "ethics": "Professional & Ethics",
    "legal": "Professional & Ethics",
    "professional": "Professional & Ethics",
}

def _looks_like_board_questions(text: str, user_note: str) -> bool:
    """Detect if content is a practice question PDF or image."""
    note_lower = (user_note or "").lower()
    if any(kw in note_lower for kw in ["board", "question", "quiz", "practice", "georgette", "blueprint", "ancc", "pmhnp"]):
        return True
    # Match both A) and A. formats (e.g. "A. ADHD" or "A) ADHD")
    # 4+ matches = at least one full A/B/C/D question
    matches = len(re.findall(r'\b[A-D][).]', text))
    return matches >= 4

def _normalize_topic(raw: str) -> str:
    """Map a raw topic string to a canonical board subcategory."""
    low = raw.lower()
    for key, val in _BOARD_TOPIC_MAP.items():
        if key in low:
            return val
    return raw.strip() or "Assessment & Diagnosis"

def parse_and_save_board_questions(extracted: str, source_name: str) -> str:
    """Parse a practice question document and save each Q as a board note."""
    # Truncate to avoid token limits but keep as much as possible
    chunk = extracted[:12000]
    parse_prompt = (
        "You are parsing a psychiatric board exam practice question document.\n"
        "Extract EVERY question you can find. Return a JSON array where each item has:\n"
        '{"topic": "one of: Assessment & Diagnosis | Psychopharmacology | Psychotherapy | Medical Management | Special Populations | Professional & Ethics", '
        '"question": "full question text", '
        '"choices": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
        '"correct_letter": "A|B|C|D", '
        '"correct_text": "text of correct choice", '
        '"rationale": "explanation if present, else empty string"}\n'
        "Rules:\n"
        "- Include ALL questions found, even if rationale is missing\n"
        "- correct_letter must be a single capital letter A-D\n"
        "- If you cannot determine the correct answer, skip that question\n"
        "- Return ONLY the JSON array, no other text\n\n"
        f"Document:\n{chunk}"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        messages=[{"role": "user", "content": parse_prompt}]
    )
    raw = response.content[0].text.strip() if response.content else "[]"
    # Strip code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        questions = json.loads(raw)
    except Exception:
        return "⚠️ Could not parse questions from this PDF. Try pasting the text directly or using the Board Q button to add questions manually."

    if not questions:
        return "⚠️ No questions found in this PDF. Make sure it contains A/B/C/D multiple choice questions."

    saved = 0
    for q in questions:
        try:
            topic = _normalize_topic(q.get("topic", ""))
            question_text = q.get("question", "").strip()
            choices = q.get("choices", {})
            correct_letter = q.get("correct_letter", "").strip().upper()
            correct_text = q.get("correct_text", choices.get(correct_letter, "")).strip()
            rationale = q.get("rationale", "").strip()
            if not question_text or not correct_letter or not choices:
                continue
            # Format matching the regex in /board-quiz/random
            choice_lines = "\n".join(f"{l}) {choices[l]}" for l in ["A","B","C","D"] if choices.get(l))
            content = (
                f"**Q:** {question_text}\n"
                f"{choice_lines}\n"
                f"✅ **Correct: {correct_letter}) {correct_text}**\n"
                + (f"**Rationale:** {rationale}\n" if rationale else "")
                + f"*Source: {source_name}*"
            )
            summary = question_text[:60] + ("…" if len(question_text) > 60 else "")
            db_save_note(f"[Board Q: {topic}]", content, summary, "boards", topic,
                         ["board exam", "PMHNP", topic.lower()], [])
            saved += 1
        except Exception:
            continue

    if saved == 0:
        return "⚠️ Found questions but couldn't save them — the format may be unusual. Try the Board Q button to add manually."

    return (
        f"✅ <strong>Saved {saved} board question{'s' if saved != 1 else ''}</strong> from {source_name}!<br><br>"
        f"They're now in your drill pool with spaced repetition active. "
        f"Head to <strong>⚡ Drill</strong> or <strong>🎯 Quick Win</strong> to start practicing. 📚"
    )

def _looks_like_study_plan(text: str, user_note: str) -> bool:
    """Detect if content is a study schedule/plan (not clinical content to quiz on)."""
    note_lower = (user_note or "").lower()
    if any(kw in note_lower for kw in ["study plan", "schedule", "georgette", "week 1", "week by week"]):
        return True
    text_lower = text.lower()
    week_count = len(re.findall(r'week\s+\d', text_lower))
    return week_count >= 3


def _looks_like_study_guide(text: str, user_note: str) -> bool:
    """Detect if content is a study guide / clinical notes (not pre-made Q&A)."""
    note_lower = (user_note or "").lower()
    if any(kw in note_lower for kw in ["study guide", "studyguide", "generate question", "make question", "create question", "clinical notes", "sarah"]):
        return True
    # Study guides have lots of bullets and few A) B) C) D) choices
    bullet_count = len(re.findall(r'^\s*[•\-\*]', text, re.MULTILINE))
    choice_count = len(re.findall(r'\b[A-D]\)', text))
    return bullet_count > 20 and choice_count < 8


def _clean_pdf_text(text: str) -> str:
    """Collapse excessive whitespace caused by character-spaced PDFs.
    Converts 'W o r d  b y  w o r d' or words-on-separate-lines into readable prose."""
    import re as _re
    # Collapse runs of whitespace (spaces, tabs, newlines) into a single space
    text = _re.sub(r'[ \t]+', ' ', text)          # multiple spaces/tabs → one space
    text = _re.sub(r'\n[ \t]*\n+', '\n', text)    # multiple blank lines → one newline
    text = _re.sub(r'(\w)\n(\w)', r'\1 \2', text) # word\nword → word word (broken lines)
    text = _re.sub(r'\n{3,}', '\n\n', text)        # 3+ newlines → 2
    # Remove lone page numbers (lines that are just a number)
    text = _re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=_re.MULTILINE)
    return text.strip()


def generate_questions_from_study_guide(extracted: str, source_name: str) -> str:
    """
    Take a clinical study guide (bullet points / prose), chunk it, generate
    ANCC-style multiple-choice questions, and save them to the board drill bank.
    """
    # Clean up whitespace-heavy PDF extraction first
    extracted = _clean_pdf_text(extracted)

    CHUNK_SIZE = 6000  # chars per Claude call (larger since text is now denser)
    SYSTEM_MSG = (
        "You are an expert ANCC PMHNP-BC board exam question writer. "
        "Given clinical study guide content (bullet points, notes, clinical pearls), "
        "generate high-quality ANCC-style multiple-choice questions that test clinical reasoning.\n\n"
        "Rules:\n"
        "- Each question must have exactly 4 choices (A, B, C, D)\n"
        "- One clearly correct answer, distractors should be plausible\n"
        "- Include a 2-3 sentence rationale explaining the correct answer\n"
        "- Assign each question to ONE of these 6 ANCC categories: "
        "Assessment & Diagnosis | Psychopharmacology | Psychotherapy | "
        "Medical Management | Special Populations | Professional & Ethics\n"
        "- Skip pure test-taking strategy tips (e.g. 'read the question carefully') but DO generate questions from any clinical facts, medications, diagnoses, or treatments\n"
        "- Generate as many questions as the content supports (aim for 3-6 per chunk)\n"
        "- If a chunk has ZERO clinical facts at all, return []\n"
        "- The text may have imperfect formatting from PDF extraction — do your best to interpret it\n\n"
        "Return ONLY valid JSON — an array of objects:\n"
        '[{"topic":"Psychopharmacology","question":"A 34-year-old...","choices":{"A":"...","B":"...","C":"...","D":"..."},'
        '"correct_letter":"B","correct_text":"Lithium","rationale":"Lithium is first-line..."}]'
    )

    # Split into chunks
    chunks = []
    for i in range(0, len(extracted), CHUNK_SIZE):
        chunk = extracted[i:i + CHUNK_SIZE].strip()
        if len(chunk) > 150:  # skip tiny fragments
            chunks.append(chunk)

    if not chunks:
        return "⚠️ Could not extract enough content from this study guide."

    total_saved = 0
    topic_tally: dict = {}
    first_error: str = ""
    api_errors = 0

    for idx, chunk in enumerate(chunks):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=3000,
                system=SYSTEM_MSG,
                messages=[{"role": "user", "content": f"Chunk {idx+1}/{len(chunks)}:\n\n{chunk}"}],
            )
            raw = resp.content[0].text.strip() if resp.content else "[]"
            # Strip code fences
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'```\s*$', '', raw).strip()
            questions = json.loads(raw)
        except Exception as e:
            api_errors += 1
            if not first_error:
                first_error = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[study-guide] chunk {idx+1} error: {type(e).__name__}: {e}")
            continue  # skip bad chunks

        for q in questions:
            try:
                topic = _normalize_topic(q.get("topic", ""))
                question_text = q.get("question", "").strip()
                choices = q.get("choices", {})
                correct_letter = q.get("correct_letter", "").strip().upper()
                correct_text = q.get("correct_text", choices.get(correct_letter, "")).strip()
                rationale = q.get("rationale", "").strip()
                if not question_text or not correct_letter or not choices:
                    continue
                choice_lines = "\n".join(f"{l}) {choices[l]}" for l in ["A","B","C","D"] if choices.get(l))
                content = (
                    f"**Q:** {question_text}\n"
                    f"{choice_lines}\n"
                    f"✅ **Correct: {correct_letter}) {correct_text}**\n"
                    + (f"**Rationale:** {rationale}\n" if rationale else "")
                    + f"*Source: {source_name}*"
                )
                summary = question_text[:60] + ("…" if len(question_text) > 60 else "")
                db_save_note(f"[Study Guide Q: {topic}]", content, summary, "boards", topic,
                             ["board exam", "PMHNP", "study-guide", topic.lower().replace(" & ","-").replace(" ","-")], [])
                total_saved += 1
                topic_tally[topic] = topic_tally.get(topic, 0) + 1
            except Exception:
                continue

    if total_saved == 0:
        if api_errors > 0:
            return f"⚠️ Study guide processed ({len(chunks)} chunks) but all API calls failed. First error: {first_error}"
        return f"⚠️ Processed the study guide ({len(chunks)} chunks, {len(extracted)} chars after cleanup) but Claude returned no questions. The content may be too introductory or strategy-focused."

    breakdown = " | ".join(f"{t}: {n}" for t, n in sorted(topic_tally.items(), key=lambda x: -x[1]))
    return (
        f"✅ <strong>Generated {total_saved} board questions</strong> from <em>{source_name}</em>!<br><br>"
        f"<strong>By category:</strong> {breakdown}<br><br>"
        f"Spaced repetition is active — your weakest topics will appear most often. "
        f"Head to <strong>⚡ Drill</strong> to start practicing. 🎯"
    )


@app.get("/people/upcoming")
async def people_upcoming(request: Request):
    """Find people notes with birthdays or events in the next 7 days."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, summary, content, created_at
        FROM notes
        WHERE category = 'people'
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()

    from datetime import timedelta
    today = datetime.now()
    upcoming = []

    for row in rows:
        content = row['content'] or ''
        date_patterns = re.findall(
            r'(Birthday|Anniversary|Surgery|Event|Appointment|Follow.?up)[:\s]+([A-Za-z]+ \d{1,2}(?:,? \d{4})?)',
            content, re.IGNORECASE
        )
        for event_type, date_str in date_patterns:
            try:
                for year in [today.year, today.year + 1]:
                    try:
                        clean = re.sub(r',?\s*\d{4}', '', date_str).strip()
                        dt = datetime.strptime(f"{clean} {year}", "%B %d %Y")
                        days_until = (dt.date() - today.date()).days
                        if 0 <= days_until <= 7:
                            upcoming.append({
                                'name': row['summary'],
                                'event': event_type,
                                'date': date_str,
                                'days_until': days_until,
                                'note_id': row['id']
                            })
                            break
                    except ValueError:
                        continue
            except Exception:
                continue

    return sorted(upcoming, key=lambda x: x['days_until'])


@app.get("/people/followup")
async def people_followup(request: Request):
    """People notes not updated in 14+ days."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT summary, created_at
        FROM notes
        WHERE category = 'people'
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()

    today = datetime.now()
    seen = {}
    for row in [_fix_ts(r) for r in rows]:
        name = row['summary']
        if name not in seen:
            seen[name] = row['created_at']

    stale = []
    for name, last_updated in seen.items():
        try:
            dt = datetime.fromisoformat(last_updated.replace('Z', ''))
            days = (today - dt).days
            if days >= 14:
                stale.append({'name': name, 'days_since': days})
        except Exception:
            continue

    return sorted(stale, key=lambda x: -x['days_since'])[:5]


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
            # Images: store the actual photo + Brain's description
            # Size check — reject files over 8MB
            if len(data) > 8 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image too large. Please use screenshots or photos under 8MB.")
            description = extract_image_text(data, content_type)
            if not description.strip():
                raise HTTPException(status_code=400, detail="Could not read this image.")
            # Spiritual content → skip auto-save, route through chat for discussion
            if _is_spiritual(description):
                user_msg = (
                    f"[DEVOTION IMAGE — do NOT save the full text, do NOT save the image. "
                    f"Read this devotion and teach me about it. Have a warm discussion with me. "
                    f"Save only the key scripture reference and one-line insight to today's daily log SPIRITUAL section.]\n\n"
                    f"{description}"
                    + (f"\n\nMy note: {note.strip()}" if note.strip() else "")
                )
                db_add_message("user", f"[Attached devotion image: {filename}]")
                result = run_agent(user_msg)
                return {"reply": result["reply"]}
            # Board question image → drill bank only (no photo needed)
            if _looks_like_board_questions(description, note.strip()):
                reply = parse_and_save_board_questions(description, filename or "Image")
                db_add_message("assistant", reply)
                return {"reply": reply}
            # Check if user explicitly wants text-only (no embedded image)
            note_lower = note.strip().lower()
            text_only = any(kw in note_lower for kw in [
                "text only", "text-only", "save text", "no image", "don't save image",
                "dont save image", "just the text", "only text", "save the text",
                "notes only", "note only",
            ])
            reply = save_image_note(data, content_type, filename or "screenshot", description, note.strip(), text_only=text_only)
            return {"reply": reply}
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Supported: images (screenshots/photos), PDF, Word, Excel, CSV.")

        if not extracted.strip():
            raise HTTPException(status_code=400, detail="Could not extract any content from this file.")

        # Board question PDFs → parse and save each question individually
        if (filename.endswith(".pdf") or content_type == "application/pdf") and _looks_like_board_questions(extracted, note):
            reply = parse_and_save_board_questions(extracted, filename or "Practice PDF")
            db_add_message("assistant", reply)
            return {"reply": reply}

        # Study plan PDFs → save as a boards planning note
        if (filename.endswith(".pdf") or content_type == "application/pdf") and _looks_like_study_plan(extracted, note):
            note_id = db_save_note(
                raw_input=f"[Uploaded: {filename}]",
                content=_clean_pdf_text(extracted),
                summary=f"📅 Study Plan: {filename.replace('.pdf','').replace('_',' ')}",
                category="boards",
                subcategory="Study Plan",
                tags=json.dumps(["study-plan", "boards", "georgette"]),
                entities=json.dumps([])
            )
            reply = f"📅 **Study plan saved!** I've added *{filename}* to your Boards notes as a study plan.\n\nI can now help you:\n- Build a personalized week-by-week schedule based on your exam date\n- Map Georgette's plan to your Brain focus areas\n- Set daily study goals\n\nJust tell me your target exam date and I'll help you plan it out! 🎓"
            db_add_message("assistant", reply)
            return {"reply": reply}

        # Study guide PDFs → generate ANCC questions from clinical content
        if (filename.endswith(".pdf") or content_type == "application/pdf") and _looks_like_study_guide(extracted, note):
            reply = generate_questions_from_study_guide(extracted, filename or "Study Guide")
            db_add_message("assistant", reply)
            return {"reply": reply}

        # Save the file content as its own note
        file_reply = run_upload_agent(file_label, extracted, "")

        # If user also typed a message, process it separately through the full agent
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


# ── User settings (cross-device sync) ─────────────────────────────────────────

@app.get("/api/settings")
async def get_settings(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM user_settings")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {r["key"]: r["value"] for r in rows}

@app.post("/api/settings")
async def save_setting(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401)
    body = await request.json()
    key   = body.get("key", "").strip()
    value = body.get("value", "")
    if not key:
        raise HTTPException(status_code=400, detail="key required")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_settings (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
    """, (key, value))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}
