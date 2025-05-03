import streamlit as st
import uuid
from datetime import date
import pandas as pd
import altair as alt
import json
from pathlib import Path
import tempfile, os
import requests  # Already imported
import base64

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(page_title="Study Progress Tracker", layout="wide")
# ── Page heading ─────────────────────────────────────────────────────────────
st.title("CA INTER PROGRESS TRACKER")


# ── Global CSS, background and main heading ───────────────────────────────────
background_image = get_base64_encoded_image("background.jpg")

st.markdown(f"""
<style>
html, body, [data-testid="stApp"] {{
    background-image: url("data:image/jpg;base64,{background_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center center;
}}

div[data-baseweb="tab-list"]        {{ justify-content: space-between !important; }}
div[data-baseweb="tab"]             {{ flex: 1 !important; font-weight: 700 !important; }}
div[data-baseweb="tab"] button p    {{ color: white !important; }}
</style>

<h1 style='text-align:center; margin-top:0.5rem; color:white;'>
    CA Inter Progress Tracker
</h1>
""", unsafe_allow_html=True)

# ── Persistence helpers ──────────────────────────────────────────────────────
DATA_PATH = Path("progress.json")     # Keep beside the script
BASE_STATE_PATH = Path("base_state.json")  # Base structure file

def _load_base_state() -> dict:
    """Load the initial structure from base_state.json"""
    if BASE_STATE_PATH.exists():
        try:
            with open(BASE_STATE_PATH, "r", encoding="utf-8") as f:
                base_state = json.load(f)
                return base_state
        except json.JSONDecodeError:
            st.warning("⚠️  base_state.json is corrupted. Using fallback structure.")
    
    # Fallback structure if base_state.json doesn't exist or is corrupted

def _load_state_from_file() -> dict:
    secrets = st.secrets["github"]
    headers = {
        "Authorization": f"Bearer {secrets['token']}",
        "Accept": "application/vnd.github+json"
    }
    url = f"https://api.github.com/repos/{secrets['repo']}/contents/{secrets['file_path']}?ref={secrets['branch']}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = base64.b64decode(response.json()["content"])
        return json.loads(content)
    else:
        st.warning("⚠️ Could not load progress.json from GitHub. Using base state.")
        return _load_base_state()

def _save_state_to_file() -> None:
    secrets = st.secrets["github"]
    headers = {
        "Authorization": f"Bearer {secrets['token']}",
        "Accept": "application/vnd.github+json"
    }
    url = f"https://api.github.com/repos/{secrets['repo']}/contents/{secrets['file_path']}"

    content_b64 = base64.b64encode(json.dumps(st.session_state["db_state"], indent=2).encode("utf-8")).decode("utf-8")

    # Step 1: Check if file exists to get its SHA
    get_resp = requests.get(url, headers=headers)

    if get_resp.status_code == 200:
        # File exists, update using SHA
        sha = get_resp.json()["sha"]
        payload = {
            "message": "Update progress.json via Streamlit",
            "content": content_b64,
            "branch": secrets["branch"],
            "sha": sha
        }
    elif get_resp.status_code == 404:
        # File does not exist, create it
        payload = {
            "message": "Create progress.json via Streamlit",
            "content": content_b64,
            "branch": secrets["branch"]
        }
    else:
        st.error(f"❌ Unexpected error retrieving file: {get_resp.status_code}")
        return

    # Step 2: PUT to GitHub
    put_resp = requests.put(url, headers=headers, json=payload)

    if put_resp.status_code not in (200, 201):
        st.error(f"❌ Failed to save progress.json: {put_resp.json().get('message')}")

# ── Session-state bootstrap ──────────────────────────────────────────────────
def ensure_state():
    """Initialize session state and extract structure info"""
    if "db_state" not in st.session_state:
        st.session_state["db_state"] = _load_state_from_file()
    
    # Extract subjects, chapters, modules, and phases from the loaded state
    data = st.session_state["db_state"]
    
    # Extract all subjects, chapters, modules, and phases
    all_subjects = list(data["subjects"].keys())
    
    # Extract all unique chapter names across all subjects
    all_chapters = set()
    for subj in all_subjects:
        all_chapters.update(data["subjects"][subj]["chapters"].keys())
    all_chapters = list(all_chapters)
    
    # Extract all unique module names across all subjects and chapters
    all_modules = set()
    for subj in all_subjects:
        for chap in data["subjects"][subj]["chapters"]:
            all_modules.update(data["subjects"][subj]["chapters"][chap]["modules"].keys())
    all_modules = list(all_modules)
    
    # Extract all unique phase names
    all_phases = set()
    for subj in all_subjects:
        for chap in data["subjects"][subj]["chapters"]:
            for mod in data["subjects"][subj]["chapters"][chap]["modules"]:
                all_phases.update(data["subjects"][subj]["chapters"][chap]["modules"][mod]["phases"].keys())
    all_phases = list(all_phases)
    
    # Store these lists in session state for use throughout the app
    st.session_state["SUBJECTS"] = all_subjects
    st.session_state["CHAPTERS"] = all_chapters
    st.session_state["MODULES"] = all_modules
    st.session_state["PHASES"] = all_phases
    
    # Initialize visualization state
    st.session_state.setdefault("viz_level", "subject")
    st.session_state.setdefault("viz_path", [])
    st.session_state.setdefault("viz_subject", None)
    st.session_state.setdefault("viz_chapter", None)
    st.session_state.setdefault("viz_module", None)

    # Remember selected chapter for each subject (left column picker)
    for subj in all_subjects:
        first_chapter = list(data["subjects"][subj]["chapters"].keys())[0]
        st.session_state.setdefault(f"selected_{subj}", first_chapter)
        
        # Remember selected module for each chapter in each subject
        for chap in data["subjects"][subj]["chapters"].keys():
            first_module = list(data["subjects"][subj]["chapters"][chap]["modules"].keys())[0]
            st.session_state.setdefault(f"selected_mod_{subj}_{chap}", first_module)

    # ✅ Fix: Preserve initially selected tab (to avoid fallback on widget rerun)
    if "selected_tab" not in st.session_state:
        st.session_state["selected_tab"] = "Targets"

# Constants for target status
TARGET_STATUS = ["Not started", "<50% done", ">50% done", "Fully achieved"]
STATUS_TO_PCT = dict(zip(TARGET_STATUS, [0, 25, 75, 100]))

# ── CRUD mutators (all call _save_state_to_file) ─────────────────────────────
def update_phase(subject: str, chapter: str, module: str, phase: str,
                 pct: int, notes: str = ""):
    entry = st.session_state["db_state"]["subjects"][subject]["chapters"][chapter]["modules"][module]["phases"][phase]
    entry.update({"status": int(pct), "notes": notes,
                  "last_updated": date.today().isoformat()})
    _save_state_to_file()

def add_target(subject: str, chapter: str, module: str, phase: str,
               tdate: date, description: str):
    st.session_state["db_state"]["targets"].append({
        "id": str(uuid.uuid4()),
        "date": tdate.isoformat(),
        "subject": subject,
        "chapter": chapter,
        "module": module,
        "phase": phase,
        "description": description,
        "status": "Not started",
        "explanation": "",
        "locked": False,
        "update_date": None
    })
    _save_state_to_file()

def update_target(idx: int, status: str, explanation: str):
    tgt = st.session_state["db_state"]["targets"][idx]
    tgt.update({
        "status": status,
        "explanation": explanation,
        "locked": True,
        "update_date": date.today().isoformat()
    })
    _save_state_to_file()

# ── Aggregation helpers ───────────────────────────────────────────────────────
def pct_subject(subject: str) -> float:          # ← return a float
    chapters = st.session_state["db_state"]["subjects"][subject]["chapters"]
    if not chapters:
        return 0.0

    # weighted by *every phase* instead of “1 chapter = 1 vote”
    vals = []
    for chap in chapters.values():
        for mod in chap["modules"].values():
            vals.extend(ph["status"] for ph in mod["phases"].values())

    return round(sum(vals) / len(vals), 1)       # keep one decimal place


def pct_chapter(subject: str, chapter: str) -> int:
    modules = st.session_state["db_state"]["subjects"][subject]["chapters"][chapter]["modules"]
    vals = [ph["status"] for mod in modules.values() 
            for ph in mod["phases"].values()]
    return int(sum(vals) / len(vals)) if vals else 0

def pct_module(subject: str, chapter: str, module: str) -> int:
    phases = st.session_state["db_state"]["subjects"][subject]["chapters"][chapter]["modules"][module]["phases"].values()
    vals = [ph["status"] for ph in phases]
    return int(sum(vals) / len(vals)) if vals else 0

# ── Chart helpers ────────────────────────────────────────────────────────────
def get_chart_data():
    level = st.session_state["viz_level"]
    if level == "subject":
        # Use completion percentage for subjects (not distribution)
        return pd.DataFrame([{
            "name": s, 
            "value": pct_subject(s),  # This is the completion percentage
            "id": s
        } for s in st.session_state["SUBJECTS"]])
    elif level == "chapter":
        subj = st.session_state["viz_subject"]
        return pd.DataFrame([{
            "name": c, 
            "value": pct_chapter(subj, c), 
            "id": f"{subj}|{c}"
        } for c in st.session_state["db_state"]["subjects"][subj]["chapters"].keys()])
    elif level == "module":
        subj = st.session_state["viz_subject"]
        chap = st.session_state["viz_chapter"]
        return pd.DataFrame([{
            "name": m, 
            "value": pct_module(subj, chap, m), 
            "id": f"{subj}|{chap}|{m}"
        } for m in st.session_state["db_state"]["subjects"][subj]["chapters"][chap]["modules"].keys()])
    else:  # phase level
        subj = st.session_state["viz_subject"]
        chap = st.session_state["viz_chapter"]
        mod = st.session_state["viz_module"]
        data = [{
            "name": ph, 
            "value": st.session_state["db_state"]["subjects"][subj]["chapters"][chap]["modules"][mod]["phases"][ph]["status"],
            "id": f"{subj}|{chap}|{mod}|{ph}"
        } for ph in st.session_state["db_state"]["subjects"][subj]["chapters"][chap]["modules"][mod]["phases"].keys()]
        return pd.DataFrame(data)
      
def generate_color_scale(level, items):
    """Generate a color scale for a set of items with unique colors"""
    # More distinct color palettes for each level
    colors = {
        "subject": ["#2986cc", "#c90076", "#38761d", "#ff9900", "#6a329f", "#b45f06", "#85200c", "#274e13", "#0c343d", "#4c1130"],
        "chapter": ["#8fce00", "#ffd966", "#f6b26b", "#93c47d", "#a4c2f4", "#d5a6bd", "#b6d7a8", "#ffe599", "#ea9999", "#b4a7d6"],
        "module": ["#d5a6bd", "#b4a7d6", "#9fc5e8", "#76a5af", "#6fa8dc", "#ffe599", "#a2c4c9", "#d9ead3", "#fce5cd", "#d9d2e9"],
        "phase": ["#6fa8dc", "#9fc5e8", "#cfe2f3", "#f9cb9c", "#ead1dc", "#d9d2e9", "#c27ba0", "#e06666", "#f4cccc", "#b6d7a8"]
    }
    
    # Use the appropriate color palette for the level
    level_colors = colors.get(level, colors["subject"])
    
    # Make sure we have enough colors, without repeating if possible
    if len(items) > len(level_colors):
        # If we need more colors than available, we'll need to create more
        import colorsys
        
        # Generate additional HSV colors with good spacing
        n_additional = len(items) - len(level_colors)
        additional_colors = []
        for i in range(n_additional):
            h = i / n_additional
            s = 0.7  # Medium-high saturation 
            v = 0.9  # High value/brightness
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            additional_colors.append(hex_color)
        
        # Combine original palette with additional colors
        all_colors = level_colors + additional_colors
    else:
        # We have enough colors in our palette
        all_colors = level_colors
    
    # Create a domain-range mapping for the scale
    return alt.Scale(domain=list(items), range=all_colors[:len(items)])

def interactive_pie_chart():
    df = get_chart_data()
    level = st.session_state["viz_level"]

    # Generate color scales based on the actual data
    if level == "subject":
        items = st.session_state["SUBJECTS"]
    elif level == "chapter":
        subj = st.session_state["viz_subject"]
        items = st.session_state["db_state"]["subjects"][subj]["chapters"].keys()
    elif level == "module":
        subj = st.session_state["viz_subject"]
        chap = st.session_state["viz_chapter"]
        items = st.session_state["db_state"]["subjects"][subj]["chapters"][chap]["modules"].keys()
    else:  # phase level
        subj = st.session_state["viz_subject"]
        chap = st.session_state["viz_chapter"]
        mod = st.session_state["viz_module"]
        items = st.session_state["db_state"]["subjects"][subj]["chapters"][chap]["modules"][mod]["phases"].keys()
    
    color_scale = generate_color_scale(level, items)

    # single-point hover selection
    hover = alt.selection_single(fields=["name"],
                                 on="mouseover",
                                 empty="none")

    # tooltip shows name and completion percentage
    tooltip = ["name:N", alt.Tooltip("value:Q", title="Completion %", format=".1f")]

    # base donut with legend
    donut = (
        alt.Chart(df)
        .mark_arc(innerRadius=70, outerRadius=100)
        .encode(
            theta=alt.Theta(field="value", type="quantitative", stack=True),  # Stack ensures proper chart
            color=alt.Color("name:N", scale=color_scale, 
                          legend=alt.Legend(title=level.capitalize(), orient="right")),
            tooltip=tooltip,
            opacity=alt.condition(hover, alt.value(1), alt.value(0.8))
        )
        .add_params(hover)
        .properties(width=400, height=400)
    )

    # text labels show completion percentage
    labels = (
        donut
        .mark_text(radiusOffset=35, fontWeight="bold")
        .encode(text=alt.Text("value:Q", format=".0f"))
    )

    # dynamic centre value – shows name when hovering
    center = (
        alt.Chart(df)
        .transform_filter(hover)
        .mark_text(fontSize=28, fontWeight="bold",
                   align="center", baseline="middle")
        .encode(text=alt.Text("name:N"))
        .properties(width=400, height=400)
    )

    return (donut + labels + center).configure_view(strokeWidth=0)

# ── UI components ────────────────────────────────────────────────────────────
def phase_widget(subject: str, chapter: str, module: str, phase: str):
    ph = st.session_state["db_state"]["subjects"][subject]["chapters"][chapter]["modules"][module]["phases"][phase]
    st.write(f"### {phase}")
    row = st.columns([6,1])
    with row[0]:
        st.progress(ph["status"] / 100)
    with row[1]:
        st.markdown(f"**{ph['status']:.1f}%**")  # Convert percentage to 0-1 scale
    with st.expander("Update"):
        pct = st.slider("Completion %", 0, 100, ph["status"],
                        key=f"pct-{subject}-{chapter}-{module}-{phase}")
        notes = st.text_area("Notes", ph["notes"],
                             key=f"note-{subject}-{chapter}-{module}-{phase}")
        if st.button("Save", key=f"save-{subject}-{chapter}-{module}-{phase}"):
            update_phase(subject, chapter, module, phase, pct, notes)
            st.rerun()

# ——— Layout for subject tab with modules ————————————————————————————————
def tab_progress(subject: str):
    sel_key = f"selected_{subject}"
    subj_pct = pct_subject(subject)
    st.subheader(f"**{subject} Completion: {subj_pct}%**")
    st.progress(subj_pct / 100)
    chosen_chapter = st.session_state[sel_key]
    left, right = st.columns([1, 2])

    # Get chapters for this subject
    subject_chapters = st.session_state["db_state"]["subjects"][subject]["chapters"].keys()

    # left: chapter picker with live %
    with left:
        st.subheader("Chapters")
        for chap in subject_chapters:
            pct = pct_chapter(subject, chap)
            if st.button(f"{chap} — {pct} %", key=f"{subject}-{chap}"):
                st.session_state[sel_key] = chap
                chosen_chapter = chap

    # right: module selector and three phase widgets for chosen module
    with right:
        st.subheader(f"{chosen_chapter} ▶ Modules")
        
        # Get modules for this chapter
        chapter_modules = st.session_state["db_state"]["subjects"][subject]["chapters"][chosen_chapter]["modules"].keys()
        
        # Module selector
        mod_key = f"selected_mod_{subject}_{chosen_chapter}"
        default_module = st.session_state.get(mod_key, list(chapter_modules)[0])
        chosen_module = st.selectbox("Select Module and Update the Phases", chapter_modules, 
                                    index=list(chapter_modules).index(default_module) if default_module in chapter_modules else 0,
                                    key=f"mod-selector-{subject}-{chosen_chapter}")
        st.session_state[mod_key] = chosen_module
        
        # Display module completion
        mod_pct = pct_module(subject, chosen_chapter, chosen_module)
        col1, col2 = st.columns([6,1])  # Wide bar + narrow text
        with col1:
            st.progress(mod_pct / 100)
        with col2:
            st.markdown(f"**{mod_pct:.1f}%**")  # Show percentage neatly

        
        # Get phases for this module
        module_phases = st.session_state["db_state"]["subjects"][subject]["chapters"][chosen_chapter]["modules"][chosen_module]["phases"].keys()
        
        # Phase widgets for selected module
        st.subheader("Phases")
        for ph in module_phases:
            phase_widget(subject, chosen_chapter, chosen_module, ph)


# ——— Updated Targets tab with module selection ————————————————
def tab_targets():
    """Targets tab with fully-dynamic cascading pickers."""
    st.header("🎯 Targets")
    add_tab, live_tab, updated_tab = st.tabs(
        ["➕ Add target", "🟢 Live targets", "✅ Updated targets"]
    )

    # ────────────────────────── ADD TARGET ────────────────────────────
    with add_tab:
        st.subheader("Create a new target")
        data = st.session_state["db_state"]

        # ---- ensure cascade defaults ---------------------------------
        if "tgt_subj" not in st.session_state:
            st.session_state["tgt_subj"] = st.session_state["SUBJECTS"][0]

        if "tgt_chap" not in st.session_state:
            st.session_state["tgt_chap"] = next(
                iter(data["subjects"][st.session_state["tgt_subj"]]["chapters"])
            )

        if "tgt_mod" not in st.session_state:
            st.session_state["tgt_mod"] = next(
                iter(
                    data["subjects"][st.session_state["tgt_subj"]]["chapters"]
                    [st.session_state["tgt_chap"]]["modules"]
                )
            )

        if "tgt_phase" not in st.session_state:
            st.session_state["tgt_phase"] = next(
                iter(
                    data["subjects"][st.session_state["tgt_subj"]]["chapters"]
                    [st.session_state["tgt_chap"]]["modules"]
                    [st.session_state["tgt_mod"]]["phases"]
                )
            )

        # ---- callbacks ------------------------------------------------
        def _on_subject_change():
            s = st.session_state.tgt_subj
            st.session_state.tgt_chap = next(iter(data["subjects"][s]["chapters"]))
            st.session_state.tgt_mod = next(
                iter(
                    data["subjects"][s]["chapters"][st.session_state.tgt_chap][
                        "modules"
                    ]
                )
            )
            st.session_state.tgt_phase = next(
                iter(
                    data["subjects"][s]["chapters"][st.session_state.tgt_chap][
                        "modules"
                    ][st.session_state.tgt_mod]["phases"]
                )
            )

        def _on_chapter_change():
            s, c = st.session_state.tgt_subj, st.session_state.tgt_chap
            st.session_state.tgt_mod = next(
                iter(data["subjects"][s]["chapters"][c]["modules"])
            )
            st.session_state.tgt_phase = next(
                iter(
                    data["subjects"][s]["chapters"][c]["modules"][
                        st.session_state.tgt_mod
                    ]["phases"]
                )
            )

        def _on_module_change():
            s, c, m = (
                st.session_state.tgt_subj,
                st.session_state.tgt_chap,
                st.session_state.tgt_mod,
            )
            st.session_state.tgt_phase = next(
                iter(
                    data["subjects"][s]["chapters"][c]["modules"][m]["phases"]
                )
            )

        # ---- widgets --------------------------------------------------
        # --- Handle tab preservation with date input ---
        # Initialize date state before the widget is created
        if "tgt_due" not in st.session_state:
            st.session_state["tgt_due"] = date.today()

        def on_date_change():
            # Store the selected tab to prevent fallback
            st.session_state["selected_tab"] = "Targets"
            # Immediately re-draw with Targets as the active tab
            st.experimental_rerun()


        # Use the date input with the callback
        due = st.date_input(
            "Due date", 
            value=st.session_state["tgt_due"],
            key="tgt_due",
            on_change=on_date_change
)




        st.selectbox(
            "Subject",
            st.session_state["SUBJECTS"],
            key="tgt_subj",
            on_change=_on_subject_change,
        )

        st.selectbox(
            "Chapter",
            data["subjects"][st.session_state.tgt_subj]["chapters"].keys(),
            key="tgt_chap",
            on_change=_on_chapter_change,
        )

        st.selectbox(
            "Module",
            data["subjects"][st.session_state.tgt_subj]["chapters"][
                st.session_state.tgt_chap
            ]["modules"].keys(),
            key="tgt_mod",
            on_change=_on_module_change,
        )

        st.selectbox(
            "Phase",
            data["subjects"][st.session_state.tgt_subj]["chapters"][
                st.session_state.tgt_chap
            ]["modules"][st.session_state.tgt_mod]["phases"].keys(),
            key="tgt_phase",
        )

        desc = st.text_area(
            "Description: elaborate your target of the day here", key="tgt_desc"
        )

        if st.button("Add target"):
            add_target(
                st.session_state.tgt_subj,
                st.session_state.tgt_chap,
                st.session_state.tgt_mod,
                st.session_state.tgt_phase,
                st.session_state.tgt_due,
                desc,
            )
            st.success("Target added ✅ — see it under *Live targets*")

    # ─────────────────────────── LIVE & DONE LISTS ──────────────────────────
    targets = st.session_state["db_state"]["targets"]
    live_targets = [t for t in targets if not t.get("locked", False)]
    done_targets = [t for t in targets if t.get("locked", False)]

    def render_target_list(tlist, editable: bool, radio_key: str):
        if not tlist:
            st.info("Nothing here yet.")
            return
        order = st.radio(
            "Sort by date",
            ["Earliest first", "Latest first"],
            horizontal=True,
            key=radio_key,
        )
        rev = order == "Latest first"
        for idx, tgt in enumerate(sorted(tlist, key=lambda x: x["date"], reverse=rev)):
            label = (
                f"📌 {tgt['subject']} – {tgt['chapter']} – "
                f"{tgt.get('module','N/A')} – {tgt['phase']} (due {tgt['date']})"
            )
            with st.expander(label):
                st.write(tgt["description"])
                if editable:
                    status = st.radio(
                        "Status",
                        TARGET_STATUS,
                        TARGET_STATUS.index(tgt["status"]),
                        key=f"stat-{idx}",
                    )
                    expl = st.text_area(
                        "Explanation / topics covered",
                        tgt["explanation"],
                        key=f"expl-{idx}",
                    )
                    if st.button("Save", key=f"save-{idx}"):
                        update_target(targets.index(tgt), status, expl)
                        st.rerun()
                else:
                    st.markdown(f"**Description:** {tgt['description']}")
                    st.markdown(
                        f"**Target:** {tgt['subject']} – {tgt['chapter']} – "
                        f"{tgt.get('module','N/A')} – {tgt['phase']} (due {tgt['date']})"
                    )
                    st.markdown(
                        f"**Explanation:** {tgt['explanation'] or '_none_'}"
                    )
                    st.info(f"Status: {tgt['status']}")
                    st.caption(f"Last updated : {tgt.get('update_date','—')}")

    with live_tab:
        render_target_list(live_targets, editable=True, radio_key="order-live")
    with updated_tab:
        render_target_list(
            done_targets, editable=False, radio_key="order-done"
        )

# ——— Updated Visuals tab with module level ————————————————————————————————
def tab_visuals():
    st.header("📊 Interactive Drill-Down Chart")

    if st.session_state["viz_level"] != "subject":
        path = " > ".join(["Subjects"] + st.session_state["viz_path"])
        c1, c2 = st.columns([3, 1])
        c1.write(f"**Path:** {path}")
        if c2.button("⬅️ Go Back"):
            lvl = st.session_state["viz_level"]
            if lvl == "phase":
                st.session_state["viz_level"] = "module"; st.session_state["viz_path"].pop()
            elif lvl == "module":
                st.session_state["viz_level"] = "chapter"; st.session_state["viz_path"].pop()
            elif lvl == "chapter":
                st.session_state["viz_level"] = "subject"; st.session_state["viz_path"] = []
            st.rerun()

    st.write("**Click on segments to drill down**")
    st.altair_chart(interactive_pie_chart(), use_container_width=True)

    # Get data for current level
    df = get_chart_data()
    
    # For better button placement, use a grid layout
    # Calculate how many buttons per row based on data length
    items_per_row = min(4, len(df))  # Max 4 buttons per row
    num_rows = (len(df) + items_per_row - 1) // items_per_row  # Ceiling division
    
    # Create buttons in a grid
    st.write("**Select to drill down:**")
    
    # Create rows
    for row_idx in range(num_rows):
        # Create columns for this row
        cols = st.columns(items_per_row)
        
        # Fill columns with buttons for this row
        for col_idx in range(items_per_row):
            item_idx = row_idx * items_per_row + col_idx
            
            # Check if we still have items to display
            if item_idx < len(df):
                row = df.iloc[item_idx]
                with cols[col_idx]:
                    if st.button(f"{row['name']}", 
                                key=f"select-{row['id']}",
                                use_container_width=True):
                        lvl = st.session_state["viz_level"]
                        if lvl == "subject":
                            st.session_state["viz_subject"] = row["name"]
                            st.session_state["viz_level"] = "chapter"
                        elif lvl == "chapter":
                            st.session_state["viz_chapter"] = row["name"]
                            st.session_state["viz_level"] = "module"
                        elif lvl == "module":
                            st.session_state["viz_module"] = row["name"]
                            st.session_state["viz_level"] = "phase"
                        st.session_state["viz_path"].append(row["name"])
                        st.rerun()  # We still need this, but with better button layout

    lvl = st.session_state["viz_level"]
    if lvl == "subject":
        st.info("Showing progress across all subjects. Click a subject to see its chapters.")
    elif lvl == "chapter":
        st.info(f"Showing chapters for {st.session_state['viz_subject']}. Click a chapter to see modules.")
    elif lvl == "module":
        st.info(f"Showing modules for "
                f"{st.session_state['viz_subject']} > {st.session_state['viz_chapter']}. Click a module to see phases.")
    else:
        st.info(f"Showing phases for "
                f"{st.session_state['viz_subject']} > {st.session_state['viz_chapter']} > {st.session_state['viz_module']}")

    # Rest of the function remains the same
    st.markdown("---")
    st.subheader("Target Status Distribution")
    tgt = st.session_state["db_state"]["targets"]
 
    if tgt:
        counts = {s: 0 for s in TARGET_STATUS}
        for t in tgt: counts[t["status"]] += 1
        df = pd.DataFrame(counts.items(), columns=["status", "count"])
        st.altair_chart(
            alt.Chart(df).mark_arc(innerRadius=60, outerRadius=120).encode(
            theta="count:Q",
            color=alt.Color("status:N",
                            legend=alt.Legend(title="Target Status"))
        ).properties(width=350, height=350),

            use_container_width=True
        )
    else:
        st.info("No targets defined yet.")

# ── Main rendering ───────────────────────────────────────────────────────────
# Replace the render_app() function with this corrected version:

def render_app():
    ensure_state()
    subjects = st.session_state["SUBJECTS"]
    tab_labels = subjects + ["Targets", "Visualizations"]
    
    # Get the currently selected tab from session state, default to "Targets"
    selected_tab = st.session_state.get("selected_tab", "Targets")
    
    # Calculate the index of the selected tab
    tab_index = tab_labels.index(selected_tab)
    
    # Create tabs with the selected tab active
    tabs = st.tabs(tab_labels)
    
    # Process each tab
    for i, tab_name in enumerate(tab_labels):
        with tabs[i]:
            # Only update the session state if we're actually in this tab
            # This prevents the first subject from always being selected during reruns
            if i == tab_index:
                st.session_state["selected_tab"] = tab_name
            
            # Render the appropriate content based on the tab type
            if i < len(subjects):  # Subject tabs
                tab_progress(subjects[i])
            elif tab_name == "Targets":
                tab_targets()
            elif tab_name == "Visualizations":
                tab_visuals()

# ── Run app ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    render_app()