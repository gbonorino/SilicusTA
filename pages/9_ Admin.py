# pages/9_Admin.py  –  Streamlit Admin console
from __future__ import annotations
import base64, hashlib, json, os, shutil, tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import json, pathlib
from json import JSONDecodeError

import requests
import streamlit as st
import pandas as pd

import base64, streamlit.components.v1 as components

# --------------------------------------------------------------------------- #
# 0. CONSTANTS & HELPERS
st.set_page_config(page_title="Silicus Admin", page_icon="🛠️")
DATA_ROOT = Path(__file__).parents[1] / "data"         # data/<course>/
MAX_COURSE_MB = 300                                    # hard cap
GH_API = "https://api.github.com"
HEADERS = {"Authorization": f"token {st.secrets['GH_TOKEN']}"}
GH_REPO = st.secrets["GH_REPO"]                        # "user/repo"


def safe_load_json(path: pathlib.Path) -> dict:
    """Return {} if file missing or malformed."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except JSONDecodeError:
        return {}


def discover_courses(root: Path) -> Dict[str, Path]:
    """Return {slug: parquet_path} for every existing course store."""
    return {p.parent.name: p for p in root.glob("*/*_pages.parquet")}

def file_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def bytes_mb(n_bytes: int) -> float:
    return n_bytes / 1_048_576

def folder_size(path: Path) -> float:
    return bytes_mb(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()))

# GitHub helper ------------------------------------------------------------- #
def github_upsert(repo_path: str, content: bytes, msg: str):
    """Create/update a file in GitHub repo via REST."""
    url = f"{GH_API}/repos/{GH_REPO}/contents/{repo_path}"
    resp = requests.get(url, headers=HEADERS)
    sha = resp.json().get("sha") if resp.status_code == 200 else None
    payload = {
        "message": msg,
        "branch": "main",
        "content": base64.b64encode(content).decode("utf-8"),
        **({"sha": sha} if sha else {}),
    }
    r = requests.put(url, headers=HEADERS, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()["commit"]["sha"]

# reuse the pipeline embedding routine
from src.precompute_embeddings import process_course   # noqa: E402

# --------------------------------------------------------------------------- #
# 1.  ENHANCED ADMIN AUTH
if "admin_ok" not in st.session_state:
    st.title("🔒 Admin login")
    pw = st.text_input("Password", type="password")
    
    # Add rate limiting for security
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    
    if pw:
        if pw == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.admin_ok = True
            st.session_state.auth_time = datetime.now()
            st.session_state.login_attempts = 0
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            wait_time = min(5 * (2 ** (st.session_state.login_attempts - 1)), 60)
            st.error(f"Invalid password. Please wait {wait_time} seconds before trying again.")
            import time
            time.sleep(wait_time)
    st.stop()

# Check session expiration (2 hours)
if "auth_time" in st.session_state:
    if (datetime.now() - st.session_state.auth_time).total_seconds() > 7200:  # 2 hours
        st.warning("Your session has expired. Please log in again.")
        st.session_state.pop("admin_ok")
        st.session_state.pop("auth_time")
        st.rerun()

# Add logout button to sidebar
with st.sidebar:
    if st.button("Logout Admin"):
        st.session_state.pop("admin_ok", None)
        st.session_state.pop("auth_time", None)
        st.rerun()

# --------------------------------------------------------------------------- #
# 2. COURSE CARDS
st.title("🛠️ Silicus TA – Course Manager")

COURSES = discover_courses(DATA_ROOT)

st.subheader("Existing courses")
if COURSES:
    # Use simple columns
    for i in range(0, len(COURSES), 3):
        # Create a row of 3 columns
        cols = st.columns(3)
        
        # Fill each column with a course (if available)
        for j in range(3):
            if i+j < len(COURSES):
                slug = sorted(COURSES.keys())[i+j]
                pq_path = COURSES[slug]
                
                with cols[j]:
                    # Simple card with border
                    st.markdown("---")
                    
                    # Course title
                    meta_path = pq_path.parent / "meta.json"
                    meta = safe_load_json(meta_path)
                    st.subheader(meta.get('title', slug.upper()))
                    
                    # Course stats
                    pdf_dir = pq_path.parent / "pdfs"
                    n_pdfs = len(list(pdf_dir.glob("*.pdf")))
                    folder_mb = folder_size(pq_path.parent)
                    
                    st.write(f"📄 {n_pdfs} PDFs • 💾 {folder_mb:.1f} MB")
                    st.write(f"🔄 Updated {datetime.fromtimestamp(pq_path.stat().st_mtime).date()}")
                    
                    # Manage button
                    if st.button("Manage", key=f"manage_{slug}", use_container_width=True):
                        st.session_state.manage_slug = slug
                        st.rerun()
                    
                    st.markdown("---")
else:
    st.info("No courses yet. Use **Create new course** below.")

# --------------------------------------------------------------------------- #
# 3. CREATE COURSE
with st.expander("➕  Create new course", expanded="manage_slug" not in st.session_state):
    new_slug = st.text_input("Course slug (e.g. econ210)", key="new_slug")
    new_title = st.text_input("Display title", key="new_title")
    new_pdf_files = st.file_uploader("Upload PDF slides",
                                     type="pdf",
                                     accept_multiple_files=True,
                                     key="new_pdfs")

    if st.button("Create course") and new_slug and new_pdf_files:
        course_dir = DATA_ROOT / new_slug / "pdfs"
        if course_dir.exists():
            st.error("Slug already exists.")
        else:
            course_dir.mkdir(parents=True, exist_ok=True)
            for f in new_pdf_files:
                (course_dir / f.name).write_bytes(f.read())

            # run embeddings right away
            process_course(course_dir.parent, api_key=st.secrets["MISTRAL_API_KEY"])

            # write meta
            meta_path = course_dir.parent / "meta.json"
            meta_path.write_text(json.dumps({
                "title": new_title or new_slug.upper(),
                "updated": datetime.utcnow().isoformat() + "Z"
            }, indent=2))

            # commit everything
            for p in course_dir.parent.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(Path(__file__).parents[1])
                    github_upsert(str(rel).replace("\\", "/"),
                                  p.read_bytes(),
                                  f"{new_slug}: add {p.name}")

            st.success("Course created and committed! Refreshing …")
            st.cache_resource.clear()
            st.rerun()


# --------------------------------------------------------------------------- #
# 4. MANAGE PANEL
# current meta
if "manage_slug" in st.session_state:
    slug = st.session_state.manage_slug
    course_dir = DATA_ROOT / slug
    pdf_dir = course_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    with st.expander("📝 Edit course title"):
        meta_path = course_dir / "meta.json"
        meta = safe_load_json(meta_path)
        new_title = st.text_input("Display title", value=meta.get("title", slug.upper()))
        if st.button("Save title"):
            meta["title"] = new_title.strip() or slug.upper()
            meta["updated"] = datetime.utcnow().isoformat() + "Z"
            meta_path.write_text(json.dumps(meta, indent=2))

            github_upsert(
                str(meta_path.relative_to(Path(__file__).parents[1])),
                meta_path.read_bytes(),
                f"{slug}: rename course to '{meta['title']}'"
            )
            st.success("Title updated!")
            st.cache_resource.clear()
            st.rerun()
        
    st.markdown("---")
    with st.expander("⚠️ Danger Zone - Delete Course", expanded=False):
        st.warning(f"This will permanently delete {slug} course and all its files")
        
        # Use session state to track deletion confirmation
        if "delete_confirmed" not in st.session_state:
            st.session_state.delete_confirmed = False
        
        # Checkbox for confirmation
        delete_confirmed = st.checkbox(
            "I understand this will permanently delete all course files and cannot be undone",
            key=f"confirm_delete_{slug}"
        )
        
        # Only show the confirmation button if checkbox is checked
        if delete_confirmed:
            if st.button("Confirm Deletion", type="primary", key=f"delete_button_{slug}"):
                try:
                    # Remove files from GitHub first
                    for p in course_dir.rglob("*"):
                        if p.is_file():
                            rel_path = str(p.relative_to(Path(__file__).parents[1])).replace("\\", "/")
                            url = f"{GH_API}/repos/{GH_REPO}/contents/{rel_path}"
                            resp = requests.get(url, headers=HEADERS)
                            if resp.status_code == 200:
                                sha = resp.json().get("sha")
                                payload = {
                                    "message": f"Delete {slug} course files",
                                    "branch": "main",
                                    "sha": sha
                                }
                                requests.delete(url, headers=HEADERS, data=json.dumps(payload))
                    
                    # Then delete local directory
                    shutil.rmtree(course_dir)
                    st.session_state.pop("manage_slug", None)
                    st.cache_resource.clear()
                    st.success(f"Course '{slug}' deleted successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting course: {e}")

    st.header(f"Manage course: {slug}")
    upload_files = st.file_uploader("Add / replace PDFs",
                                    type="pdf",
                                    accept_multiple_files=True,
                                    key=f"upload_{slug}")

    if "upload_queue" in st.session_state:
        # carry over files from new‑course flow
        upload_files = upload_files or st.session_state.pop("upload_queue")

    # ---------- DEDUP & SAVE ------------------------------------------------ #
    if st.button("Save PDFs to workspace") and upload_files:
        saved, skipped = 0, 0
        for file in upload_files:
            b = file.read()
            if any(file_sha256(b) == file_sha256(p.read_bytes()) for p in pdf_dir.glob("*.pdf")):
                skipped += 1
                continue
            (pdf_dir / file.name).write_bytes(b)
            saved += 1
        st.success(f"Saved {saved}; skipped {skipped} duplicates.")
        st.rerun()

    st.write(f"📁  Current folder size: **{folder_size(course_dir):.1f} MB** / {MAX_COURSE_MB} MB")
    if folder_size(course_dir) > MAX_COURSE_MB:
        st.error("Folder exceeds limit. Delete slides or split the course before embedding.")
        st.stop()
    
    # ---------- LIST & DELETE ---------------------------------------------- #
    st.subheader("Current slides")
    for p in sorted(pdf_dir.glob("*.pdf")):
        col1, col2, col3 = st.columns([4, 1, 1], gap="small")

        # filename
        col1.write(p.name)

        # download PDF feature
        if col2.button("👁️ View", key=f"view_{p.name}"):
            with open(p, "rb") as file:
                col2.download_button(
                    label="📄 Download",
                    data=file,
                    file_name=p.name,
                    mime="application/pdf",
                    key=f"dl_admin_{p.name.replace('.', '_')}"
                )

        # 🗑️  delete file
        if col3.button("🗑️ Delete", key=f"del_{p.name}"):
            p.unlink()
            st.warning(f"Deleted {p.name}")
            st.rerun()

    # ---------- EMBED & COMMIT --------------------------------------------- #
    if st.button("Rebuild embeddings ➜ Commit to GitHub"):
        with st.spinner("Running OCR + embeddings … this may take a minute"):
            process_course(course_dir, api_key=st.secrets["MISTRAL_API_KEY"])

        # write / update meta.json
        meta_path = course_dir / "meta.json"
        meta = {"title": st.session_state.get("new_course_title", slug.upper()),
                "updated": datetime.utcnow().isoformat() + "Z"}
        meta_path.write_text(json.dumps(meta, indent=2))

        # commit all PDFs, Parquet, meta to GitHub
        changed_files: List[Tuple[Path, bytes]] = []
        for p in course_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(Path(__file__).parents[1])
                changed_files.append((rel, p.read_bytes()))

        for rel, b in changed_files:
            github_upsert(str(rel).replace("\\", "/"),
                          b,
                          f"{slug}: update {rel.name}")

        st.success("Committed to GitHub ✅")
        st.toast("✅ Course updated!", icon="🎉")
        st.cache_resource.clear()   # refresh cached embeddings in chat
        st.rerun()
