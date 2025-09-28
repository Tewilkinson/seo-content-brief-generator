import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="SEO Content Brief Generator", layout="wide")
st.title("SEO Content Brief Generator")

# Keys expected in st.secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")

# OpenAI client
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# ---------------------- SESSION STATE ----------------------
def ss_init():
    defaults = {
        "uploaded_urls": [],
        "keyword": "",
        "serp_raw": {},
        "top_pages": [],
        "paa": [],
        "internal_links": [],
        "h2_sections": [],
        "h2_content": [],
        "faqs": [],  # list of dicts {question, answer}
        "brief_table": None,
        "analysis_table": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ss_init()

# ---------------------- HELPERS ----------------------
def fetch_serp(keyword: str, location: str = "United States") -> Dict[str, Any]:
    search = GoogleSearch({"q": keyword, "location": location, "api_key": SERPAPI_KEY})
    return search.get_dict() or {}

def extract_top_pages(serp: Dict[str, Any], n: int = 10) -> List[str]:
    links = []
    for res in serp.get("organic_results", [])[:n]:
        link = res.get("link")
        if link:
            links.append(link)
    return links

def extract_paa(serp: Dict[str, Any], n: int = 10) -> List[str]:
    # SerpAPI key for People Also Ask is typically 'related_questions'
    qs = []
    for item in serp.get("related_questions", [])[:n]:
        q = item.get("question")
        if q:
            qs.append(q)
    return qs

def safe_get(url: str, timeout: int = 12) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SEO-Brief/1.0; +https://example.com/bot)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.ok and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
    except Exception:
        pass
    return ""

def analyze_page(url: str) -> Tuple[str, str, List[str], int]:
    """
    Returns: (title, meta_description, h2_list, outlink_count)
    """
    html = safe_get(url)
    if not html:
        return "", "", [], 0

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()

    # Meta description
    meta_desc = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = md["content"].strip()
    else:
        # Some sites use og:description
        ogd = soup.find("meta", property="og:description")
        if ogd and ogd.get("content"):
            meta_desc = ogd["content"].strip()

    # H2 list (deduplicated, trimmed)
    h2s = []
    for h2 in soup.find_all("h2"):
        txt = h2.get_text(" ", strip=True)
        if txt:
            h2s.append(txt)
    # Deduplicate preserving order
    seen = set()
    h2_clean = []
    for h in h2s:
        if h not in seen:
            seen.add(h)
            h2_clean.append(h)

    # Outlinks (anchor tags with http(s) href)
    outlinks = 0
    page_domain = urlparse(url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("http"):
            # Count as outlink if external (different domain)
            try:
                if urlparse(href).netloc and urlparse(href).netloc != page_domain:
                    outlinks += 1
            except Exception:
                continue

    return title, meta_desc, h2_clean, outlinks

def generate_openai_text(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 400) -> str:
    if not openai_client:
        return "(OpenAI key missing in secrets)"
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

def pick_internal_links(all_urls: List[str], keyword: str) -> List[str]:
    k = (keyword or "").lower().strip()
    if not k:
        return []
    picks = [u for u in all_urls if k in u.lower()]
    # Fallback: up to 10 recs if none matched
    if not picks:
        picks = all_urls[:10]
    return picks[:20]

# ---------------------- UI: INPUTS ----------------------
with st.sidebar:
    st.subheader("Upload URLs (optional)")
    uploaded_file = st.file_uploader("CSV with a single column of URLs", type=["csv"])
    if uploaded_file:
        try:
            df_urls = pd.read_csv(uploaded_file)
            urls_list = df_urls.iloc[:, 0].dropna().astype(str).tolist()
            st.session_state.uploaded_urls = urls_list
            st.success(f"Loaded {len(urls_list)} URLs.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# Tabs
tab1, tab2 = st.tabs(["Generate SEO Brief", "Top Ranking Analysis"])

# ---------------------- TAB 1: SEO BRIEF ----------------------
with tab1:
    st.header("SEO Content Brief")

    st.session_state.keyword = st.text_input("Enter Primary Keyword", value=st.session_state.keyword)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        location = st.text_input("Search Location (SerpAPI)", value="United States")
    with col_b:
        topn = st.number_input("Top N results to analyze", min_value=1, max_value=20, value=10, step=1)

    generate = st.button("Generate Brief", type="primary", use_container_width=True)

    if generate:
        if not SERPAPI_KEY:
            st.error("Missing SERPAPI_KEY in `st.secrets`.")
        elif not st.session_state.keyword.strip():
            st.error("Please enter a keyword.")
        else:
            stages = [
                "Fetching SERP data",
                "Extracting top pages & PAA",
                "Computing internal links",
                "Drafting H2 sections",
                "Writing H2 content",
                "Answering FAQs",
                "Compiling brief",
                "Analyzing top pages",
            ]
            progress = st.progress(0)
            step = 0
            def tick():
                nonlocal step
                step += 1
                progress.progress(step / len(stages))

            # Stage 1: SERP
            st.info(stages[step])
            serp = fetch_serp(st.session_state.keyword, location=location)
            st.session_state.serp_raw = serp
            tick()

            # Stage 2: top pages + PAA
            st.info(stages[step])
            top_pages = extract_top_pages(serp, n=topn)
            paa = extract_paa(serp, n=10)
            st.session_state.top_pages = top_pages
            st.session_state.paa = paa
            tick()

            # Stage 3: internal links
            st.info(stages[step])
            internal_links = pick_internal_links(st.session_state.uploaded_urls, st.session_state.keyword)
            st.session_state.internal_links = internal_links
            tick()

            # Stage 4: Draft H2 sections (use LLM to propose)
            st.info(stages[step])
            h2_sections_prompt = f"""
You are creating an SEO content outline.
Primary keyword: "{st.session_state.keyword}"

Propose 6–8 highly relevant H2 sections that comprehensively cover the topic.
Return them as a simple bullet list with no extra commentary.
"""
            h2_sections_text = generate_openai_text(h2_sections_prompt, max_tokens=300)
            # Parse bullets to list
            h2_sections = []
            for line in h2_sections_text.splitlines():
                line = line.strip("-• \t").strip()
                if line:
                    h2_sections.append(line)
            # Fallback if parsing failed
            if not h2_sections:
                h2_sections = [f"{st.session_state.keyword} – Section {i+1}" for i in range(6)]
            st.session_state.h2_sections = h2_sections[:8]
            tick()

            # Stage 5: H2 content
            st.info(stages[step])
            h2_content = []
            for section in st.session_state.h2_sections:
                content_prompt = f"Write ~120–150 words for the section: **{section}**. Be concise and helpful. Avoid fluff."
                h2_content.append(generate_openai_text(content_prompt, max_tokens=220))
            st.session_state.h2_content = h2_content
            tick()

            # Stage 6: FAQ answers
            st.info(stages[step])
            faqs = []
            for q in (st.session_state.paa or [])[:6]:
                faq_ans = generate_openai_text(
                    f"Provide a concise (2–3 sentences) answer to this FAQ for SEO content: {q}",
                    max_tokens=180,
                )
                faqs.append({"question": q, "answer": faq_ans})
            st.session_state.faqs = faqs
            tick()

            # Stage 7: Compile brief table
            st.info(stages[step])
            est_outlinks = max(10, len(st.session_state.top_pages) * 5)
            url_slug = st.session_state.keyword.lower().strip().replace(" ", "-")

            brief_rows = []
            brief_rows.append(("Title", f"Best {st.session_state.keyword}: Complete Guide & Top Picks"))
            brief_rows.append(("Meta Description", f"Learn everything about {st.session_state.keyword}. Clear sections, FAQs, and expert tips to help you decide."))
            brief_rows.append(("H1", f"{st.session_state.keyword}: The Complete Guide"))
            brief_rows.append(("Navigation Sidebar", "\n".join(st.session_state.h2_sections)))
            brief_rows.append(("People Also Ask", "\n".join(st.session_state.paa)))
            brief_rows.append(("Top Ranking Pages", "\n".join(st.session_state.top_pages)))
            brief_rows.append(("URL Structure", f"https://www.example.com/{url_slug}"))
            brief_rows.append(("Estimated Outlinks", est_outlinks))
            brief_rows.append(("Internal Link Recommendations", "\n".join(st.session_state.internal_links)))
            brief_rows.append(("Number of Page Sections", len(st.session_state.h2_sections)))
            # Add H2 sections with generated copy
            for section, copy in zip(st.session_state.h2_sections, st.session_state.h2_content):
                brief_rows.append((section, copy))
            # FAQs at the end
            faq_block = "\n".join([f"Q: {f['question']}\nA: {f['answer']}" for f in st.session_state.faqs]) if st.session_state.faqs else ""
            brief_rows.append(("FAQs", faq_block))

            brief_df = pd.DataFrame(brief_rows, columns=["Section", "Output"])
            st.session_state.brief_table = brief_df
            tick()

            # Stage 8: Analyze top pages
            st.info(stages[step])
            analysis_data = []
            for url in st.session_state.top_pages:
                title, meta_desc, h2s, outlinks = analyze_page(url)
                analysis_data.append(
                    {
                        "URL": url,
                        "Title": title,
                        "Meta Description": meta_desc,
                        "H2 Sections (sample)": ", ".join(h2s[:8]),
                        "Outlinks (external)": outlinks,
                    }
                )
            analysis_df = pd.DataFrame(analysis_data) if analysis_data else pd.DataFrame(columns=["URL","Title","Meta Description","H2 Sections (sample)","Outlinks (external)"])
            st.session_state.analysis_table = analysis_df

            progress.progress(1.0)
            st.success("SEO Brief Generated ✔")

    # Render result if present
    if st.session_state.brief_table is not None:
        st.dataframe(st.session_state.brief_table, height=720, use_container_width=True)

# ---------------------- TAB 2: TOP RANKING ANALYSIS ----------------------
with tab2:
    st.header("Top Ranking Articles Analysis")
    if st.session_state.analysis_table is not None and not st.session_state.analysis_table.empty:
        # Controls
        with st.expander("Filters"):
            domain_filter = st.text_input("Filter by domain (optional, e.g., 'nytimes.com')", value="")
        df = st.session_state.analysis_table.copy()
        if domain_filter.strip():
            keep = []
            for _, row in df.iterrows():
                try:
                    if urlparse(row["URL"]).netloc.endswith(domain_filter.strip()):
                        keep.append(True)
                    else:
                        keep.append(False)
                except Exception:
                    keep.append(False)
            df = df[keep]
        st.dataframe(df, height=600, use_container_width=True)
    else:
        st.info("No analysis yet. Generate a brief in the first tab.")
