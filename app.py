# app.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="SEO Content Brief Generator", layout="wide")
st.title("SEO Content Brief Generator")

# ---------------------- SECRETS ----------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")

# OpenAI client (optional)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# SerpAPI (provided by package: google-search-results)
_serpapi_available = True
try:
    from serpapi import GoogleSearch
except Exception:
    _serpapi_available = False
    GoogleSearch = None  # type: ignore

# ---------------------- SESSION STATE ----------------------
def ss_init():
    defaults = {
        "uploaded_urls": [],
        "keyword": "",
        "location": "United States",
        "topn": 10,
        "serp_raw": {},
        "top_pages": [],
        "paa": [],
        "internal_links": [],
        "h2_sections": [],
        "h2_content": [],
        "faqs": [],
        "brief_table": None,
        "analysis_table": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ss_init()

# ---------------------- HELPERS ----------------------
def generate_openai_text(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 400, temperature: float = 0.7) -> str:
    if not openai_client:
        return "(OpenAI key missing in secrets)"
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

def fetch_serp(keyword: str, location: str) -> Dict[str, Any]:
    if not _serpapi_available:
        st.error("SerpAPI client not found. Add 'google-search-results' to requirements.txt. Keep import: 'from serpapi import GoogleSearch'.")
        return {}
    if not SERPAPI_KEY:
        st.error("Missing SERPAPI_KEY in st.secrets.")
        return {}
    try:
        search = GoogleSearch({"q": keyword, "location": location, "api_key": SERPAPI_KEY})
        return search.get_dict() or {}
    except Exception as e:
        st.error(f"SerpAPI error: {e}")
        return {}

def extract_top_pages(serp: Dict[str, Any], n: int = 10) -> List[str]:
    links: List[str] = []
    for res in serp.get("organic_results", [])[:n]:
        link = res.get("link")
        if link:
            links.append(link)
    return links

def extract_paa(serp: Dict[str, Any], n: int = 10) -> List[str]:
    items: List[str] = []
    for obj in serp.get("related_questions", [])[:n]:
        q = obj.get("question")
        if q:
            items.append(q)
    return items

def safe_get(url: str, timeout: int = 12) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SEO-Brief/1.0; +https://example.com/bot)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.ok and "text/html" in r.headers.get("Content-Type", ""):
            return r.text
    except Exception:
        pass
    return ""

def analyze_page(url: str) -> Tuple[str, str, List[str], int]:
    """
    Returns: (title, meta_description, h2_list, external_outlink_count)
    """
    html = safe_get(url)
    if not html:
        return "", "", [], 0

    soup = BeautifulSoup(html, "html.parser")

    # Title
    try:
        title = (soup.title.string or "").strip() if soup.title else ""
    except Exception:
        title = ""

    # Meta description
    meta_desc = ""
    try:
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            meta_desc = md["content"].strip()
        else:
            ogd = soup.find("meta", property="og:description")
            if ogd and ogd.get("content"):
                meta_desc = ogd["content"].strip()
    except Exception:
        meta_desc = ""

    # H2s (deduped)
    h2s_raw: List[str] = []
    try:
        for h2 in soup.find_all("h2"):
            txt = h2.get_text(" ", strip=True)
            if txt:
                h2s_raw.append(txt)
    except Exception:
        pass
    seen = set()
    h2s: List[str] = []
    for h in h2s_raw:
        if h not in seen:
            seen.add(h)
            h2s.append(h)

    # External outlinks
    outlinks = 0
    try:
        page_domain = urlparse(url).netloc
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("http"):
                try:
                    if urlparse(href).netloc != page_domain:
                        outlinks += 1
                except Exception:
                    continue
    except Exception:
        outlinks = 0

    return title, meta_desc, h2s, outlinks

def pick_internal_links(all_urls: List[str], keyword: str) -> List[str]:
    k = (keyword or "").lower().strip()
    if not k:
        return []
    picks = [u for u in all_urls if k in u.lower()]
    if not picks:
        picks = all_urls[:10]
    return picks[:20]

def make_ticker(progress, total_stages: int):
    step = 0
    def tick():
        nonlocal step
        step += 1
        progress.progress(step / total_stages)
    return tick

# ---------------------- SIDEBAR ----------------------
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

# ---------------------- TABS ----------------------
tab1, tab2 = st.tabs(["Generate SEO Brief", "Top Ranking Analysis"])

# ---------------------- TAB 1: BRIEF ----------------------
with tab1:
    st.header("SEO Content Brief")

    st.session_state.keyword = st.text_input("Enter Primary Keyword", value=st.session_state.keyword)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.session_state.location = st.text_input("Search Location (SerpAPI)", value=st.session_state.location)
    with col_b:
        st.session_state.topn = st.number_input("Top N results to analyze", min_value=1, max_value=20, value=int(st.session_state.topn), step=1)

    generate = st.button("Generate Brief", type="primary", use_container_width=True)

    if generate:
        if not st.session_state.keyword.strip():
            st.error("Please enter a keyword.")
        else:
            stages = [
                "Fetching SERP data",
                "Extracting top pages and PAA",
                "Computing internal links",
                "Drafting H2 sections",
                "Writing H2 content",
                "Answering FAQs",
                "Compiling brief",
                "Analyzing top pages",
            ]
            progress = st.progress(0)
            tick = make_ticker(progress, len(stages))
            i = 0

            # Stage 1: SERP
            st.info(stages[i])
            serp = fetch_serp(st.session_state.keyword, st.session_state.location)
            st.session_state.serp_raw = serp
            tick(); i += 1

            # Stage 2: extract top pages and PAA
            st.info(stages[i])
            top_pages = extract_top_pages(serp, n=int(st.session_state.topn))
            paa = extract_paa(serp, n=10)
            st.session_state.top_pages = top_pages
            st.session_state.paa = paa
            tick(); i += 1

            # Stage 3: internal links
            st.info(stages[i])
            internal_links = pick_internal_links(st.session_state.uploaded_urls, st.session_state.keyword)
            st.session_state.internal_links = internal_links
            tick(); i += 1

            # Stage 4: Draft H2 sections (LLM)
            st.info(stages[i])
            h2_sections_prompt = (
                "You are creating an SEO content outline.\n"
                f'Primary keyword: "{st.session_state.keyword}"\n\n'
                "Propose 6-8 highly relevant H2 sections that comprehensively cover the topic.\n"
                "Return them as a simple bullet list with no extra commentary."
            )
            h2_sections_text = generate_openai_text(h2_sections_prompt, max_tokens=300)
            h2_sections: List[str] = []
            for line in (h2_sections_text or "").splitlines():
                line = line.strip()
                if line.startswith(("-", "*", "•")):
                    line = line[1:].strip()
                if line:
                    h2_sections.append(line)
            if not h2_sections:
                h2_sections = [f"{st.session_state.keyword} - Section {n+1}" for n in range(6)]
            st.session_state.h2_sections = h2_sections[:8]
            tick(); i += 1

            # Stage 5: H2 content (LLM)
            st.info(stages[i])
            h2_content: List[str] = []
            for section in st.session_state.h2_sections:
                content_prompt = (
                    "Write approximately 120-150 words for the section: "
                    + section
                    + ". Be concise and helpful. Avoid fluff."
                )
                h2_content.append(generate_openai_text(content_prompt, max_tokens=220))
            st.session_state.h2_content = h2_content
            tick(); i += 1

            # Stage 6: FAQs (LLM)
            st.info(stages[i])
            faqs = []
            for q in (st.session_state.paa or [])[:6]:
                faq_ans = generate_openai_text(
                    "Provide a concise (2-3 sentences) answer to this FAQ for SEO content: " + q,
                    max_tokens=180,
                )
                faqs.append({"question": q, "answer": faq_ans})
            st.session_state.faqs = faqs
            tick(); i += 1

            # Stage 7: Compile brief table
            st.info(stages[i])
            est_outlinks = max(10, len(st.session_state.top_pages) * 5)
            url_slug = st.session_state.keyword.lower().strip().replace(" ", "-")

            rows = []
            rows.append(("Title", "Best " + st.session_state.keyword + ": Complete Guide & Top Picks"))
            rows.append(("Meta Description", "Learn everything about " + st.session_state.keyword + ". Clear sections, FAQs, and expert tips to help you decide."))
            rows.append(("H1", st.session_state.keyword + ": The Complete Guide"))
            rows.append(("Navigation Sidebar", "\n".join(st.session_state.h2_sections)))
            rows.append(("People Also Ask", "\n".join(st.session_state.paa)))
            rows.append(("Top Ranking Pages", "\n".join(st.session_state.top_pages)))
            rows.append(("URL Structure", "https://www.example.com/" + url_slug))
            rows.append(("Estimated Outlinks", est_outlinks))
            rows.append(("Internal Link Recommendations", "\n".join(st.session_state.internal_links)))
            rows.append(("Number of Page Sections", len(st.session_state.h2_sections)))

            for section, copy in zip(st.session_state.h2_sections, st.session_state.h2_content):
                rows.append((section, copy))

            faq_block = ""
            if st.session_state.faqs:
                faq_lines = []
                for f in st.session_state.faqs:
                    faq_lines.append("Q: " + f.get("question", ""))
                    faq_lines.append("A: " + f.get("answer", ""))
                faq_block = "\n".join(faq_lines)
            rows.append(("FAQs", faq_block))

            brief_df = pd.DataFrame(rows, columns=["Section", "Output"])
            st.session_state.brief_table = brief_df
            tick(); i += 1

            # Stage 8: Analyze top pages
            st.info(stages[i])
            analysis_rows = []
            for url in st.session_state.top_pages:
                title, meta_desc, h2s, outlinks = analyze_page(url)
                analysis_rows.append(
                    {
                        "URL": url,
                        "Title": title,
                        "Meta Description": meta_desc,
                        "H2 Sections (sample)": ", ".join(h2s[:8]),
                        "Outlinks (external)": outlinks,
                    }
                )
            if analysis_rows:
                st.session_state.analysis_table = pd.DataFrame(analysis_rows)
            else:
                st.session_state.analysis_table = pd.DataFrame(
                    columns=["URL", "Title", "Meta Description", "H2 Sections (sample)", "Outlinks (external)"]
                )

            progress.progress(1.0)
            st.success("SEO Brief Generated")

    # Render output
    if st.session_state.brief_table is not None:
        st.dataframe(st.session_state.brief_table, height=720, use_container_width=True)
        csv = st.session_state.brief_table.to_csv(index=False).encode("utf-8")
        st.download_button("Download Brief CSV", data=csv, file_name="seo_brief.csv", mime="text/csv", use_container_width=True)

# ---------------------- TAB 2: ANALYSIS ----------------------
with tab2:
    st.header("Top Ranking Articles Analysis")
    if st.session_state.analysis_table is not None and not st.session_state.analysis_table.empty:
        with st.expander("Filters"):
            domain_filter = st.text_input("Filter by domain (optional, e.g., 'nytimes.com')", value="")
        df = st.session_state.analysis_table.copy()
        if domain_filter.strip():
            keep = []
            for _, row in df.iterrows():
                ok = False
                try:
                    ok = urlparse(row["URL"]).netloc.endswith(domain_filter.strip())
                except Exception:
                    ok = False
                keep.append(ok)
            df = df[keep]
        st.dataframe(df, height=600, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Analysis CSV", data=csv, file_name="top_ranking_analysis.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No analysis yet. Generate a brief in the first tab.")
