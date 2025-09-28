import streamlit as st
import pandas as pd
import requests
from urllib.parse import urlparse
from io import BytesIO
from docx import Document
from openai import OpenAI
import numpy as np
from bs4 import BeautifulSoup

st.set_page_config(page_title="SEO Brief Generator", layout="wide")
st.title("SEO Blog Brief Generator")

# -------------------------------
# SECRETS
# -------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# -------------------------------
# INPUTS
# -------------------------------
keyword = st.text_input("Enter target keyword:")
uploaded_csv = st.file_uploader("Upload CSV of pages (column: 'url')", type=["csv"])
generate_btn = st.button("Generate SEO Brief")

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def fetch_top_serp_articles(keyword, api_key, num_results=2):
    """Fetch top N ranking articles using SerpAPI."""
    params = {
        "engine": "google",
        "q": keyword,
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "api_key": api_key
    }
    r = requests.get("https://serpapi.com/search", params=params)
    data = r.json()
    results = []
    for res in data.get("organic_results", [])[:num_results]:
        results.append({
            "url": res.get("link", ""),
            "title": res.get("title", ""),
            "meta": res.get("snippet", "")
        })
    return results

def extract_url_topics(url):
    """Use URL path segments as semantic topics."""
    path = urlparse(url).path
    segments = [seg for seg in path.split("/") if seg]
    topics = []
    for seg in segments:
        topics += seg.replace("-", " ").split()
    return " ".join(topics)

def generate_embeddings(texts):
    """Batch generate embeddings using OpenAI client."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def score_urls_semantic_batch(keyword, urls):
    """Batch semantic scoring for URLs."""
    url_topics = [extract_url_topics(u) for u in urls]
    embeddings = generate_embeddings([keyword] + url_topics)
    keyword_emb = embeddings[0]
    url_embs = embeddings[1:]
    return [(u, cosine_similarity(keyword_emb, emb)) for u, emb in zip(urls, url_embs)]

def extract_h2_sections(url):
    """Scrape H2 sections from a page."""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")
        h2s = [h.get_text().strip() for h in soup.find_all("h2")]
        return h2s
    except:
        return []

def count_outlinks(url):
    """Count internal/external links in body excluding nav/footer."""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")
        links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        # exclude obvious nav/footer based on common patterns
        links = [l for l in links if not any(x in l for x in ["#","javascript","mailto"])]
        return len(links)
    except:
        return 0

def generate_docx_table(brief_df):
    """Create a .docx file from the brief DataFrame."""
    doc = Document()
    doc.add_heading("SEO Content Brief", 0)
    table = doc.add_table(rows=1, cols=len(brief_df.columns))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(brief_df.columns):
        hdr_cells[i].text = col
    for _, row in brief_df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    f = BytesIO()
    doc.save(f)
    f.seek(0)
    return f

def fetch_people_also_ask(keyword):
    """Fetch PAA questions from SERPAPI."""
    try:
        params = {
            "engine": "google",
            "q": keyword,
            "location": "United States",
            "hl": "en",
            "gl": "us",
            "api_key": SERPAPI_KEY
        }
        r = requests.get("https://serpapi.com/search", params=params)
        data = r.json()
        questions = []
        for q in data.get("related_questions", []):
            questions.append(q.get("question", ""))
        return questions
    except:
        return []

# -------------------------------
# MAIN
# -------------------------------
if generate_btn:
    if not keyword:
        st.warning("Please enter a keyword.")
    else:
        progress = st.progress(0)
        status_text = st.empty()
        progress_val = 0

        # 1. Top 2 SERP articles
        status_text.text("Fetching top ranking pages...")
        top_pages = fetch_top_serp_articles(keyword, SERPAPI_KEY)
        progress_val += 15
        progress.progress(progress_val)

        # 2. CSV URLs semantic scoring
        status_text.text("Processing uploaded CSV for internal link recommendations...")
        if uploaded_csv:
            df_pages = pd.read_csv(uploaded_csv)
            urls = df_pages["url"].tolist()
            semantic_scores = score_urls_semantic_batch(keyword, urls)
            top_semantic_links = sorted(semantic_scores, key=lambda x: x[1], reverse=True)[:5]
        else:
            top_semantic_links = []
        progress_val += 20
        progress.progress(progress_val)

        # 3. Extract H2s, outlinks, URL structure
        status_text.text("Scraping top pages for H2s, outlinks, URL structure...")
        sections = []
        for i, page in enumerate(top_pages):
            h2s = extract_h2_sections(page["url"])
            num_outlinks = count_outlinks(page["url"])
            url_struct = f"www.example.com/{keyword.replace(' ','-')}"
            sections.append({
                "Section": f"Top Page {i+1} Title",
                "Output": page["title"],
                "How I expect": "SEO optimized title based on top ranking page"
            })
            sections.append({
                "Section": f"Top Page {i+1} Meta",
                "Output": page["meta"],
                "How I expect": "Unique meta description"
            })
            sections.append({
                "Section": f"Top Page {i+1} H2s",
                "Output": ", ".join(h2s[:5]),
                "How I expect": "List of H2 sections on page"
            })
            sections.append({
                "Section": f"Top Page {i+1} Est. Outlinks",
                "Output": num_outlinks,
                "How I expect": "Number of body links excluding nav/footer"
            })
            sections.append({
                "Section": f"Top Page {i+1} URL structure",
                "Output": url_struct,
                "How I expect": "Suggested URL for new content"
            })
        progress_val += 30
        progress.progress(progress_val)

        # 4. People also ask
        status_text.text("Fetching People Also Ask questions...")
        paa_questions = fetch_people_also_ask(keyword)
        sections.append({
            "Section": "People Also Ask",
            "Output": "\n".join(paa_questions),
            "How I expect": "List of relevant questions from PAA"
        })
        progress_val += 15
        progress.progress(progress_val)

        # 5. Internal link recommendations
        internal_links_text = "\n".join([f"{u} (score: {s:.2f})" for u, s in top_semantic_links])
        sections.append({
            "Section": "Internal link recommendations",
            "Output": internal_links_text,
            "How I expect": "Semantic links to include in article"
        })

        # 6. Consider writing topics
        sections.append({
            "Section": "Consider writing topics on",
            "Output": "Other linked entities or pages missing in current content cluster",
            "How I expect": "Suggest additional cluster content to cover"
        })

        progress_val = 100
        progress.progress(progress_val)
        status_text.text("SEO brief generation complete!")

        # Convert to DataFrame for table display
        brief_df = pd.DataFrame(sections)
        st.subheader("SEO Brief Table")
        st.dataframe(brief_df)

        # Download .docx
        doc_file = generate_docx_table(brief_df)
        st.download_button(
            "Download SEO Brief (.docx)",
            doc_file,
            file_name=f"{keyword}_seo_brief.docx"
        )
