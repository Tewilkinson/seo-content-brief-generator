import streamlit as st
import requests
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from io import BytesIO
from docx import Document
import streamlit.components.v1 as components
from openai import OpenAI

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="SEO Brief Generator", layout="wide")
st.title("SEO Blog Brief Generator")

# --------------------------
# SECRETS
# --------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# --------------------------
# INPUTS
# --------------------------
keyword = st.text_input("Enter target keyword:")
uploaded_csv = st.file_uploader("Upload CSV of pages (column: 'url')", type=["csv"])
generate_btn = st.button("Generate SEO Brief")

# --------------------------
# UTILITY FUNCTIONS
# --------------------------
def fetch_top_serp_article(keyword, api_key):
    """Fetch top-ranking US SERP article URL using SerpAPI."""
    params = {
        "engine": "google",
        "q": keyword,
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "api_key": api_key
    }
    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = r.json()
        top_url = data.get("organic_results", [{}])[0].get("link", "")
        top_title = data.get("organic_results", [{}])[0].get("title", "")
        top_snippet = data.get("organic_results", [{}])[0].get("snippet", "")
        return {"url": top_url, "title": top_title, "meta": top_snippet}
    except Exception as e:
        st.warning(f"Failed to fetch top SERP article: {e}")
        return {"url": "", "title": f"Suggested: {keyword}", "meta": ""}

def extract_topic_from_url(url):
    """Extract words from URL path segments as semantic topics."""
    path = urlparse(url).path
    segments = [seg for seg in path.split('/') if seg]
    topics = []
    for seg in segments:
        topics += seg.replace('-', ' ').split()
    return " ".join(topics)

def generate_embeddings(text_list):
    """Generate embeddings using the new OpenAI client."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def score_urls_semantic_batch(keyword, urls):
    """Batch process URLs for semantic similarity."""
    # Prepare URL topics
    url_texts = [extract_topic_from_url(u) for u in urls]
    embeddings = generate_embeddings([keyword] + url_texts)
    keyword_emb = embeddings[0]
    url_embs = embeddings[1:]
    # Compute similarities
    return [(u, cosine_similarity(keyword_emb, emb)) for u, emb in zip(urls, url_embs)]

def create_docx(brief):
    """Generate .docx file from brief data."""
    doc = Document()
    doc.add_heading(brief["title"], 0)
    doc.add_paragraph(brief["meta"])
    for s in brief["sections"]:
        doc.add_heading(s["heading"], level=2)
        doc.add_paragraph(s["what_to_write"])
        doc.add_paragraph(f"Why: {s['why']}")
    if brief["internal_links"]:
        doc.add_heading("Internal Links", level=2)
        for link in brief["internal_links"]:
            doc.add_paragraph(f"{link[0]} (Score: {link[1]:.2f})")
    f = BytesIO()
    doc.save(f)
    f.seek(0)
    return f

# --------------------------
# MAIN PROCESS
# --------------------------
if generate_btn:
    if not keyword:
        st.warning("Please enter a keyword.")
    else:
        progress = st.progress(0)
        status_text = st.empty()
        progress_val = 0

        # 1. Fetch top SERP article
        status_text.text("Fetching top-ranking article...")
        top_article = fetch_top_serp_article(keyword, SERPAPI_KEY)
        progress_val += 20
        progress.progress(progress_val)

        # 2. Process uploaded CSV pages in batch
        status_text.text("Processing uploaded CSV pages for semantic relevance...")
        if uploaded_csv:
            pages_df = pd.read_csv(uploaded_csv)
            if "url" not in pages_df.columns:
                st.error("CSV must have a 'url' column.")
                st.stop()
            urls = pages_df["url"].tolist()
            semantic_scores = score_urls_semantic_batch(keyword, urls)
            top_semantic = sorted(semantic_scores, key=lambda x: x[1], reverse=True)[:5]
        else:
            top_semantic = []
        progress_val += 30
        progress.progress(progress_val)

        # 3. Generate sections for brief (placeholder)
        status_text.text("Generating brief sections...")
        sections = [
            {
                "heading": f"Introduction to {keyword}",
                "what_to_write": f"Explain the concept of {keyword} in 150-200 words. Reference top-ranking article: {top_article['url']}.",
                "why": "Provides overview and establishes topic relevance."
            },
            {
                "heading": f"Key Concepts of {keyword}",
                "what_to_write": f"Discuss main concepts related to {keyword}. Include semantic links where relevant.",
                "why": "Ensures thorough coverage of the topic."
            }
        ]
        progress_val += 30
        progress.progress(progress_val)

        # 4. Compile brief
        brief = {
            "title": top_article["title"] or f"Suggested: {keyword}",
            "meta": top_article["meta"] or f"Meta for {keyword}",
            "sections": sections,
            "internal_links": top_semantic
        }
        progress_val = 100
        progress.progress(progress_val)
        status_text.text("Brief generation complete!")

        # 5. Display brief preview
        st.subheader("SEO Brief Preview")
        st.markdown(f"**Title:** {brief['title']}")
        st.markdown(f"**Meta Description:** {brief['meta']}")

        st.subheader("Sections / Headings")
        for idx, s in enumerate(brief["sections"]):
            st.markdown(f"**{s['heading']}**")
            st.text_area(f"What to write (section {idx+1})", value=s["what_to_write"], key=f"section_{idx}")
            st.caption(f"Why: {s['why']}")

        if brief["internal_links"]:
            st.subheader("Recommended Internal Links")
            for name, score in brief["internal_links"]:
                st.markdown(f"- {name} (Score: {score:.2f})")

        # 6. Download .docx
        doc_file = create_docx(brief)
        st.download_button(
            "Download SEO Brief (.docx)",
            doc_file,
            file_name=f"{keyword}_seo_brief.docx"
        )

        # 7. Google Docs iframe preview
        st.subheader("Preview SEO Brief in Google Docs")
        gdoc_url = st.text_input("Enter shareable Google Docs link (view only):", value="")
        if gdoc_url:
            if "edit" in gdoc_url:
                gdoc_url = gdoc_url.replace("/edit", "/preview")
            components.html(f'<iframe src="{gdoc_url}" width="100%" height="600px" style="border:none;"></iframe>', height=600)
