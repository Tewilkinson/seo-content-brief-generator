import streamlit as st
from bs4 import BeautifulSoup
from googlesearch_results import GoogleSearch
import requests
import openai
import numpy as np
from docx import Document
from io import BytesIO
import streamlit.components.v1 as components

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="SEO Brief Generator", layout="wide")
st.title("SEO Blog Brief Generator")

# --------------------------
# SECRETS
# --------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# --------------------------
# INPUTS
# --------------------------
keyword = st.text_input("Enter target keyword:")
uploaded_pages = st.file_uploader(
    "Upload pages for semantic analysis (txt/html/pdf)", 
    type=["txt","html","pdf"], 
    accept_multiple_files=True
)
generate_btn = st.button("Generate SEO Brief")

# --------------------------
# UTILITY FUNCTIONS
# --------------------------
def fetch_top_serp_article(keyword, api_key):
    search = GoogleSearch({
        "q": keyword,
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "api_key": api_key
    })
    results = search.get_dict()
    top_url = results.get("organic_results", [{}])[0].get("link", "")
    return top_url

def scrape_article(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        headings = [h.get_text() for h in soup.find_all(["h1","h2","h3"])]
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = "\n".join(paragraphs)
        title = soup.title.string if soup.title else ""
        meta = soup.find("meta", attrs={"name":"description"})
        meta_desc = meta["content"] if meta else ""
        return {"url": url, "title": title, "meta": meta_desc, "headings": headings, "text": text}
    except Exception as e:
        st.warning(f"Failed to scrape {url}: {e}")
        return {"url": url, "title": "", "meta": "", "headings": [], "text": ""}

def parse_uploaded_file(f):
    if f.type == "text/plain":
        return f.read().decode()
    elif f.type == "text/html":
        soup = BeautifulSoup(f.read(), "html.parser")
        return " ".join([p.get_text() for p in soup.find_all("p")])
    elif f.type == "application/pdf":
        import fitz  # PyMuPDF
        pdf = fitz.open(stream=f.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    else:
        return ""

def generate_embeddings(text_list):
    response = openai.Embedding.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    return [d["embedding"] for d in response["data"]]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_docx(brief):
    doc = Document()
    doc.add_heading(brief["title"], 0)
    doc.add_paragraph(brief["meta"])
    for s in brief["sections"]:
        doc.add_heading(s["heading"], level=2)
        doc.add_paragraph(s["what_to_write"])
        doc.add_paragraph(f"Why: {s['why']}")
    if brief["faqs"]:
        doc.add_heading("FAQs", level=2)
        for f in brief["faqs"]:
            doc.add_paragraph(f"Q: {f['question']}")
            doc.add_paragraph(f"A: {f['suggested_content']}")
            doc.add_paragraph(f"Why: {f['why']}")
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
        top_url = fetch_top_serp_article(keyword, SERPAPI_KEY)
        top_article = scrape_article(top_url)
        progress_val += 20
        progress.progress(progress_val)

        # 2. Parse uploaded pages for semantic links
        status_text.text("Processing uploaded pages for semantic relevance...")
        pages_texts = [parse_uploaded_file(f) for f in uploaded_pages]
        pages_names = [f.name for f in uploaded_pages]

        if pages_texts:
            all_embeddings = generate_embeddings(pages_texts + [keyword])
            keyword_emb = all_embeddings[-1]
            page_embs = all_embeddings[:-1]
            semantic_scores = []
            for i, emb in enumerate(page_embs):
                score = cosine_similarity(keyword_emb, emb)
                semantic_scores.append((pages_names[i], score))
            top_semantic = sorted(semantic_scores, key=lambda x: x[1], reverse=True)[:5]
        else:
            top_semantic = []
        progress_val += 30
        progress.progress(progress_val)

        # 3. Generate sections for brief
        status_text.text("Generating brief sections...")
        sections = []
        for idx, h in enumerate(top_article["headings"]):
            sections.append({
                "heading": h,
                "what_to_write": f"Explain this topic in 150-200 words. Pull insights from {top_url}.",
                "why": "Covers key topic points and semantic relevance."
            })
        progress_val += 30
        progress.progress(progress_val)

        # 4. PAA placeholder
        status_text.text("Fetching People Also Ask...")
        paa_questions = []  # To integrate SERPAPI PAA later
        faqs = [{"question": q, "suggested_content": f"Write 50-100 words for '{q}'", "why": "Covers search intent"} for q in paa_questions]
        progress_val += 10
        progress.progress(progress_val)

        # 5. Compile brief
        brief = {
            "title": top_article["title"] or f"Suggested: {keyword}",
            "meta": top_article["meta"] or f"Meta for {keyword}",
            "sections": sections,
            "faqs": faqs,
            "internal_links": top_semantic
        }
        progress_val = 100
        progress.progress(progress_val)
        status_text.text("Brief generation complete!")

        # 6. Display brief preview
        st.subheader("SEO Brief Preview")
        st.markdown(f"**Title:** {brief['title']}")
        st.markdown(f"**Meta Description:** {brief['meta']}")

        st.subheader("Sections / Headings")
        for idx, s in enumerate(brief["sections"]):
            st.markdown(f"**{s['heading']}**")
            st.text_area(f"What to write (section {idx+1})", value=s["what_to_write"], key=f"section_{idx}")
            st.caption(f"Why: {s['why']}")

        if brief["faqs"]:
            st.subheader("FAQs")
            for idx, f in enumerate(brief["faqs"]):
                st.text_area(f"FAQ: {f['question']} (FAQ {idx+1})", value=f["suggested_content"], key=f"faq_{idx}")
                st.caption(f"Why: {f['why']}")

        if brief["internal_links"]:
            st.subheader("Recommended Internal Links")
            for name, score in brief["internal_links"]:
                st.markdown(f"- {name} (Score: {score:.2f})")

        # 7. Download .docx
        doc_file = create_docx(brief)
        st.download_button(
            "Download SEO Brief (.docx)", 
            doc_file, 
            file_name=f"{keyword}_seo_brief.docx"
        )

        # 8. Google Docs iframe preview
        st.subheader("Preview SEO Brief in Google Docs")
        gdoc_url = st.text_input("Enter shareable Google Docs link (view only):", value="")
        if gdoc_url:
            if "edit" in gdoc_url:
                gdoc_url = gdoc_url.replace("/edit", "/preview")
            components.html(f'<iframe src="{gdoc_url}" width="100%" height="600px" style="border:none;"></iframe>', height=600)
