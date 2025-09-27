# app.py
import streamlit as st
import streamlit.components.v1 as components
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import numpy as np
from openai import OpenAI
from docx import Document
from io import BytesIO

# -------------------------
# Setup OpenAI Client
# -------------------------
try:
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API key missing. Add it to Streamlit Secrets as OPENAI_API_KEY.")
    st.stop()

try:
    serpapi_key = st.secrets["SERPAPI_KEY"]
except KeyError:
    st.error("SerpAPI key missing. Add it to Streamlit Secrets as SERPAPI_KEY.")
    st.stop()

# -------------------------
# Helper Functions
# -------------------------

def fetch_serp_urls(keyword, num_results=3):
    params = {"engine": "google", "q": keyword, "api_key": serpapi_key, "num": num_results}
    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
    data = resp.json()
    urls = [r["link"] for r in data.get("organic_results", [])[:num_results]]
    return urls

def fetch_paa(keyword):
    params = {"engine": "google", "q": keyword, "api_key": serpapi_key}
    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
    data = resp.json()
    paa = data.get("related_questions", [])
    return [q["question"] for q in paa][:5]

def scrape_article(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        headings = [h.get_text().strip() for h in soup.find_all(['h1','h2','h3'])]
        paragraphs = " ".join([p.get_text().strip() for p in soup.find_all('p')])
        links = [a['href'] for a in soup.find_all('a', href=True)]
        return {"headings": headings, "paragraphs": paragraphs, "links": links}
    except:
        return {"headings": [], "paragraphs": "", "links": []}

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def parse_sitemap(sitemap_url):
    try:
        resp = requests.get(sitemap_url, timeout=10)
        urls = []
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                loc = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is not None:
                    urls.append(loc.text)
        return urls
    except:
        return []

def semantic_related_links(keyword, urls):
    try:
        kw_emb = openai_client.embeddings.create(
            model="text-embedding-3-small", input=keyword
        ).data[0].embedding
        scored_urls = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=5)
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string if soup.title else url
                emb = openai_client.embeddings.create(
                    model="text-embedding-3-small", input=title
                ).data[0].embedding
                score = np.dot(kw_emb, emb) / (np.linalg.norm(kw_emb) * np.linalg.norm(emb))
                scored_urls.append((url, score))
            except:
                continue
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        return [u[0] for u in scored_urls[:5]]
    except:
        return []

def generate_brief(keyword, urls, paa_questions, sitemap_links):
    brief = {}
    brief["title"] = f"{keyword.title()}: Everything You Need to Know"
    brief["title_why"] = "Matches informational intent; clearly communicates topic to readers."
    brief["meta"] = f"Learn what {keyword} is, how it works, and why it matters in modern data management."
    brief["meta_why"] = "SEO meta description that explains content and improves CTR."

    brief["sections"] = []
    for url in urls:
        article = scrape_article(url)
        for h in article["headings"]:
            brief["sections"].append({
                "heading": h,
                "what_to_write": f"Explain this topic in 150-200 words. Pull insights from {url}.",
                "why": "Covers key topic points and semantic relevance."
            })

    brief["faqs"] = []
    for q in paa_questions:
        brief["faqs"].append({
            "question": q,
            "suggested_content": f"Write a 50-100 word answer to '{q}'",
            "why": "Addresses additional search intent and potential rich snippets."
        })

    brief["internal_links"] = semantic_related_links(keyword, sitemap_links)
    return brief

def create_docx(brief):
    doc = Document()
    doc.add_heading(brief["title"], level=0)
    doc.add_paragraph(f"Meta: {brief['meta']} ({brief['meta_why']})")
    doc.add_paragraph("\nSections:\n")
    for s in brief["sections"]:
        doc.add_heading(s["heading"], level=1)
        doc.add_paragraph(f"{s['what_to_write']}\nWhy: {s['why']}")
    doc.add_paragraph("\nPeople Also Ask:\n")
    for f in brief["faqs"]:
        doc.add_heading(f["question"], level=2)
        doc.add_paragraph(f"{f['suggested_content']}\nWhy: {f['why']}")
    doc.add_paragraph("\nSuggested Internal Links:\n")
    for l in brief["internal_links"]:
        doc.add_paragraph(l)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# -------------------------
# Streamlit UI
# -------------------------

st.title("SEO Blog Brief Generator")

keyword = st.text_input("Enter target keyword (informational intent recommended)")
sitemap_url = st.text_input("Enter your sitemap URL (optional)")

if st.button("Generate Brief"):
    if not keyword:
        st.warning("Please provide a keyword.")
    else:
        progress = st.progress(0)
        with st.spinner("Generating brief..."):
            # Step 1: Fetch SERP URLs
            progress.caption("Fetching top SERP results...")
            urls = fetch_serp_urls(keyword)
            progress.progress(20)

            # Step 2: Fetch People Also Ask
            progress.caption("Fetching People Also Ask questions...")
            paa_questions = fetch_paa(keyword)
            progress.progress(40)

            # Step 3: Parse sitemap and generate semantic links
            progress.caption("Parsing sitemap and generating semantic links...")
            if sitemap_url:
                sitemap_links = parse_sitemap(sitemap_url)
                semantic_links = semantic_related_links(keyword, sitemap_links)
            else:
                sitemap_links = []
                semantic_links = []
            progress.progress(70)

            # Step 4: Generate brief
            progress.caption("Compiling brief...")
            brief = generate_brief(keyword, urls, paa_questions, sitemap_links)
            brief["internal_links"] = semantic_links
            doc_file = create_docx(brief)
            progress.progress(100)

        # -------------------------
        # Render Brief
        # -------------------------
        st.subheader("Suggested Title")
        st.text_input("Title", value=brief["title"], key="title_input")
        st.caption(f"Why: {brief['title_why']}")

        st.subheader("Meta Description")
        st.text_area("Meta", value=brief["meta"], key="meta_input")
        st.caption(f"Why: {brief['meta_why']}")

        st.subheader("Sections / Headings")
        for i, s in enumerate(brief["sections"]):
            st.markdown(f"**{s['heading']}**")
            st.text_area("What to write:", value=s["what_to_write"], key=f"section_write_{i}")
            st.caption(f"Why: {s['why']}")

        st.subheader("People Also Ask")
        for i, f in enumerate(brief["faqs"]):
            st.text_area("Question:", value=f["question"], key=f"paa_q_{i}")
            st.text_area("Suggested Answer:", value=f["suggested_content"], key=f"paa_ans_{i}")
            st.caption(f"Why: {f['why']}")

        if brief["internal_links"]:
            st.subheader("Suggested Internal Links")
            for i, link in enumerate(brief["internal_links"]):
                st.text_input("Internal link:", value=link, key=f"internal_link_{i}")
            st.caption("These links are semantically related to the target keyword and strengthen topical authority.")

        st.subheader("Preview & Download Brief")
        st.download_button("Download Brief (.docx)", doc_file, file_name="seo_brief.docx")
        iframe_url = st.text_input("Google Docs shareable URL for iframe preview", "")
        if iframe_url:
            components.iframe(iframe_url.replace("/edit","/preview"), height=600, scrolling=True)

        st.success("Brief generation complete!")
