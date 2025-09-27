# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import numpy as np
from openai import OpenAI

# -------------------------
# Setup OpenAI Client
# -------------------------
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
serpapi_key = st.secrets["SERPAPI_KEY"]

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
    except Exception as e:
        st.warning(f"Error scraping {url}: {e}")
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
    except Exception as e:
        st.warning(f"Error parsing sitemap: {e}")
        return []

def semantic_related_links(keyword, urls):
    try:
        kw_emb = openai_client.embeddings.create(model="text-embedding-3-small", input=keyword).data[0].embedding
        scored_urls = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=5)
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string if soup.title else url
                emb = openai_client.embeddings.create(model="text-embedding-3-small", input=title).data[0].embedding
                score = np.dot(kw_emb, emb) / (np.linalg.norm(kw_emb) * np.linalg.norm(emb))
                scored_urls.append((url, score))
            except:
                continue
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        return [u[0] for u in scored_urls[:5]]
    except Exception as e:
        st.warning(f"Error generating semantic links: {e}")
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
        st.info("Fetching SERP results, People Also Ask, and semantic internal links...")
        urls = fetch_serp_urls(keyword)
        paa_questions = fetch_paa(keyword)
        sitemap_links = parse_sitemap(sitemap_url) if sitemap_url else []
        brief = generate_brief(keyword, urls, paa_questions, sitemap_links)

        # Title
        st.subheader("Suggested Title")
        st.text_input("Title", value=brief["title"], key="title")
        st.caption(f"Why: {brief['title_why']}")

        # Meta
        st.subheader("Meta Description")
        st.text_area("Meta", value=brief["meta"], key="meta")
        st.caption(f"Why: {brief['meta_why']}")

        # Sections
        st.subheader("Sections / Headings")
        for i, s in enumerate(brief["sections"]):
            st.markdown(f"**{s['heading']}**")
            st.text_area("What to write:", value=s["what_to_write"], key=f"write_{i}")
            st.caption(f"Why: {s['why']}")

        # PAA
        st.subheader("People Also Ask")
        for i, f in enumerate(brief["faqs"]):
            st.text_area("Question:", value=f["question"], key=f"q_{i}")
            st.text_area("Suggested Answer:", value=f["suggested_content"], key=f"ans_{i}")
            st.caption(f"Why: {f['why']}")

        # Internal links
        if brief["internal_links"]:
            st.subheader("Suggested Internal Links")
            for i, link in enumerate(brief["internal_links"]):
                st.markdown(f"- [{link}]({link})", key=f"link_{i}")
            st.caption("These links are semantically related to the target keyword and strengthen topical authority.")

        st.success("Brief generation complete! You can now edit and export to Google Docs or .docx.")
