# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from urllib.parse import urlparse
import textwrap

# Optional: SerpAPI client
from serpapi import GoogleSearch

# -------------------------
# Helper Functions
# -------------------------

def fetch_serp_urls(keyword, api_key, num_results=3):
    """Fetch top SERP URLs using SerpAPI"""
    params = {
        "engine": "google",
        "q": keyword,
        "api_key": api_key,
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    urls = [r["link"] for r in results.get("organic_results", [])[:num_results]]
    return urls

def fetch_paa(keyword, api_key):
    """Fetch People Also Ask using SerpAPI"""
    params = {
        "engine": "google",
        "q": keyword,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    paa = results.get("related_questions", [])
    questions = [q["question"] for q in paa]
    return questions[:5]

def scrape_article(url):
    """Scrape headings and paragraphs from a URL"""
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
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def generate_brief(keyword, urls, paa_questions):
    """Generate a full SEO brief structure"""
    brief = {}

    # Suggest title aligned with informational intent
    brief["title"] = f"{keyword.title()}: Everything You Need to Know"
    brief["title_why"] = "Matches informational intent; clearly communicates topic to readers."

    # Meta description
    brief["meta"] = f"Learn what {keyword} is, how it works, and why it matters in modern data management."
    brief["meta_why"] = "SEO meta description that explains content and improves CTR."

    # Article sections
    brief["sections"] = []
    for url in urls:
        article = scrape_article(url)
        for h in article["headings"]:
            brief["sections"].append({
                "heading": h,
                "what_to_write": f"Explain this topic in 150-200 words. Pull insights from {url}.",
                "why": "Covers key topic points and semantic relevance."
            })

    # People Also Ask
    brief["faqs"] = []
    for q in paa_questions:
        brief["faqs"].append({
            "question": q,
            "suggested_content": f"Write a 50-100 word answer to '{q}'",
            "why": "Addresses additional search intent and potential rich snippets."
        })

    return brief

# -------------------------
# Streamlit UI
# -------------------------
st.title("SEO Blog Brief Generator")

keyword = st.text_input("Enter target keyword (informational intent recommended)")
serpapi_key = st.text_input("Enter SerpAPI Key", type="password")

if st.button("Generate Brief"):
    if not keyword or not serpapi_key:
        st.warning("Please provide both a keyword and SerpAPI key.")
    else:
        st.info("Fetching SERP results and People Also Ask...")
        urls = fetch_serp_urls(keyword, serpapi_key)
        paa_questions = fetch_paa(keyword, serpapi_key)
        brief = generate_brief(keyword, urls, paa_questions)

        # Display title
        st.subheader("Suggested Title")
        st.text_input("Title", value=brief["title"])
        st.caption(f"Why: {brief['title_why']}")

        # Display meta
        st.subheader("Meta Description")
        st.text_area("Meta", value=brief["meta"])
        st.caption(f"Why: {brief['meta_why']}")

        # Display sections
        st.subheader("Sections / Headings")
        for s in brief["sections"]:
            st.markdown(f"**{s['heading']}**")
            st.text_area("What to write:", value=s["what_to_write"])
            st.caption(f"Why: {s['why']}")

        # Display PAA
        st.subheader("People Also Ask")
        for f in brief["faqs"]:
            st.text_area("Question:", value=f["question"])
            st.text_area("Suggested Answer:", value=f["suggested_content"])
            st.caption(f"Why: {f['why']}")

        st.success("Brief generation complete! You can now edit and export to Google Docs or .docx.")
