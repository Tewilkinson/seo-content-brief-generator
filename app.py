import streamlit as st
import pandas as pd
import requests
from openai import OpenAI
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tiktoken
import time

# --- Streamlit Secrets ---
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

st.set_page_config(page_title="SEO Content Brief Generator", layout="wide")

# --- Helper functions ---
def fetch_top_pages(keyword, num=2):
    params = {
        "engine": "google",
        "q": keyword,
        "num": num,
        "api_key": SERPAPI_KEY,
        "gl": "us"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    pages = []
    for r in results.get("organic_results", []):
        pages.append(r["link"])
    return pages[:num]

def extract_page_metadata(url):
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    h1 = soup.find("h1")
    h2s = [h2.get_text(strip=True) for h2 in soup.find_all("h2")]
    outlinks = [a["href"] for a in soup.find_all("a", href=True) if urlparse(a["href"]).netloc]
    return {
        "meta_description": meta_desc["content"] if meta_desc else "",
        "h1": h1.get_text(strip=True) if h1 else "",
        "h2": h2s,
        "outlinks": outlinks
    }

def generate_ai_content(prompt, max_tokens=200):
    resp = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def semantic_internal_links(urls):
    # Simple example: extract unique keywords from URLs for internal linking
    links = set()
    for u in urls:
        path = urlparse(u).path.strip("/").split("/")
        for p in path:
            if p and p.lower() not in ["en", "us", "blog", "articles", "fundamentals"]:
                links.add(p.replace("-", " "))
    return list(links)

# --- Sidebar ---
st.sidebar.title("SEO Content Brief Generator")
uploaded_csv = st.sidebar.file_uploader("Upload CSV (keywords)", type=["csv"])

# --- Main ---
if uploaded_csv:
    df_keywords = pd.read_csv(uploaded_csv)
    if "keyword" not in df_keywords.columns:
        st.error("CSV must have a 'keyword' column")
    else:
        keywords = df_keywords["keyword"].tolist()
        st.title("SEO Content Briefs")
        
        progress = st.progress(0)
        all_briefs = []

        for i, keyword in enumerate(keywords):
            progress.progress(int((i / len(keywords)) * 100))
            st.subheader(f"Keyword: {keyword}")

            # Stage 1: Fetch Top Pages
            top_pages = fetch_top_pages(keyword)
            st.text(f"Top pages: {top_pages}")

            # Stage 2: Extract metadata
            metadata_list = []
            for page in top_pages:
                meta = extract_page_metadata(page)
                metadata_list.append(meta)
                time.sleep(0.5)
            
            # Stage 3: Generate AI brief
            brief = {
                "Title": generate_ai_content(f"Write SEO optimized title for: {keyword}"),
                "Meta description": generate_ai_content(f"Write unique meta description for: {keyword}"),
                "H1": generate_ai_content(f"Write H1 tag for: {keyword}"),
                "Navigation sidebar": "\n".join([h2 for meta in metadata_list for h2 in meta["h2"]]),
                "People also asks": generate_ai_content(f"Generate People Also Ask questions for: {keyword}"),
                "Top ranking pages": ", ".join(top_pages),
                "URL structure": f"https://www.snowflake.com/en/fundamentals/{keyword.replace(' ', '-')}",
                "Est. Number of outlinks": sum([len(meta["outlinks"]) for meta in metadata_list]),
                "Internal link recommendations": ", ".join(semantic_internal_links(top_pages)),
                "Number of page sections": max([len(meta["h2"]) for meta in metadata_list]) + 1,
            }

            # H2 Sections + What to write
            for j in range(brief["Number of page sections"]):
                h2_key = f"H2 section {j+1}"
                prompt = f"Generate a recommended H2 section for {keyword}, provide 50-100 words of content."
                brief[h2_key] = generate_ai_content(prompt)

            # FAQs
            brief["FAQs"] = generate_ai_content(f"Generate 3 FAQs with answers for {keyword}")

            # Suggested cluster topics
            brief["Consider writing topics on"] = generate_ai_content(
                f"Analyze the top ranking pages for {keyword} and suggest related content topics"
            )

            all_briefs.append(brief)

        progress.progress(100)

        # Display briefs in table
        st.subheader("Generated SEO Briefs")
        df_briefs = pd.DataFrame(all_briefs)
        st.data_editor(df_briefs, height=600, use_container_width=True)

        # Analysis tab
        st.subheader("Top Ranking Articles Analysis")
        for keyword, brief in zip(keywords, all_briefs):
            st.markdown(f"### Keyword: {keyword}")
            st.text(f"Top Pages: {brief['Top ranking pages']}")
            st.text(f"H1: {brief['H1']}")
            st.text(f"Number of H2 Sections: {brief['Number of page sections']}")
            st.text(f"Estimated Outlinks: {brief['Est. Number of outlinks']}")
            st.text(f"Internal link recommendations: {brief['Internal link recommendations']}")
