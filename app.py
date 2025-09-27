# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import openai
from urllib.parse import urlparse
import textwrap

# ---------------------------
# Helper functions
# ---------------------------

def fetch_sitemap_urls(sitemap_url):
    """Parse sitemap and return a list of URLs"""
    try:
        resp = requests.get(sitemap_url)
        tree = ET.fromstring(resp.content)
        urls = [elem.text for elem in tree.findall('.//{*}loc')]
        return urls
    except Exception as e:
        st.error(f"Error fetching sitemap: {e}")
        return []

def fetch_page_content(url):
    """Fetch page HTML and extract text + headings + links"""
    try:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        headings = [h.get_text().strip() for h in soup.find_all(['h1','h2','h3'])]
        paragraphs = " ".join([p.get_text().strip() for p in soup.find_all('p')])
        links = [a['href'] for a in soup.find_all('a', href=True)]
        title = soup.title.string if soup.title else ""
        return {
            'title': title,
            'headings': headings,
            'text': paragraphs,
            'links': links
        }
    except Exception as e:
        st.warning(f"Error fetching page {url}: {e}")
        return {}

def extract_paa_questions(keyword):
    """Mock function: replace with SERP API for People Also Ask"""
    # Example static for demo
    return [
        f"What shoes are best for {keyword}?",
        f"How to prevent injuries with {keyword}?",
        f"Are expensive shoes better for {keyword}?"
    ]

def chunk_text(text, max_words=200):
    """Split text into ≤ max_words chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def generate_brief_section(keyword, sitemap_urls, top_pages_data):
    """Generate a mock brief section"""
    brief = {}

    # Suggested title
    brief['title'] = f"Best {keyword.title()} – Top Picks & Tips"
    brief['title_why'] = f"Including the keyword improves SEO relevance and CTR. Top competitor pages use similar titles."

    # Meta description
    brief['meta'] = f"Discover the best {keyword}. Expert recommendations, features, and tips to improve comfort and performance."
    brief['meta_why'] = "Meta description increases CTR and ensures Google understands page topic."

    # Headings
    brief['headings'] = [
        {"heading": f"H1: Best {keyword}", "what": "Introduce topic and benefits", "why": "Main topic, improves relevance"},
        {"heading": "H2: Top Features to Look For", "what": "Explain key features, cushioning, support", "why": "Readers look for guidance; semantic coverage"},
        {"heading": "H2: Top Recommended Products", "what": "List products with pros/cons", "why": "Actionable recommendations; matches competitor depth"},
        {"heading": "H2: How to Improve Performance", "what": "Tips on technique, insoles, stretching", "why": "Covers semantic subtopics; adds value"},
        {"heading": "H3: FAQs", "what": "Answer People Also Ask questions", "why": "Targets additional search intent"}
    ]

    # Key keywords/entities
    brief['keywords'] = ['running shoes', 'flat feet', 'stability shoes', 'cushioning', 'arch support', 'insoles', 'foot health']
    brief['keywords_why'] = "Ensures semantic coverage and relevance for SEO."

    # Internal link suggestions (mock top 2 sitemap URLs)
    brief['internal_links'] = []
    for url in sitemap_urls[:2]:
        brief['internal_links'].append({
            "url": url,
            "anchor": urlparse(url).path.strip('/').replace('-', ' ').title(),
            "why": "Linked by top pages for relevance; strengthens topical authority."
        })

    # People Also Ask
    paa_questions = extract_paa_questions(keyword)
    brief['faqs'] = []
    for q in paa_questions:
        brief['faqs'].append({
            "question": q,
            "suggested_content": f"Write 50–80 words answering: {q}",
            "why": "Targets search intent and potential rich snippets."
        })

    # Content chunks
    all_text = " ".join([d.get('text','') for d in top_pages_data])
    brief['chunks'] = []
    for chunk in chunk_text(all_text, 200):
        brief['chunks'].append({
            "content": chunk,
            "why": "Provides context, matches competitor depth, and is manageable for writing."
        })

    return brief

# ---------------------------
# Streamlit App
# ---------------------------

st.title("SEO Blog Creation Tool")

# Sidebar inputs
keyword = st.text_input("Enter Target Keyword")
sitemap_url = st.text_input("Enter Sitemap URL (optional)")

if st.button("Generate Brief"):
    if not keyword:
        st.warning("Please enter a keyword.")
    else:
        st.info("Fetching sitemap and top pages...")

        # 1. Sitemap URLs
        sitemap_urls = fetch_sitemap_urls(sitemap_url) if sitemap_url else []

        # 2. Mock top pages URLs (replace with SERP API)
        top_pages_urls = [
            f"https://competitor1.com/{keyword.replace(' ','-')}",
            f"https://competitor2.com/{keyword.replace(' ','-')}"
        ]

        top_pages_data = [fetch_page_content(u) for u in top_pages_urls]

        # 3. Generate brief
        brief = generate_brief_section(keyword, sitemap_urls, top_pages_data)

        # ---------------------------
        # Display Brief in Streamlit
        # ---------------------------

        st.subheader("Suggested Title")
        st.text_input("Title:", value=brief['title'])
        st.caption(f"Why: {brief['title_why']}")

        st.subheader("Meta Description")
        st.text_area("Meta:", value=brief['meta'])
        st.caption(f"Why: {brief['meta_why']}")

        st.subheader("Headings / Outline")
        for h in brief['headings']:
            st.markdown(f"**{h['heading']}**")
            st.text_area("What to write:", value=h['what'])
            st.caption(f"Why: {h['why']}")

        st.subheader("Key Keywords / Entities")
        st.text_area("Keywords (comma-separated):", value=", ".join(brief['keywords']))
        st.caption(f"Why: {brief['keywords_why']}")

        st.subheader("Recommended Internal Links")
        for l in brief['internal_links']:
            st.text_input("URL:", value=l['url'])
            st.text_input("Anchor Text:", value=l['anchor'])
            st.caption(f"Why: {l['why']}")

        st.subheader("Recommended FAQs")
        for f in brief['faqs']:
            st.text_area("Question:", value=f['question'])
            st.text_area("Suggested Content:", value=f['suggested_content'])
            st.caption(f"Why: {f['why']}")

        st.subheader("Content Chunks (≤200 words each)")
        for i, c in enumerate(brief['chunks']):
            st.text_area(f"Chunk {i+1}:", value=c['content'])
            st.caption(f"Why: {c['why']}")

        st.success("Brief generation complete! You can now edit and export to Google Docs.")

# ---------------------------
# TODO / Next Steps:
# ---------------------------
# - Replace mock top pages with actual SERP scraping / SerpAPI
# - Use OpenAI embeddings for semantic internal link recommendations
# - Fetch real PAA questions from SERP
# - Export to Google Docs via API
# - Optional: color-coded highlights for writers
