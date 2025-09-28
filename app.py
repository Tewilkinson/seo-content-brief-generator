import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import tiktoken

# Initialize OpenAI client
openai_client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
SERPAPI_KEY = st.secrets['SERPAPI_KEY']

st.set_page_config(page_title="SEO Content Brief Generator", layout="wide")

# Tabs
tab1, tab2 = st.tabs(["Generate SEO Brief", "Top Ranking Analysis"])

# ---------------------- TAB 1: SEO BRIEF ----------------------
with tab1:
    st.header("SEO Content Brief Generator")

    uploaded_file = st.file_uploader("Upload a CSV of URLs", type=['csv'])
    urls_list = []
    if uploaded_file:
        df_urls = pd.read_csv(uploaded_file)
        urls_list = df_urls.iloc[:,0].dropna().tolist()
        st.success(f"{len(urls_list)} URLs loaded.")

    keyword = st.text_input("Enter Primary Keyword")

    if st.button("Generate Brief"):
        progress = st.progress(0)
        stages = ["Fetching SERP data", "Extracting top pages", "Generating semantic internal links",
                  "Generating H2 sections", "Generating FAQs", "Compiling brief"]

        brief = {}
        step = 0

        # Stage 1: Fetch SERP Data
        progress.progress(step/len(stages))
        search = GoogleSearch({"q": keyword, "location": "United States", "api_key": SERPAPI_KEY})
        serp_results = search.get_dict()
        step += 1
        progress.progress(step/len(stages))

        # Stage 2: Extract Top Pages and PAA
        top_pages = []
        people_also_ask = []
        for res in serp_results.get('organic_results', [])[:2]:
            top_pages.append(res['link'])
        for paa in serp_results.get('related_questions', []):
            people_also_ask.append(paa['question'])
        step += 1
        progress.progress(step/len(stages))

        # Stage 3: Generate Semantic Internal Links (placeholder logic)
        internal_links = []
        for url in urls_list:
            if keyword.lower() in url.lower():
                internal_links.append(url)
        step += 1
        progress.progress(step/len(stages))

        # Stage 4: Generate H2 Sections
        h2_sections = [f"{keyword} Section {i+1}" for i in range(5)]
        h2_content = []
        for idx, section in enumerate(h2_sections):
            resp = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role":"user","content":f"Write 150 words for section: {section}"}]
            )
            h2_content.append(resp.choices[0].message.content)
        step += 1
        progress.progress(step/len(stages))

        # Stage 5: Generate FAQs
        faqs = []
        for q in people_also_ask[:5]:
            resp = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role":"user","content":f"Provide a concise answer for FAQ: {q}"}]
            )
            faqs.append({'question': q, 'answer': resp.choices[0].message.content})
        step += 1
        progress.progress(step/len(stages))

        # Stage 6: Compile Brief
        brief_table = pd.DataFrame({
            'Section': ["Title", "Meta Description", "H1", "Navigation Sidebar", "People Also Ask", 
                        "Top Ranking Pages", "URL Structure", "Estimated Outlinks", "Internal Link Recommendations",
                        "Number of Page Sections"] + h2_sections + ["FAQs"],
            'Output': [
                f"SEO Optimized Title for {keyword}",
                f"Meta description for {keyword}",
                f"H1 including {keyword}",
                f"- " + "\n- ".join(h2_sections),
                "\n".join(people_also_ask),
                "\n".join(top_pages),
                f"www.example.com/{keyword.replace(' ', '-').lower()}",
                len(top_pages)*10,  # placeholder for outlinks
                "\n".join(internal_links),
                len(h2_sections),
            ] + h2_content + ["\n".join([f"{f['question']}: {f['answer']}" for f in faqs])]
        })
        step += 1
        progress.progress(step/len(stages))

        st.success("SEO Brief Generated")
        st.dataframe(brief_table, height=700)

# ---------------------- TAB 2: TOP RANKING ANALYSIS ----------------------
with tab2:
    st.header("Top Ranking Articles Analysis")
    if urls_list:
        analysis_table = pd.DataFrame({
            'URL': top_pages,
            'Title': ["Title Placeholder"]*len(top_pages),
            'Meta Description': ["Meta Placeholder"]*len(top_pages),
            'H2 Sections': [", ".join(h2_sections)]*len(top_pages),
            'Outlinks': [10]*len(top_pages),  # Placeholder
        })
        st.dataframe(analysis_table)
    else:
        st.info("Upload a CSV and generate a brief first.")
