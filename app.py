import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import asyncio
import validators
import requests
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import streamlit as st
import pandas as pd

class URLClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1)

    @staticmethod
    def _load_urls_from_text(file_content):
        text = file_content.decode('utf-8')
        prefix = "URL="
        urls = []
        for line in text.splitlines():
            if line.startswith(prefix):
                url = line[len(prefix):].strip()
                if validators.url(url):
                    urls.append(url)
        return urls

    async def _fetch_url(self, url, session):
        try:
            async with session.get(url, timeout=10) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                return text
        except Exception as e:
            return ""

    async def _fetch_all(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(url, session) for url in urls]
            return await asyncio.gather(*tasks)

    def _determine_clusters(self, X):
        if len(X.toarray()) < 2:
            return 1
        best_score = -1
        best_k = 2
        max_clusters = min(10, len(X.toarray()) - 1)
        for k in range(2, max_clusters + 1):
            try:
                model = KMeans(n_clusters=k)
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError:
                continue
        return best_k

    def classify_urls(self, urls):
        data = asyncio.run(self._fetch_all(urls))
        contents = [d for d in data if d]
        if not contents:
            return []

        X = self.vectorizer.fit_transform(contents)
        num_clusters = self._determine_clusters(X)
        model = KMeans(n_clusters=num_clusters)
        labels = model.fit_predict(X)

        clustered_urls = zip(urls, labels)
        return sorted(clustered_urls, key=lambda x: x[1])

    def load_urls_from_files(self, files):
        urls = []
        for file in files:
            urls.extend(self._load_urls_from_text(file.getvalue()))
        return urls

    def scrape_and_analyze_content(self, urls):
        scraped_data = []
        for url in urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                main_text = soup.get_text()
                theme = self.analyze_text(main_text)
                scraped_data.append((url, theme))
            except Exception as e:
                print(f"Erreur lors du scraping de {url}: {e}")
        return scraped_data

    def analyze_text(self, text):
        text = re.sub(r'\W+', ' ', text.lower())
        nltk.download('punkt')
        tokens = nltk.word_tokenize(text)
        nltk.download('stopwords')
        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
        word_freq = Counter(filtered_tokens)
        most_common_words = word_freq.most_common(5)
        return ', '.join([word for word, freq in most_common_words])

def main():
    st.set_page_config(page_title="URL Classifier App", layout="wide")
    st.title("URL Classifier App")

    st.sidebar.title("Session State")
    st.sidebar.markdown("URLs submitted in this session:")

    classifier = URLClassifier()

    if 'url_data' not in st.session_state:
        st.session_state.url_data = []

    if st.sidebar.button("Réinitialisation"):
        st.session_state.url_data = []
        st.experimental_rerun()

    handle_file_upload(classifier)

    if st.session_state.url_data:
        st.sidebar.markdown("### URLs Submitted")
        st.sidebar.dataframe(st.session_state.url_data)

    st.write("Envoyez vos URLs pour les catégoriser.")

    if st.button("Cluster URLs"):
        perform_url_clustering(classifier)

def handle_file_upload(classifier):
    uploaded_files = st.file_uploader("Upload files with URLs", accept_multiple_files=True)
    if uploaded_files:
        urls = classifier.load_urls_from_files(uploaded_files)
        st.session_state.url_data.extend(urls)

def perform_url_clustering(classifier):
    if st.session_state.url_data:
        with st.spinner('Classifying URLs...'):
            scraped_data = classifier.scrape_and_analyze_content(st.session_state.url_data)
            clustered_urls = classifier.classify_urls([data[0] for data in scraped_data])
            if clustered_urls:
                display_clustered_urls(clustered_urls)
            else:
                st.error("No valid content found for clustering.")
    else:
        st.error("No URLs to cluster. Please upload some files first.")

def display_clustered_urls(clustered_urls):
    cluster_dict = {}
    for url, cluster in clustered_urls:
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append((url, cluster))

    st.markdown("### Clusters")
    for cluster, urls in cluster_dict.items():
        st.subheader(f"Cluster {cluster + 1}")
        data = {'Nom du Lien': [url for url, _ in urls], 'Cluster': [cluster for _, cluster in urls], 'Lien URL': [url for url, _ in urls]}
        df = pd.DataFrame(data)
        st.write(df)

        # Enregistrement du tableau dans un fichier CSV
        csv_filename = f'cluster_{cluster + 1}.csv'
        st.markdown(f"[Télécharger le CSV du Cluster {cluster + 1}](data:text/csv;charset=utf-8,{df.to_csv(index=False)})")

if __name__ == "__main__":
    main()
