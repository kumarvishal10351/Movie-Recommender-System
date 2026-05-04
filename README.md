# 🎬 CineMatch — Content-Based Movie Recommender

![App Preview](https://via.placeholder.com/1000x500/12121a/e50914?text=CineMatch+Movie+Recommender)

A production-grade, AI-powered movie recommendation engine built entirely with Python and Streamlit. CineMatch uses natural language processing (NLP) and cosine similarity to find and recommend movies based on their content (genres, keywords, cast, crew, and plot overview) from a dataset of over 4,800 films.

## ✨ Features

- **🧠 Content-Based Filtering:** Recommends movies by vectorizing and comparing metadata tags using `CountVectorizer` and cosine similarity.
- **🎨 Cinematic UI/UX:** A stunning, fully responsive dark-mode interface built natively in Streamlit with custom CSS.
- **🌐 Live TMDB Integration:** Dynamically fetches high-quality movie posters, taglines, budget/revenue data, and production studios via the TMDB API.
- **⚡ High Performance:** Utilizes Streamlit's `@st.cache_data` to ensure the heavy similarity matrix computation and API calls are cached, providing lightning-fast subsequent loads.
- **🎛️ Interactive Filtering:** Refine your recommendations on the fly by Release Year and Genre without recalculating the core similarity matrix.
- **💾 Session Management:** Seamlessly remembers your last searched movie and filters across interactions.

## 🛠️ Tech Stack

- **Core Logic:** Python, Pandas, NumPy
- **Machine Learning / NLP:** Scikit-Learn (`CountVectorizer`, `cosine_similarity`), NLTK (`PorterStemmer`)
- **Frontend / UI:** Streamlit (with custom injected CSS)
- **External Data:** TMDB REST API (for posters and extended metadata)

## 🚀 How It Works

1. **Data Preprocessing:** The app ingests the TMDB 5000 Movies & Credits datasets. It extracts and cleans crucial metadata (genres, keywords, top 3 cast members, director, and plot overview).
2. **Text Processing:** The extracted metadata is combined into a single "tag" string for each movie. NLTK's `PorterStemmer` reduces words to their root forms (e.g., "action", "actions", "actionable" -> "action").
3. **Vectorization:** `CountVectorizer` converts these text tags into vectors in a 5,000-dimensional space.
4. **Similarity Calculation:** The cosine angle between these vectors is calculated. Movies with smaller angles (higher cosine similarity scores) are deemed most similar.

## 💻 Local Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Movie-Recommender-System.git
   cd Movie-Recommender-System
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn nltk requests
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

```text
Movie Recommender System/
├── app.py                      # Main Streamlit application and UI
├── movie-recommender.ipynb     # Original Jupyter notebook with EDA & model building
├── tmdb_5000_movies.csv        # Dataset: Movie metadata
├── tmdb_5000_credits.csv       # Dataset: Cast and crew metadata
└── README.md                   # Project documentation
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/Movie-Recommender-System/issues).

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---
*Developed to showcase end-to-end Machine Learning deployment and polished Frontend Engineering skills.*
