import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD FILE
# ==============================
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

df = pd.read_pickle('data.pkl')

# ==============================
# FUNCTION
# ==============================
def recommend_perfume(user_input, weather=None, min_rating=0, min_reviews=0, top_brand=False, top_n=5):
    
    user_vec = tfidf.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix)[0]

    df_temp = df.copy()
    df_temp['score'] = sim_scores

    # filter cuaca
    if weather:
        df_temp = df_temp[df_temp['weather_suitability'] == weather]

    # filter rating
    df_temp = df_temp[
        (df_temp['Rating Value'] >= min_rating) &
        (df_temp['Rating Count'] >= min_reviews)
    ]

    # filter top brand
    if top_brand:
        top_brands = df['Brand'].value_counts().head(10).index
        df_temp = df_temp[df_temp['Brand'].isin(top_brands)]

    results = df_temp.sort_values(by='score', ascending=False)

    return results[['Perfume','Brand','Rating Value','score','weather_suitability']].head(top_n)

# ==============================
# UI
# ==============================
st.title("💎 Perfume Recommender System")

st.write("Temukan parfum terbaik sesuai preferensi lo 🔥")

# INPUT USER
notes = st.text_input("Masukkan notes parfum (contoh: vanilla woody)")

weather = st.radio("Pilih kondisi:", ["Semua", "Panas", "Dingin"])

min_rating = st.slider("Minimum Rating", 0.0, 5.0, 4.0)

top_brand = st.checkbox("Hanya tampilkan top brand")

# mapping
if weather == "Panas":
    weather_filter = "panas"
elif weather == "Dingin":
    weather_filter = "dingin"
else:
    weather_filter = None

# BUTTON
if st.button("🔍 Rekomendasikan"):
    if notes.strip() == "":
        st.warning("Masukkan notes dulu bro!")
    else:
        results = recommend_perfume(
            user_input=notes,
            weather=weather_filter,
            min_rating=min_rating,
            min_reviews=50,
            top_brand=top_brand
        )

        if len(results) == 0:
            st.error("Ga ada rekomendasi yang cocok 😢")
        else:
            st.success("Ini rekomendasi buat lo 👇")

            for i, row in results.iterrows():
                st.write(f"### {row['Perfume']}")
                st.write(f"Brand: {row['Brand']}")
                st.write(f"Rating: {row['Rating Value']}")
                st.write(f"Score: {round(row['score'], 3)}")
                st.write(f"Cuaca: {row['weather_suitability']}")
                st.markdown("---")
