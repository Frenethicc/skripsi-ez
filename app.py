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

    return results.head(top_n)

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Perfume Recommender", page_icon="💎", layout="centered")

st.title("💎 Perfume Recommender System")
st.markdown("Temukan parfum terbaik sesuai preferensi lo 🔥")

# ==============================
# NOTES GUIDE
# ==============================
with st.expander("📚 Panduan Notes Parfum (Klik untuk lihat)"):
    st.markdown("""
    ### 🌸 Floral (Bunga)
    rose, jasmine, lily, tuberose, orange blossom, iris, violet  

    ### 🍋 Fresh / Citrus
    lemon, bergamot, orange, grapefruit, mandarin  

    ### 🌿 Green / Herbal
    tea, grass, mint, basil, lavender, rosemary  

    ### 🍬 Sweet / Gourmand
    vanilla, caramel, chocolate, honey, sugar, tonka bean  

    ### 🌳 Woody
    sandalwood, cedar, oud, vetiver, patchouli  

    ### 🔥 Spicy
    cinnamon, pepper, clove, nutmeg, cardamom  

    ### 🌊 Aquatic / Marine
    ocean, sea salt, marine notes, algae  

    ### 🍎 Fruity
    apple, pear, peach, berries, pineapple  

    ### 🧼 Clean / Musky
    musk, powdery, soapy, aldehydes  

    ### 🪵 Warm / Amber
    amber, resin, balsamic, incense  

    ---

    ### 👉 Tips Kombinasi:
    - fresh citrus → cocok siang 🌞  
    - woody spicy → cocok malam 🌙  
    - sweet vanilla → cozy & hangat  
    - floral fruity → feminin & ringan  

    ---

    ### ✏️ Contoh input:
    - vanilla woody  
    - fresh citrus  
    - sweet floral  
    - musky powdery  
    - amber spicy
    """)
# ==============================
# INPUT
# ==============================
notes = st.text_input("✏️ Masukkan notes parfum", placeholder="contoh: vanilla woody")

weather = st.radio("🌤️ Pilih kondisi cuaca:", ["Semua", "Panas", "Dingin"])

min_rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 4.0)

brand_option = st.radio(
    "🏷️ Pilih kategori brand:",
    ["Semua", "Designer", "Niche", "Timur Tengah"]
)

# mapping cuaca
if weather == "Panas":
    weather_filter = "panas"
elif weather == "Dingin":
    weather_filter = "dingin"
else:
    weather_filter = None

# mapping brand
if brand_option == "Designer":
    brand_filter = "designer"
elif brand_option == "Niche":
    brand_filter = "niche"
elif brand_option == "Timur Tengah":
    brand_filter = "middle_east"
else:
    brand_filter = None

# ==============================
# BUTTON
# ==============================
if st.button("🔍 Rekomendasikan"):
    if notes.strip() == "":
        st.warning("Masukkan notes dulu bro!")
    else:
        results = recommend_perfume(
    user_input=notes,
    weather=weather_filter,
    min_rating=min_rating,
    min_reviews=50,
    brand_type=brand_filter
)

        if len(results) == 0:
            st.error("Ga ada rekomendasi yang cocok 😢")
        else:
            st.success("Ini rekomendasi buat lo 👇")

            for i, row in results.iterrows():
                st.markdown(f"### 💎 {row['Perfume']}")
                st.write(f"**Brand:** {row['Brand']}")
                st.write(f"⭐ Rating: {row['Rating Value']}")
                st.write(f"🌤️ Cuaca: {row['weather_suitability']}")

                # 🔗 Link Fragrantica
                if 'url' in row:
                    st.markdown(f"[🔗 Lihat di Fragrantica]({row['url']})")

                st.markdown("---")
