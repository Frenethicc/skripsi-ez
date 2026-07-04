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
designer_brands = [
    "Dior", "Chanel", "Yves Saint Laurent", "Givenchy",
    "Giorgio Armani", "Gucci", "Prada", "Versace",
    "Dolce & Gabbana", "Calvin Klein", "Burberry",
    "Hermes", "Bvlgari"
]

niche_brands = [
    "Parfums de Marly", "Maison Francis Kurkdjian",
    "Byredo", "Diptyque", "Amouage",
    "Xerjoff", "Creed", "Initio Parfums Prives",
    "Le Labo", "Kilian", "Frederic Malle",
    "Mancera", "Montale", "Nishane"
]

middle_east_brands = [
    "Lattafa", "Armaf", "Rasasi", "Ajmal",
    "Al Haramain", "Swiss Arabian", "Afnan",
    "Nabeel", "Arabian Oud", "Khadlaj"
]


def recommend_perfume(user_input, weather=None, min_rating=0, min_reviews=0,
                      brand_type=None, top_n=5):

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

    # 🔥 filter brand
    if brand_type == "designer":
        df_temp = df_temp[
            df_temp['Brand'].str.contains('|'.join(designer_brands), case=False, na=False)
        ]

    elif brand_type == "niche":
        df_temp = df_temp[
            df_temp['Brand'].str.contains('|'.join(niche_brands), case=False, na=False)
        ]

    elif brand_type == "middle_east":
        df_temp = df_temp[
            df_temp['Brand'].str.contains('|'.join(middle_east_brands), case=False, na=False)
        ]

    # ==============================
    # SORTING
    # ==============================
    results = (
        df_temp
        .sort_values(by="score", ascending=False)
        .head(top_n)
        .copy()
    )

    # ==============================
    # MATCH PERCENTAGE
    # ==============================
    min_score = results["score"].min()
    max_score = results["score"].max()

    if max_score != min_score:
        results["match_percentage"] = (
            70 + (
                (results["score"] - min_score)
                /
                (max_score - min_score)
            ) * 30
        ).round(1)
    else:
        results["match_percentage"] = 100.0

    return results   

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Perfume Recommender", page_icon="💎", layout="centered")

st.title("💎 Perfume Recommender System")
st.markdown("Temukan parfum terbaik sesuai preferensi lo 🔥")
# ==============================
# ABOUT SYSTEM
# ==============================
with st.expander("ℹ️ Tentang Sistem"):

    st.markdown("""
Sistem ini memberikan rekomendasi parfum berdasarkan **fragrance notes**
menggunakan metode **Content-Based Filtering** dengan
**TF-IDF Vectorization** dan **Cosine Similarity**.

### Cara kerja sistem

1. Masukkan satu atau beberapa fragrance notes
2. Pilih kondisi cuaca
3. Tentukan minimum rating
4. Pilih kategori brand parfum
5. Sistem akan menghitung tingkat kemiripan parfum
6. Lima parfum terbaik akan direkomendasikan

---

### Fitur Sistem

✅ Pencarian berdasarkan fragrance notes

✅ Filter cuaca (Panas / Dingin)

✅ Filter minimum rating

✅ Filter kategori brand
- Designer
- Niche
- Timur Tengah

✅ Explainable Recommendation

✅ Link menuju Fragrantica
""")

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
        user_notes = notes.lower().split()
        results = recommend_perfume(
            user_input=notes,
            weather=weather_filter,
            min_rating=min_rating,
            min_reviews=300,
            brand_type=brand_filter
        )

        # ============================================
        # HASIL REKOMENDASI
        # ============================================

        if len(results) == 0:

            st.error("😢 Tidak ditemukan parfum yang sesuai.")

            st.info("""
        Coba salah satu berikut:

        • Kurangi minimum rating
        • Pilih kategori brand 'Semua'
        • Pilih cuaca 'Semua'
        • Gunakan notes yang lebih umum
        """)

        else:

            st.success("✨ Berikut rekomendasi parfum untuk Anda")

            # ===============================
            # RINGKASAN PREFERENSI
            # ===============================
            st.markdown("## 📋 Preferensi Anda")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**📝 Notes :** {notes}")

                st.write(f"**⭐ Minimum Rating :** {min_rating}")

            with col2:

                st.write(f"**🌤️ Cuaca :** {weather}")

                st.write(f"**🏷️ Brand :** {brand_option}")

            st.markdown("---")

            # ===============================
            # HASIL
            # ===============================
            for i, row in results.iterrows():

                st.markdown(f"# 💎 {row['Perfume']}")

                st.metric(
                    "🎯 Tingkat Kemiripan",
                    f"{row['match_percentage']}%"
                )

                st.progress(row["match_percentage"]/100)

                st.write(f"**🏷️ Brand** : {row['Brand']}")

                st.write(f"**⭐ Rating** : {row['Rating Value']}")

                st.write(f"**👥 Review** : {int(row['Rating Count'])}")

                st.write(f"**🌤️ Cuaca** : {row['weather_suitability']}")

                st.markdown("### 🧠 Mengapa parfum ini direkomendasikan?")

                combined_text = str(row["combined_clean"]).lower()

                matched_notes = []

                for note in user_notes:

                    if note in combined_text:

                        matched_notes.append(note.capitalize())

                if matched_notes:

                    st.success(
                        "✅ Notes yang sesuai : "
                        + ", ".join(matched_notes)
                    )

                else:

                    st.info(
                        "Tidak ditemukan notes yang sama persis, namun parfum memiliki karakter aroma yang mirip."
                    )

                st.info(
                    f"⭐ Rating tinggi ({row['Rating Value']}) berdasarkan {int(row['Rating Count'])} review."
                )

                if weather_filter:

                    st.info(
                        f"☀️ Direkomendasikan untuk cuaca {weather_filter.capitalize()}."
                    )

                # ===============================
                # DETAIL NOTES
                # ===============================
                with st.expander("📖 Detail Notes Parfum"):

                    st.write(f"🌸 **Top Notes**")
                    st.write(row["Top"])

                    st.write(f"🌿 **Middle Notes**")
                    st.write(row["Middle"])

                    st.write(f"🪵 **Base Notes**")
                    st.write(row["Base"])

                if pd.notna(row["url"]):

                    st.markdown(
                        f"🔗 [Lihat detail di Fragrantica]({row['url']})"
                    )

                st.markdown("---")
# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.caption(
"""
💎 **Perfume Recommender System**

Developed as an undergraduate research project
using **Content-Based Filtering**, **TF-IDF Vectorization**,
and **Cosine Similarity**.

Dataset source:
Fragrantica Dataset (Kaggle)

Built with ❤️ using Streamlit.
"""
)
