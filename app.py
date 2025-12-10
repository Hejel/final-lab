import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Visualisasi Data - Final Lab Machine Learning")
st.write("Menampilkan eksplorasi dataset mobile price classification.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

train = load_data()

st.subheader("Preview Dataset")
st.dataframe(train.head())

# =============================
# VISUALISASI 1 — Countplot
# =============================
st.subheader("Distribusi price_range")
fig1, ax1 = plt.subplots()
sns.countplot(x="price_range", data=train, ax=ax1)
ax1.set_title("Jumlah Sample per Kelas Harga")
st.pyplot(fig1)

# =============================
# VISUALISASI 2 — Histogram RAM
# =============================
st.subheader("Distribusi RAM")
fig2, ax2 = plt.subplots()
sns.histplot(train["ram"], kde=True, bins=40, ax=ax2)
ax2.set_title("Distribusi RAM (MB)")
st.pyplot(fig2)

# =============================
# VISUALISASI 3 — Boxplot RAM vs price_range
# =============================
st.subheader("RAM berdasarkan price_range")
fig3, ax3 = plt.subplots()
sns.boxplot(x="price_range", y="ram", data=train, ax=ax3)
ax3.set_title("RAM by price_range")
st.pyplot(fig3)

# =============================
# VISUALISASI 4 — Barplot Battery Power
# =============================
st.subheader("Rata-rata Battery Power per Kelas Harga")
fig4, ax4 = plt.subplots()
sns.barplot(x="price_range", y="battery_power", data=train, ax=ax4)
ax4.set_title("Rata-rata Battery Power")
st.pyplot(fig4)

st.write("Visualisasi selesai ✔")
