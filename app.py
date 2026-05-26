import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Final Project - Kelompok 17", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #0f172a;}
    .reportview-container {background: #0f172a;}
    div[data-testid="stMetricContainer"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricContainer"] label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricContainer"] div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 30px; border-radius: 15px; border: 1px solid #2563eb; margin-bottom: 25px; text-align: center;">
        <span style="color: #38bdf8; font-weight: bold; tracking-spacing: 1.5px; text-transform: uppercase; font-size: 13px;">
            🚀 PYTHON FOR DATA SCIENCE FINAL PROJECT
        </span>
        <h1 style="color: #ffffff; margin: 10px 0 5px 0; font-size: 32px; font-weight: 800;">
            Sleep Health Analytics & AI Dashboard
        </h1>
        <p style="color: #94a3b8; font-size: 16px; margin-bottom: 15px;">
            Sistem visualisasi data klinis komprehensif terintegrasi dengan Predictive Model Engine
        </p>
        <div style="display: inline-block; background-color: #1e293b; padding: 6px 20px; border-radius: 20px; border: 1px solid #334155;">
            <span style="color: #f8fafc; font-weight: 600; font-size: 14px;">⚡ KELOMPOK 17</span>
        </div>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna("Normal")
    return df

df_raw = load_data()

@st.cache_resource
def kalkulasi_intelijen_ai():
    df_ml = df_raw.copy()
    
    le_gender = LabelEncoder()
    df_ml['Gender_encoded'] = le_gender.fit_transform(df_ml['Gender'])
    
    le_bmi = LabelEncoder()
    df_ml['BMI_encoded'] = le_bmi.fit_transform(df_ml['BMI Category'])
    
    le_target = LabelEncoder()
    df_ml['Target_encoded'] = le_target.fit_transform(df_ml['Sleep Disorder'])
    
    X = df_ml[[
        'Gender_encoded', 'Age', 'Sleep Duration', 'Quality of Sleep', 
        'Stress Level', 'Heart Rate', 'BMI_encoded', 'Daily Steps'
    ]]
    y = df_ml['Target_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Hutan Acak (Random Forest)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Regresi Logistik": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "K-Tetangga Terdekat (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Peningkatan Gradien (Gradient Boosting)": GradientBoostingClassifier(random_state=42)
    }
    
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, predictions) * 100
        
    return models, accuracies, le_gender, le_bmi, le_target

dict_model, dict_akurasi, encoder_gender, encoder_bmi, encoder_target = kalkulasi_intelijen_ai()

st.sidebar.markdown("### ⚙️ Konfigurasi Model")

model_aktif = st.sidebar.selectbox(
    "Pilih Model ML:",
    list(dict_model.keys()),
    index=1
)

score_aktif = dict_akurasi[model_aktif]
st.sidebar.markdown(f"""
    <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px;">
        <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Model Akurasi</span>
        <div style="font-size: 28px; font-weight: bold; color: #f8fafc; margin-top: 2px;">{score_aktif:.2f}%</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("### 🎨 Kustomisasi Visual UI")
tema_pilihan = st.sidebar.radio("Pilih Mode Kontras Monitor:", ["Deep Slate Blue (Default Dark)", "Clinical Minimalist (Light Mode)"])
if tema_pilihan == "Clinical Minimalist (Light Mode)":
    st.markdown("""
        <style>
        .main {background-color: #f8fafc !important;}
        h1, h2, h3, h4, h5, h6, p, label {color: #0f172a !important;}
        .stTabs [data-baseweb="tab"] {color: #475569 !important;}
        </style>
    """, unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("### 🎛️ Filter Analisis Eksploratif")

if st.sidebar.button("🔄 Reset Semua Filter", use_container_width=True):
    st.rerun()

pilihan_gender = st.sidebar.multiselect(
    "Filter Jenis Kelamin:",
    options=list(df_raw["Gender"].unique()),
    default=list(df_raw["Gender"].unique())
)

pilihan_profesi = st.sidebar.multiselect(
    "Filter Jenis Pekerjaan:",
    options=list(df_raw["Occupation"].unique()),
    default=list(df_raw["Occupation"].unique())
)

min_age = int(df_raw["Age"].min())
max_age = int(df_raw["Age"].max())
rentang_usia = st.sidebar.slider(
    "Batas Rentang Usia:",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

df_filtered = df_raw[
    (df_raw["Gender"].isin(pilihan_gender)) & 
    (df_raw["Occupation"].isin(pilihan_profesi)) & 
    (df_raw["Age"].between(rentang_usia[0], rentang_usia[1]))
]

st.sidebar.write("---")
st.sidebar.markdown("📊 **Informasi Kumpulan Data**")
st.sidebar.info(f"Jumlah sampel: {len(df_filtered)} \n\nCiri: 12 \n\nKelas: 3")

tab_eda, tab_prediction, tab_methodology = st.tabs([
    "📊 Analisis Eksploratif (EDA)", 
    "🔮 Pusat Prediksi AI", 
    "📖 Dokumentasi Proyek"
])

with tab_eda:
    st.markdown("## 🎯 Ringkasan Indikator Kesehatan Utama")
    
    m1, m2, m3, m4 = st.columns(4)
    if not df_filtered.empty:
        total_p = len(df_filtered)
        avg_sleep = df_filtered['Sleep Duration'].mean()
        avg_stress = df_filtered['Stress Level'].mean()
        avg_heart = df_filtered['Heart Rate'].mean()
    else:
        total_p, avg_sleep, avg_stress, avg_heart = 0, 0, 0, 0
        
    m1.metric("Total Partisipan", f"{total_p} Orang")
    m2.metric("Rerata Durasi Tidur", f"{avg_sleep:.1f} Jam")
    m3.metric("Rerata Skala Stres", f"{avg_stress:.1f} / 10")
    m4.metric("Rerata Detak Jantung", f"{avg_heart:.1f} Bpm")
    
    st.write("")
    @st.cache_data
    def convert_df(df_target):
        return df_target.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df(df_filtered)
    st.download_button(
        label="📥 Download Data Hasil Filter (.csv)",
        data=csv_data,
        file_name='sleep_health_filtered.csv',
        mime='text/csv',
    )
    
    st.markdown("---")
    st.markdown("### 🔍 Eksplorasi Granular: Sebaran Gangguan Tidur Lintas Profesi")
    
    if not df_filtered.empty:
        df_chart1 = df_filtered.groupby(['Occupation', 'Sleep Disorder']).size().reset_index(name='Jumlah Kasus/Responden')
        fig_bar = px.bar(
            df_chart1, x="Occupation", y="Jumlah Kasus/Responden", color="Sleep Disorder",
            barmode="stack", template="plotly_white",
            color_discrete_map={"Normal": "#10b981", "Insomnia": "#ef4444", "Sleep Apnea": "#f59e0b"}
        )
        fig_bar.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Data kosong. Silakan sesuaikan kembali kombinasi filter Anda di sidebar.")
        
    st.markdown("---")
    
    col_graph, col_insight = st.columns([1, 1])
    with col_graph:
        st.markdown("##### Proporsi Kategori Indeks Massa Tubuh (BMI)")
        if not df_filtered.empty:
            df_chart2 = df_filtered['BMI Category'].value_counts().reset_index()
            df_chart2.columns = ['BMI Category', 'Count']
            fig_pie = px.pie(
                df_chart2, values='Count', names='BMI Category', hole=0.5,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with col_insight:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if not df_filtered.empty:
            highest_stress_job = df_filtered.groupby('Occupation')['Stress Level'].mean().idxmax()
            highest_stress_val = df_filtered.groupby('Occupation')['Stress Level'].mean().max()
        else:
            highest_stress_job, highest_stress_val = "Tidak ada", 0
            
        st.markdown(f"""
            <div style="background-color: #f0f7ff; padding: 20px; border-radius: 12px; border-left: 5px solid #3b82f6;">
                <h5 style="color: #1e3a8a; margin-top:0px;">💡 Dynamic Business Insight:</h5>
                <p style="color: #1e293b; font-size:15px; line-height:1.6;">
                    Berdasarkan parameter filter saat ini, kelompok profesi <b>{highest_stress_job}</b> 
                    tercatat memiliki tingkat stres rata-rata tertinggi sebesar <b>{highest_stress_val:.1f} / 10</b>.
                </p>
                <p style="color: #1e293b; font-size:15px; line-height:1.6;">
                    <b>Rekomendasi Manajemen:</b> Divisi HRD disarankan melakukan evaluasi beban kerja atau 
                    mengadakan program intervensi kesehatan mental khusus untuk kelompok profesi tersebut.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌡️ Matriks Korelasi Interaktif Antar Variabel Medis")
    st.write("Menganalisis keeratan hubungan linear antar indikator fisik kuantitatif pasien.")
    kolom_numerik = df_raw[['Age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Heart Rate', 'Daily Steps']]
    matriks_korelasi = kolom_numerik.corr()
    fig_heatmap = px.imshow(
        matriks_korelasi, text_auto=".2f", 
        color_continuous_scale='RdBu_r', 
        aspect="auto",
        labels=dict(color="Koefisien Korelasi")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab_prediction:
    st.subheader(f"🔮 Mesin Diagnosis Kesehatan Utama ({model_aktif})")
    
    st.markdown("##### ⚡ Live Model Analytics Leaderboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    keys = list(dict_akurasi.keys())
    vals = list(dict_akurasi.values())
    c1.metric(keys[0], f"{vals[0]:.2f}%")
    c2.metric(keys[1], f"{vals[1]:.2f}%")
    c3.metric(keys[2], f"{vals[2]:.2f}%")
    c4.metric(keys[3], f"{vals[3]:.2f}%")
    c5.metric(keys[4], f"{vals[4]:.2f}%")
    
    st.markdown("---")
    col_inputs, col_outputs = st.columns([3, 2])
    
    with col_inputs:
        st.markdown("#### 🩺 Masukkan Parameter Medis Anda:")
        cc1, cc2 = st.columns(2)
        with cc1:
            f_gender = st.selectbox("Jenis Kelamin:", ["Male", "Female"])
            f_age = st.number_input("Usia Pengguna (Tahun):", 10, 100, 30)
            f_duration = st.slider("Durasi Istirahat (Jam Semalam):", 3.0, 10.0, 7.0)
            f_steps = st.slider("Jumlah Langkah Harian (Daily Steps):", 1000, 15000, 6000, step=500)
            
        with cc2:
            f_weight = st.number_input("Berat Badan Pengguna (kg):", 30.0, 200.0, 70.0)
            f_height = st.number_input("Tinggi Badan Pengguna (cm):", 100.0, 250.0, 170.0)
            f_quality = st.slider("Skor Kualitas Tidur (1-10):", 1, 10, 7)
            f_stress = st.slider("Skala Beban Pikiran / Stres (1-10):", 1, 10, 5)
            f_heart = st.number_input("Detak Jantung Istirahat (Bpm):", 40, 120, 72)
            
        t_m = f_height / 100
        calc_bmi = f_weight / (t_m ** 2)
        
        if calc_bmi < 18.5:
            k_bmi = "Normal"
        elif 18.5 <= calc_bmi < 25.0:
            k_bmi = "Normal"
        elif 25.0 <= calc_bmi < 30.0:
            k_bmi = "Overweight"
        else:
            k_bmi = "Obese"
            
        st.info(f"📐 **Kalkulator BMI Otomatis:** Skor Anda **{calc_bmi:.1f}** — Skenario: **{k_bmi}**")
        
        if st.button("⚡ Jalankan Komputasi Diagnosis Klasifikasi", use_container_width=True):
            g_encoded = encoder_gender.transform([f_gender])[0]
            try:
                b_encoded = encoder_bmi.transform([k_bmi])[0]
            except:
                b_encoded = encoder_bmi.transform(["Normal"])[0]
                
            engine = dict_model[model_aktif]
            prediksi_array = engine.predict([[g_encoded, f_age, f_duration, f_quality, f_stress, f_heart, b_encoded, f_steps]])
            label_hasil = encoder_target.inverse_transform(prediksi_array)[0]
            
            st.markdown("#### 🏁 Hasil Analisis Prediktif AI:")
            
            col_hasil_text, col_pdf_btn = st.columns([3, 1])
            
            with col_hasil_text:
                if label_hasil == "Normal":
                    st.success("🟢 **Kondisi Normal (Bebas Gangguan Tidur)** — Tubuh Anda memiliki koordinasi sirkadian yang sehat.")
                    st.markdown("""
                        <div style="background-color: #f0fdf4; border-left: 5px solid #16a34a; padding: 15px; border-radius: 6px; margin-top: 10px;">
                            <strong style="color: #16a34a;">💡 Smart Recommendation:</strong><br>
                            <p style="color: #1e293b; font-size:13px; margin: 5px 0 0 0;">
                                Pertahankan konsistensi jam tidur dan pertahankan aktivitas fisik harian Anda di kisaran 6000-8000 langkah untuk menyokong fase Deep Sleep malam ini.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    pdf_text = f"DIAGNOSIS REPORT:\nPatient Age: {f_age}\nStatus: NORMAL\nBMI: {calc_bmi:.1f}\nRecommendation: Maintain habits."
                elif label_hasil == "Sleep Apnea":
                    st.error(f"🔴 **Waspada! Risiko Sleep Apnea Terdeteksi.** \n\nStatus BMI Anda saat ini: {k_bmi}.")
                    st.markdown("""
                        <div style="background-color: #fef2f2; border-left: 5px solid #dc2626; padding: 15px; border-radius: 6px; margin-top: 10px;">
                            <strong style="color: #dc2626;">💡 Smart Recommendation:</strong><br>
                            <p style="color: #1e293b; font-size:13px; margin: 5px 0 0 0;">
                                Terdeteksi korelasi indeks massa tubuh berlebih. Disarankan untuk mengubah posisi tidur miring (bukan terlentang) untuk mereduksi hambatan jalan napas, dan konsultasikan ke dokter.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    pdf_text = f"DIAGNOSIS REPORT:\nPatient Age: {f_age}\nStatus: SLEEP APNEA\nBMI: {calc_bmi:.1f}\nRecommendation: Medical check-up advised."
                else:
                    st.warning(f"🟡 **Waspada! Gejala Gangguan Insomnia Terdeteksi.** \n\nSkala beban pikiran Anda berada di tingkat {f_stress}/10.")
                    st.markdown("""
                        <div style="background-color: #fffbeb; border-left: 5px solid #d97706; padding: 15px; border-radius: 6px; margin-top: 10px;">
                            <strong style="color: #d97706;">💡 Smart Recommendation:</strong><br>
                            <p style="color: #1e293b; font-size:13px; margin: 5px 0 0 0;">
                                Skala stres Anda tinggi. Kurangi paparan cahaya biru (gawai/laptop) minimal 45 menit sebelum tidur dan cobalah teknik pernapasan 4-7-8 untuk menenangkan sistem saraf.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    pdf_text = f"DIAGNOSIS REPORT:\nPatient Age: {f_age}\nStatus: INSOMNIA\nBMI: {calc_bmi:.1f}\nRecommendation: Reduce evening screen time."
            
            with col_pdf_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📄 Export Report (.txt)",
                    data=pdf_text,
                    file_name="Clinical_AI_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    with col_outputs:
        st.markdown("#### 📈 Visualisasi Akurasi Seluruh Model")
        for n_m, s_m in dict_akurasi.items():
            st.markdown(f"**{n_m}** : {s_m:.2f}%")
            st.progress(int(s_m))

with tab_methodology:
    st.markdown("## 📋 Kontrak Dokumen & Arsitektur Sistem Komputasi")
    st.write("Spesifikasi lengkap mengenai rekayasa data dan pemetaan kelas model cerdas.")
    
    data_blueprint = {
        "Komponen Utama Proyek": [
            "Judul Penelitian", 
            "Arsitektur Data Platform", 
            "Metode Pembelajaran Mesin"
        ],
        "Spesifikasi Teknis Detail": [
            "Klasifikasi Gangguan Tidur Menggunakan Machine Learning Berdasarkan Data Kesehatan dan Gaya Hidup",
            "Kaggle Core API (ucm19034aa/sleep-health-and-lifestyle-dataset)",
            "Supervised Learning via Multi-Model Engine (Akurasi Teroptimasi)"
        ]
    }
    df_blueprint = pd.DataFrame(data_blueprint)
    st.table(df_blueprint)
    
    st.write("---")
    
    st.markdown("### 🎯 Pemetaan Indikator Kelas Target Medis")
    st.write("Sistem mendeteksi tiga orientasi biologis unik berdasarkan parameter masukan:")
    
    c_normal, c_insomnia, c_apnea = st.columns(3)
    with c_normal:
        st.markdown("""
        <div style="background-color: #047857; padding: 20px; border-radius: 12px; color: #ffffff; min-height: 120px;">
            <h4 style="margin: 0px; color:#ffffff;">🟢 01. KELAS NORMAL</h4>
            <p style="font-size: 13px; margin-top: 8px; opacity: 0.9;">Kondisi istirahat stabil. Struktur ritme biologis sirkadian tubuh berfungsi secara optimal.</p>
        </div>
        """, unsafe_allow_html=True)
    with c_insomnia:
        st.markdown("""
        <div style="background-color: #b45309; padding: 20px; border-radius: 12px; color: #ffffff; min-height: 120px;">
            <h4 style="margin: 0px; color:#ffffff;">🟡 02. KELAS INSOMNIA</h4>
            <p style="font-size: 13px; margin-top: 8px; opacity: 0.9;">Tingginya resistensi sirkadian untuk memulai tidur nyenyak akibat fluktuasi kortisol/stres.</p>
        </div>
        """, unsafe_allow_html=True)
    with c_apnea:
        st.markdown("""
        <div style="background-color: #be123c; padding: 20px; border-radius: 12px; color: #ffffff; min-height: 120px;">
            <h4 style="margin: 0px; color:#ffffff;">🔴 03. SLEEP APNEA</h4>
            <p style="font-size: 13px; margin-top: 8px; opacity: 0.9;">Indikasi obstruksi klinis pernapasan akibat penyempitan jalan napas dan indeks BMI berlebih.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    st.markdown("### 🛣️ Garis Waktu Pengembangan Sistem (10 Tahapan Kerja)")
    st.write("Rangkaian modul integrasi backend dan rekayasa data secara berkala:")
    
    step1, step2, step3, step4, step5 = st.columns(5)
    with step1:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #38bdf8; min-height:140px;">
            <strong style="color:#38bdf8; font-size:13px;">MODUL 01</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Relational SQL Base</span>
            <span style="font-size:11px; color:#94a3b8;">Sinkronisasi Skema Data Relasional SQL</span>
        </div>
        """, unsafe_allow_html=True)
    with step2:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #38bdf8; min-height:140px;">
            <strong style="color:#38bdf8; font-size:13px;">MODUL 02</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Data Ingestion</span>
            <span style="font-size:11px; color:#94a3b8;">Pengumpulan & Pemuatan Data API Kaggle</span>
        </div>
        """, unsafe_allow_html=True)
    with step3:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #38bdf8; min-height:140px;">
            <strong style="color:#38bdf8; font-size:13px;">MODUL 03</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Exploratory Data</span>
            <span style="font-size:11px; color:#94a3b8;">Analisis Data Eksplorasi (EDA) & Korelasi</span>
        </div>
        """, unsafe_allow_html=True)
    with step4:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #38bdf8; min-height:140px;">
            <strong style="color:#38bdf8; font-size:13px;">MODUL 04</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Data Cleansing</span>
            <span style="font-size:11px; color:#94a3b8;">Pengolahan, Imputasi & Pembersihan Data</span>
        </div>
        """, unsafe_allow_html=True)
    with step5:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #38bdf8; min-height:140px;">
            <strong style="color:#38bdf8; font-size:13px;">MODUL 05</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Feature Engineering</span>
            <span style="font-size:11px; color:#94a3b8;">Rekayasa Fitur & Label Encoding Sklearn</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("") 

    step6, step7, step8, step9, step10 = st.columns(5)
    with step6:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #10b981; min-height:140px;">
            <strong style="color:#10b981; font-size:13px;">MODUL 06</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Data Visualization</span>
            <span style="font-size:11px; color:#94a3b8;">Visualisasi Data Interaktif via Plotly</span>
        </div>
        """, unsafe_allow_html=True)
    with step7:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #10b981; min-height:140px;">
            <strong style="color:#10b981; font-size:13px;">MODUL 07</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Data Splitting</span>
            <span style="font-size:11px; color:#94a3b8;">Pembagian Data Pelatihan (80:20) & Scaling</span>
        </div>
        """, unsafe_allow_html=True)
    with step8:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #10b981; min-height:140px;">
            <strong style="color:#10b981; font-size:13px;">MODUL 08</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Model Training</span>
            <span style="font-size:11px; color:#94a3b8;">Pelatihan Model (RF, LR, SVM, KNN, GBM)</span>
        </div>
        """, unsafe_allow_html=True)
    with step9:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #10b981; min-height:140px;">
            <strong style="color:#10b981; font-size:13px;">MODUL 09</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Model Evaluation</span>
            <span style="font-size:11px; color:#94a3b8;">Evaluasi Metriks Akurasi & Confusion Matrix</span>
        </div>
        """, unsafe_allow_html=True)
    with step10:
        st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-top:4px solid #10b981; min-height:140px;">
            <strong style="color:#10b981; font-size:13px;">MODUL 10</strong><br>
            <span style="font-size:12px; font-weight:600; color:#ffffff; display:block; margin:5px 0;">Cloud Deployment</span>
            <span style="font-size:11px; color:#94a3b8;">Deployment Sistem Cerdas via Streamlit Cloud</span>
        </div>
        """, unsafe_allow_html=True)