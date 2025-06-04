
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spotify Cohort Analysis", layout="wide")

st.title("🎵 Spotify Cohort Analysis Web App")
st.markdown("Upload your Spotify dataset to generate cohort analysis and insights.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Sample of Uploaded Data")
        st.dataframe(df.head())

        # Example column check
        required_columns = ["user_id", "song_id", "listen_ts"]
        if all(col in df.columns for col in required_columns):

            df['listen_ts'] = pd.to_datetime(df['listen_ts'])
            df['cohort_month'] = df['listen_ts'].dt.to_period('M')

            cohort_data = df.groupby(['cohort_month', 'user_id']).size().reset_index(name='plays')
            cohort_table = cohort_data.pivot_table(index='cohort_month', columns='user_id', values='plays', fill_value=0)

            st.subheader("Cohort Table (Plays by User)")
            st.dataframe(cohort_table)

            st.subheader("Cohort Heatmap")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(cohort_table, cmap="YlGnBu", cbar_kws={'label': 'Number of Plays'})
            st.pyplot(fig)

        else:
            st.error(f"CSV must contain the following columns: {', '.join(required_columns)}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting CSV upload.")
