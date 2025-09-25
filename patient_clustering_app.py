import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Patient Clustering Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Patient Clustering Analysis Dashboard")
st.markdown("---")

# Sidebar for upload
st.sidebar.header("📁 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.sidebar.success("✅ File uploaded successfully!")
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            st.metric("Data Types", f"{len(df.select_dtypes(include=[np.number]).columns)} numerical")
        
        st.dataframe(df.head(), use_container_width=True)
        
        # Clustering parameters
        st.sidebar.header("⚙️ Clustering Settings")
        
        # Auto-detect numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        features = st.sidebar.multiselect(
            "Select features for clustering:",
            options=numerical_features,
            default=['Age', 'Length_of_Stay', 'Admission_Deposit', 'Visitors_Per_Week', 
                    'Number_of_Procedures', 'Previous_Admissions', 'Chronic_Conditions']
        )
        
        n_clusters = st.sidebar.slider("Number of clusters:", 2, 10, 6)
        
        if st.sidebar.button("🚀 Run Clustering Analysis", type="primary"):
            with st.spinner("Performing clustering analysis..."):
                
                # Data preprocessing
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(X_scaled)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
                
                # PCA for visualization
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_scaled)
                df['PC1'] = principal_components[:, 0]
                df['PC2'] = principal_components[:, 1]
                
                st.success(f"✅ Clustering completed! Silhouette Score: {silhouette_avg:.3f}")
                
                # Display quality metric
                if silhouette_avg > 0.7:
                    quality = "Excellent"
                    color = "green"
                elif silhouette_avg > 0.5:
                    quality = "Good"
                    color = "blue"
                elif silhouette_avg > 0.3:
                    quality = "Fair"
                    color = "orange"
                else:
                    quality = "Poor"
                    color = "red"
                
                st.sidebar.markdown(f"**Clustering Quality:** :{color}[{quality}]")
                st.sidebar.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                
                # Results section
                st.subheader("🎯 Clustering Results")
                
                # Cluster distribution and PCA visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📈 Cluster Distribution**")
                    cluster_counts = df['Cluster'].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
                    plt.xlabel('Cluster')
                    plt.ylabel('Number of Patients')
                    plt.title('Patient Distribution Across Clusters')
                    for i, count in enumerate(cluster_counts.values):
                        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
                    st.pyplot(fig)
                
                with col2:
                    st.write("**🔍 PCA Visualization**")
                    fig = px.scatter(df, x='PC1', y='PC2', color='Cluster', 
                                   title='Patient Clusters (PCA)',
                                   hover_data=features,
                                   color_continuous_scale='viridis')
                    st.plotly_chart(fig)
                
                # Cluster profiles
                st.subheader("📋 Cluster Profiles")
                cluster_profile = df.groupby('Cluster')[features].mean().round(2)
                st.dataframe(cluster_profile, use_container_width=True)
                
                # Cluster Personas & Insights
                st.subheader("🎭 Cluster Personas & Insights")
                
                # Auto-generate personas based on characteristics
                for cluster_num in sorted(df['Cluster'].unique()):
                    cluster_data = df[df['Cluster'] == cluster_num]
                    
                    with st.expander(f"🔹 Cluster {cluster_num} - {len(cluster_data)} Patients", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f} years")
                        with col2:
                            st.metric("Avg Stay", f"{cluster_data['Length_of_Stay'].mean():.1f} days")
                        with col3:
                            st.metric("Avg Deposit", f"${cluster_data['Admission_Deposit'].mean():,.0f}")
                        with col4:
                            st.metric("Emergency Rate", f"{cluster_data['Emergency_Admission'].mean():.1%}")
                        
                        # Generate persona description
                        age_desc = "Young" if cluster_data['Age'].mean() < 40 else "Middle-aged" if cluster_data['Age'].mean() < 60 else "Elderly"
                        stay_desc = "Short" if cluster_data['Length_of_Stay'].mean() < 10 else "Moderate" if cluster_data['Length_of_Stay'].mean() < 20 else "Long"
                        cost_desc = "Low-cost" if cluster_data['Admission_Deposit'].mean() < 6000 else "High-cost"
                        
                        persona = f"{age_desc} {cost_desc} {stay_desc}-Stay"
                        
                        st.write(f"**Persona:** {persona}")
                        st.write(f"**Key Characteristics:**")
                        st.write(f"- Chronic Conditions: {cluster_data['Chronic_Conditions'].mean():.1f}")
                        st.write(f"- Procedures: {cluster_data['Number_of_Procedures'].mean():.1f}")
                        st.write(f"- Previous Admissions: {cluster_data['Previous_Admissions'].mean():.1f}")
                
                # Key Performance Indicators
                st.subheader("📊 Key Performance Indicators")
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                with kpi1:
                    st.metric("Total Patients", f"{len(df):,}")
                with kpi2:
                    st.metric("Average Age", f"{df['Age'].mean():.1f} years")
                with kpi3:
                    st.metric("Average Stay", f"{df['Length_of_Stay'].mean():.1f} days")
                with kpi4:
                    st.metric("Average Deposit", f"${df['Admission_Deposit'].mean():,.0f}")
                
                # Download section
                st.subheader("📥 Download Results")
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Download full data
                    csv_full = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Full Clustered Data",
                        data=csv_full,
                        file_name='patient_clusters_full.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                with download_col2:
                    # Download cluster summary
                    cluster_summary = df.groupby('Cluster').agg({
                        'Age': 'mean',
                        'Length_of_Stay': 'mean',
                        'Admission_Deposit': 'mean',
                        'Emergency_Admission': 'mean'
                    }).round(2)
                    csv_summary = cluster_summary.to_csv()
                    st.download_button(
                        label="📈 Download Cluster Summary",
                        data=csv_summary,
                        file_name='cluster_summary.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                # Patient Search
                st.subheader("🔍 Patient Search")
                
                patient_id = st.number_input("Enter Patient ID to view cluster assignment:", 
                                           min_value=int(df['Patient_ID'].min()), 
                                           max_value=int(df['Patient_ID'].max()),
                                           value=int(df['Patient_ID'].iloc[0]))
                
                if patient_id in df['Patient_ID'].values:
                    patient_data = df[df['Patient_ID'] == patient_id].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Patient {patient_id}** is in **Cluster {patient_data['Cluster']}**")
                        st.info(f"**Persona:** {age_desc} {cost_desc} {stay_desc}-Stay")
                    
                    with col2:
                        st.write("**Patient Details:**")
                        st.write(f"- Age: {patient_data['Age']}")
                        st.write(f"- Length of Stay: {patient_data['Length_of_Stay']} days")
                        st.write(f"- Admission Deposit: ${patient_data['Admission_Deposit']:,}")
                
                st.markdown("---")
                st.success("🎉 Analysis Complete! Your patient clustering dashboard is ready.")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please make sure your CSV file has the correct format.")

else:
    st.info("👈 Please upload a CSV file to get started!")
    st.markdown("""
    ### 📋 Expected CSV Format:
    Your file should contain these columns (or similar):
    - `Patient_ID` - Unique patient identifier
    - `Age` - Patient's age
    - `Length_of_Stay` - Duration of hospital stay
    - `Admission_Deposit` - Deposit amount
    - `Visitors_Per_Week` - Number of weekly visitors
    - `Number_of_Procedures` - Medical procedures count
    - `Emergency_Admission` - 1 for emergency, 0 for planned
    - `Previous_Admissions` - History of admissions
    - `Chronic_Conditions` - Number of chronic conditions
    - `Medication_Count` - Medications prescribed
    
    ### 🚀 How to Use:
    1. Upload your patient data CSV
    2. Select features for clustering
    3. Choose number of clusters
    4. Click "Run Clustering Analysis"
    5. Explore results and download insights!
    """)