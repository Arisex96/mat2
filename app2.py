import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load the data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\kr96a\Desktop\biomaterials\try_project1\data\Data.csv')

        # Convert numeric columns to appropriate types
        numeric_columns = ['Ultimate_Tensile_Strength_MPa', 'Yield_Strength_MPa',
                           'Elastic_Modulus_MPa', 'Shear_Modulus_MPa',
                           'Poissons_Ratio', 'Density_kg_per_m3']

        # Map old column names to new full names if needed
        column_mapping = {
            'Su': 'Ultimate_Tensile_Strength_MPa',
            'Sy': 'Yield_Strength_MPa',
            'E': 'Elastic_Modulus_MPa',
            'G': 'Shear_Modulus_MPa',
            'mu': 'Poissons_Ratio',
            'Ro': 'Density_kg_per_m3'
        }

        # Rename columns if old abbreviations exist
        df = df.rename(columns=column_mapping)

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle 'Use' column if it exists
        if 'Use' in df.columns:
            df['Use'] = df['Use'].astype(str).str.lower().map({'true': True, 'false': False})

        # Drop rows with NaN values in numeric columns
        df = df.dropna(subset=numeric_columns)
        return df

    except FileNotFoundError:
        st.error("DATA.csv file not found. Using sample data instead.")
        
        # Create sample data with full property names
        data = {
            'Material': ['ANSI Steel SAE 1015 as-rolled', 'ANSI Steel SAE 1015 normalized',
                         'ANSI Steel SAE 1015 annealed', 'ANSI Steel SAE 1020 as-rolled',
                         'ANSI Steel SAE 1020 normalized'],
            'Ultimate_Tensile_Strength_MPa': [421, 424, 386, 448, 441],
            'Yield_Strength_MPa': [314, 324, 284, 331, 346],
            'Elastic_Modulus_MPa': [207000, 207000, 207000, 207000, 207000],
            'Shear_Modulus_MPa': [79000, 79000, 79000, 79000, 79000],
            'Poissons_Ratio': [0.3, 0.3, 0.3, 0.3, 0.3],
            'Density_kg_per_m3': [7860, 7860, 7860, 7860, 7860],
            'Use': [True, True, True, True, True]
        }
        
        return pd.DataFrame(data)

# Find similar materials based on requirements and weights
def find_similar_materials(df, requirements, weights, num_results=5):
    if df.empty:
        return pd.DataFrame()

    usable_df = df[df['Use'] == True].copy() if ('Use' in df.columns) else df.copy()
    numeric_columns = ['Ultimate_Tensile_Strength_MPa', 'Yield_Strength_MPa', 
                       'Elastic_Modulus_MPa', 'Shear_Modulus_MPa', 
                       'Poissons_Ratio', 'Density_kg_per_m3']

    req_df = pd.DataFrame([requirements], columns=numeric_columns)
    scaler = MinMaxScaler()
    combined = pd.concat([usable_df[numeric_columns], req_df])
    scaler.fit(combined)

    usable_scaled = scaler.transform(usable_df[numeric_columns])
    req_scaled = scaler.transform(req_df)

    weighted_usable = usable_scaled * np.array(weights)
    weighted_req = req_scaled * np.array(weights)

    distances = euclidean_distances(weighted_req, weighted_usable)[0]
    usable_df['Distance_Score'] = distances

    num_results = min(num_results, len(usable_df))
    similar_materials = usable_df.sort_values('Distance_Score').head(num_results)
    
    return similar_materials

# Run the Streamlit app
def run_app():
    st.title('Advanced Material Selection Tool')
    st.write('Find optimal materials based on your mechanical property requirements')

    df = load_data()
    if df.empty:
        st.error("No data available. Please check your DATA.csv file.")
        return

    if st.checkbox("Show data preview"):
        st.write(df.head())

    # Sidebar for requirements input
    st.sidebar.header('Material Requirements Specification')
    
    properties = {
        'Ultimate_Tensile_Strength_MPa': {'display_name': 'Ultimate Tensile Strength (MPa)'},
        'Yield_Strength_MPa': {'display_name': 'Yield Strength (MPa)'},
        'Elastic_Modulus_MPa': {'display_name': 'Elastic Modulus (MPa)'},
        'Shear_Modulus_MPa': {'display_name': 'Shear Modulus (MPa)'},
        'Poissons_Ratio': {'display_name': "Poisson's Ratio"},
        'Density_kg_per_m3': {'display_name': "Density (kg/m³)"}
    }

    requirements = {}
    weights = {}

    # Create sliders for each property
    for prop_key in properties.keys():
        min_val = float(df[prop_key].min())
        max_val = float(df[prop_key].max())
        
        requirements[prop_key] = st.sidebar.slider(
            f"Target value for {properties[prop_key]['display_name']}",
            min_val,
            max_val,
            (min_val + max_val) / 2,
            step=1.0
        )
        
        weights[prop_key] = st.sidebar.slider(
            f"Importance weight for {properties[prop_key]['display_name']}",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    num_results = st.sidebar.slider("Number of recommended materials", min_value=1, max_value=10)

    if st.sidebar.button("Find Recommended Materials"):
        similar_materials = find_similar_materials(df, list(requirements.values()), list(weights.values()), num_results)
        
        if similar_materials.empty:
            st.error("No suitable materials found. Please adjust your requirements.")
            return

        st.header(f'Top {num_results} Recommended Materials')
        
        display_columns = ['Material'] + list(properties.keys()) + ['Distance_Score']
        display_columns = [col for col in display_columns if col in similar_materials.columns]
        
        st.dataframe(similar_materials[display_columns].style.format({
            prop: '{:.2f}' for prop in properties.keys()
        }).format({'Distance_Score': '{:.4f}'}))

# Main function to run the app
if __name__ == "__main__":
    run_app()
