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

def find_similar_materials(df, requirements, weights, num_results=5):
    if df.empty:
        return pd.DataFrame()
    
    if 'Use' in df.columns:
        if df['Use'].dtype == object:
            df['Use'] = df['Use'].map({'TRUE': True, 'True': True, 'true': True, 
                                      'FALSE': False, 'False': False, 'false': False})
        
        usable_df = df[df['Use'] == True].copy()
        
        if usable_df.empty:
            return pd.DataFrame()
    else:
        usable_df = df.copy()
    
    # Use full property names
    numeric_columns = ['Ultimate_Tensile_Strength_MPa', 'Yield_Strength_MPa', 
                      'Elastic_Modulus_MPa', 'Shear_Modulus_MPa', 
                      'Poissons_Ratio', 'Density_kg_per_m3']
    
    for col in numeric_columns:
        if col not in usable_df.columns:
            st.error(f"Required column '{col}' not found in data.")
            return pd.DataFrame()
        if not pd.api.types.is_numeric_dtype(usable_df[col]):
            usable_df[col] = pd.to_numeric(usable_df[col], errors='coerce')
    
    usable_df = usable_df.dropna(subset=numeric_columns)
    
    if usable_df.empty:
        return pd.DataFrame()
    
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

def create_safe_slider(label, min_val, max_val, default_val):
    if min_val == max_val:
        if min_val == 0:
            min_val = 0
            max_val = 0.001
        else:
            min_val = min_val * 0.99
            max_val = max_val * 1.01
    
    if default_val < min_val:
        default_val = min_val
    elif default_val > max_val:
        default_val = max_val
    
    return st.slider(label, min_value=float(min_val), max_value=float(max_val), value=float(default_val))

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
        'Poissons_Ratio': {'display_name': 'Poisson\'s Ratio'},
        'Density_kg_per_m3': {'display_name': 'Density (kg/m³)'}
    }
    
    # Calculate min, max, and default values for each property
    for prop, info in properties.items():
        if prop in df.columns and pd.api.types.is_numeric_dtype(df[prop]):
            min_val = df[prop].min()
            max_val = df[prop].max()
            
            if min_val == max_val:
                if min_val == 0:
                    min_val = 0
                    max_val = 0.001
                else:
                    min_val = min_val * 0.99
                    max_val = max_val * 1.01
            
            properties[prop]['min'] = min_val
            properties[prop]['max'] = max_val
            properties[prop]['default'] = (min_val + max_val) / 2
        else:
            st.sidebar.warning(f"Property '{prop}' not found in data or not numeric.")
            properties[prop]['min'] = 0
            properties[prop]['max'] = 100
            properties[prop]['default'] = 50
    
    requirements = {}
    weights = {}
    
    # Create sliders for each property
    for prop, info in properties.items():
        st.sidebar.subheader(info['display_name'])
        requirements[prop] = create_safe_slider(
            f"Target value for {info['display_name']}", 
            info['min'], 
            info['max'], 
            info['default']
        )
        weights[prop] = st.sidebar.slider(
            f"Importance weight for {info['display_name']}", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            step=0.1
        )
    
    num_results = st.sidebar.slider("Number of recommended materials", 1, 10, 5)
    
    if st.sidebar.button('Find Recommended Materials'):
        req_list = [
            requirements.get('Ultimate_Tensile_Strength_MPa', 0),
            requirements.get('Yield_Strength_MPa', 0),
            requirements.get('Elastic_Modulus_MPa', 0),
            requirements.get('Shear_Modulus_MPa', 0),
            requirements.get('Poissons_Ratio', 0),
            requirements.get('Density_kg_per_m3', 0)
        ]
        
        weight_list = [
            weights.get('Ultimate_Tensile_Strength_MPa', 0.5),
            weights.get('Yield_Strength_MPa', 0.5),
            weights.get('Elastic_Modulus_MPa', 0.5),
            weights.get('Shear_Modulus_MPa', 0.5),
            weights.get('Poissons_Ratio', 0.5),
            weights.get('Density_kg_per_m3', 0.5)
        ]
        
        similar = find_similar_materials(df, req_list, weight_list, num_results)
        
        if similar.empty:
            st.error("No suitable materials found. Please adjust your requirements.")
            return
        
        st.header(f'Top {len(similar)} Recommended Materials')
        
        similar['Distance_Score'] = similar['Distance_Score'].round(4)
        
        display_columns = ['Material'] + list(properties.keys()) + ['Distance_Score']
        display_columns = [col for col in display_columns if col in similar.columns]
        
        st.dataframe(similar[display_columns].style.format({
            'Ultimate_Tensile_Strength_MPa': '{:.1f}',
            'Yield_Strength_MPa': '{:.1f}',
            'Elastic_Modulus_MPa': '{:.1f}',
            'Shear_Modulus_MPa': '{:.1f}',
            'Poissons_Ratio': '{:.3f}',
            'Density_kg_per_m3': '{:.1f}',
            'Distance_Score': '{:.4f}'
        }))
        
        available_props = [prop for prop in properties if prop in similar.columns]
        
        if not available_props:
            st.warning("No property columns available for visualization.")
            return
        
        # Radar chart
        st.header('Material Property Radar Chart (Normalized)')
        
        radar_df = similar[['Material'] + available_props].copy()
        
        req_data = {'Material': ['Your Requirements']}
        for prop in available_props:
            req_data[prop] = [requirements.get(prop, 0)]
        
        req_row = pd.DataFrame(req_data)
        radar_df = pd.concat([req_row, radar_df], ignore_index=True)
        
        scaler = MinMaxScaler()
        radar_df[available_props] = scaler.fit_transform(radar_df[available_props])
        
        fig = go.Figure()
        
        for i, row in radar_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row[available_props].values,
                theta=[properties[prop]['display_name'] for prop in available_props],
                fill='toself',
                name=row['Material']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual property comparison
        st.header('Detailed Property Comparison')
        
        selected_prop = st.selectbox('Select property to analyze', 
                                   [(prop, properties[prop]['display_name']) for prop in available_props],
                                   format_func=lambda x: x[1])
        
        selected_prop_key = selected_prop[0]
        
        bar_fig = px.bar(
            similar, 
            x='Material', 
            y=selected_prop_key,
            title=f"Comparison of {properties[selected_prop_key]['display_name']}",
            color='Distance_Score',
            color_continuous_scale='viridis_r',
            labels={selected_prop_key: properties[selected_prop_key]['display_name']}
        )
        
        bar_fig.add_hline(
            y=requirements.get(selected_prop_key, 0),
            line_dash="dash",
            line_color="red",
            annotation_text="Your Target",
            annotation_position="top left"
        )
        
        st.plotly_chart(bar_fig, use_container_width=True)
        
        # Deviation analysis
        st.header('Deviation from Target Requirements')
        
        deviation_df = similar[['Material'] + available_props].copy()
        for prop in available_props:
            req_val = requirements.get(prop, 0)
            if req_val != 0:
                deviation_df[prop] = ((deviation_df[prop] - req_val) / req_val * 100).round(2)
            else:
                deviation_df[prop] = (deviation_df[prop] - req_val).round(2)
        
        deviation_df = deviation_df.set_index('Material')
        
        fig_heatmap = px.imshow(
            deviation_df,
            labels=dict(x="Property", y="Material", color="% Deviation from Target"),
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title="Percentage Deviation from Your Specified Requirements"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Property correlations
        st.header('Material Property Correlations')
        
        corr = df[available_props].corr()
        
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Viridis',
            labels=dict(color="Correlation Strength"),
            title="Correlation Matrix of Material Properties"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # PCA analysis
        if len(df) >= 3:
            from sklearn.decomposition import PCA
            
            st.header('Material Property Dimensionality Analysis')
            st.write('This 2D projection shows how materials relate based on all their properties')
            
            try:
                pca = PCA(n_components=2)
                components = pca.fit_transform(df[available_props])
                
                pca_df = pd.DataFrame(data=components, columns=['Principal_Component_1', 'Principal_Component_2'])
                pca_df['Material'] = df['Material']
                
                fig_pca = px.scatter(
                    pca_df, 
                    x='Principal_Component_1', 
                    y='Principal_Component_2', 
                    hover_data=['Material'],
                    title='Material Similarity Projection'
                )
                
                st.plotly_chart(fig_pca, use_container_width=True)
                
                st.write(f"First two components explain {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")
            except Exception as e:
                st.warning(f"PCA analysis skipped due to: {str(e)}")
        
        # Export options
        st.header("Export Results")
        
        csv = similar.to_csv(index=False)
        st.download_button(
            label="Download Recommendations as CSV",
            data=csv,
            file_name="material_recommendations.csv",
            mime="text/csv",
        )
    
    else:
        st.info('Configure your requirements and click "Find Recommended Materials"')
        
        st.header("Dataset Overview")
        st.write(f"Total materials in database: {len(df)}")
        
        if st.checkbox("Show property distributions"):
            prop_to_plot = st.selectbox(
                "Select property to visualize", 
                [(prop, info['display_name']) for prop, info in properties.items()],
                format_func=lambda x: x[1]
            )
            
            prop_key = prop_to_plot[0]
            
            if prop_key in df.columns and pd.api.types.is_numeric_dtype(df[prop_key]):
                fig = px.histogram(
                    df, 
                    x=prop_key, 
                    title=f"Distribution of {properties[prop_key]['display_name']}",
                    nbins=20,
                    labels={prop_key: properties[prop_key]['display_name']}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Cannot display distribution for {properties[prop_key]['display_name']}")

if __name__ == "__main__":
    run_app()