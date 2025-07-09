import pandas as pd
import streamlit as st
import numpy as np
import joblib
from io import BytesIO

# Load the pre-trained model
model = joblib.load('pollution_model.pkl')
model_cols = joblib.load('model_columns.pkl')

# Streamlit UI
st.title("💧 Water Pollution Predictor")
st.write("🔍 Predict pollutant levels based on Year and Station ID")

# User inputs
year = st.number_input("Enter Year: ", min_value=2000, max_value=2040, value=2025, step=1)
station_id = st.text_input("Enter Station Id: ", value='1')

# Predict button
if st.button("Predict"):
    if not station_id:
        st.warning("⚠️ Please enter a valid Station Id")
    else:
        # Prepare input
        input_data = pd.DataFrame({'year': [year], 'station_id': [station_id]})
        input_encoded = pd.get_dummies(input_data, columns=['station_id'])

        # Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        prediction = model.predict(input_encoded)[0]

        # Pollutant labels
        pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        results_df = pd.DataFrame([prediction], columns=pollutants)

        # Style table
        styled_df = results_df.style.format("{:.2f}").set_properties(**{
            'text-align': 'center',
            'background-color': '#f0f8ff',
            'border-color': 'black'
        })

        st.subheader(f"📊 Predicted pollutant levels for Station ID **{station_id}**, Year **{year}**:")
        st.dataframe(styled_df, use_container_width=True)

        # Download as CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"pollution_prediction_{station_id}_{year}.csv",
            mime='text/csv'
        )

        # Download as Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Prediction')

            st.download_button(
                label="📥 Download as Excel (XLSX)",
                data=output.getvalue(),
                file_name=f"pollution_prediction_{station_id}_{year}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
