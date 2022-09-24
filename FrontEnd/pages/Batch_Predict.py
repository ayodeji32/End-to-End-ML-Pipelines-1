import streamlit as st
import pandas as pd
import requests
import time
import os

st.markdown("## Batch Customer Prediction")
st.sidebar.markdown("## Batch Predict")

customer_data = st.file_uploader("Upload Customer Segmentation Data", type={"csv", "txt"})
if customer_data is not None:
    customer_data_df = pd.read_csv(customer_data)
    file_container = st.expander("View Uploaded Data")
    file_container.write(customer_data_df)
    data_folder_dir = os.path.normpath(__file__).rsplit(os.sep, maxsplit=3)[0]
    customer_data_df.to_csv(os.path.join(data_folder_dir,'Data/uploadedfile.csv'))
    #predict = st.button('Predict')

else:
    st.info(
        f"""
            👆 Upload a .csv file first. Sample to try: [Test_Data.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
            """
    )

   #st.stop()


if st.button('Predict'):
    deployment_data = pd.DataFrame.to_json(customer_data_df)
    #deployment_data = pd.DataFrame.to_dict(spectra_df, orient='records')
    page_response = ''
    while page_response == '':
        try:
            response = requests.post('http://127.0.0.1:8000/batch_predict', json=deployment_data)
            break
        except:
            time.sleep(15)
            continue
    parsed_prediction_df = pd.DataFrame.from_dict(response.json())
    file_container2 = st.expander("View Model Predictions")
    file_container2.write(parsed_prediction_df)