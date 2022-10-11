import streamlit as st
import pandas as pd
import requests
import time
import os
import json
import base64
import time

def csv_downloader(df, link_text, data_type):
    csvfile = df.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    save_file_name = "{}_{}.csv".format(time_str,data_type)
    #st.markdown("##### Download File ###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{save_file_name}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("## Batch Customer Prediction")
st.sidebar.markdown("## Batch Predict")

customer_data = st.file_uploader("Upload Customer Segmentation Data", type={"csv", "txt"})
data_folder_dir = os.path.normpath(__file__).rsplit(os.sep, maxsplit=2)[0]

if customer_data is not None:
    customer_data_df = pd.read_csv(customer_data)
    file_container = st.expander("View Uploaded Data")
    file_container.write(customer_data_df)
    #customer_data_df.to_csv(os.path.join(data_folder_dir,'Data/uploadedfile.csv'))

else:
    customer_data_df = pd.read_csv(os.path.join(data_folder_dir,'sample_data.csv'),index_col=0)

    st.info(
        f"""
            👆 Upload a .csv file first. Sample to try: 
            """
    )
    link_text = "Download Sample Test Data Here!"
    data_type = "Sample_Data"
    csv_downloader(customer_data_df, link_text, data_type)

   #st.stop()


if st.button('Predict'):
    customer_data_df.dropna(inplace=True)
    #deployment_data = pd.DataFrame.to_json(customer_data_df, orient='records')
    deployment_data = customer_data_df.to_dict('records')
    #deployment_data = json.loads(customer_data_df.to_json(orient='records'))
    page_response = ''
    while page_response == '':
        try:  #    127.0.0.1
            response = requests.post('http://localhost:8000/batch_predict', json=deployment_data)
            break
        except:
            time.sleep(1)
            continue
    parsed_prediction_df = pd.DataFrame(response.json()) #pd.DataFrame.from_dict(response, orient='columns')
    #parsed_prediction_df = pd.json_normalize(response)
    file_container2 = st.expander("View Model Predictions")
    file_container2.write(parsed_prediction_df)
    if parsed_prediction_df is not None:
        link_text = "Download Model Predictions"
        data_type = "Predictions"
        csv_downloader(parsed_prediction_df,link_text, data_type)