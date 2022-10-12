FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
#ENTRYPOINT ["streamlit", "run"]
CMD python -m streamlit run Single_Predict.py --server.port=80
