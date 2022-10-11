FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD python -m streamlit run Single_Predict.py --server.port 8501 
# CMD heroku container:push frontend 
# -slim-bullseye