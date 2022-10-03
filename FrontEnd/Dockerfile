FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD python -m streamlit run Single_Predict.py --server.port 8080 && heroku container:push frontend 
# CMD heroku container:push frontend 