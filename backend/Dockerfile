FROM python:3.10
COPY requirements.txt /usr/app/
WORKDIR /usr/app/
RUN pip install -r requirements.txt
COPY ./model_config/ ./model_config/
COPY ./fastapp.py .
EXPOSE 8000
CMD python fastapp.py
#CMD heroku container:push backendheroku container:push backend