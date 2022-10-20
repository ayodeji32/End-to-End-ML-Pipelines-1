# End-to-End-ML-Pipeline

This is a project showing how to deploy an end to end machine learning pipeline using some of the current state of art technologies such as MLflow, Docker and FastAPI. A Customer Segmentation machine learning model is built for a fictitous manufacturer using a toy dataset with the objective of predicting the category to which the customer belongs based on some features such as Customer's Profession, Experience, Age, Family Size, Spend Score e.t.c. The ML model is served as a web application in AWS Elastic Container Service (ECS) using Elastic Beanstalk and docker containers. The front-end is a Streamlit app running in a docker container while the model is served using a FastAPI endpoint running in another container.

A separate repo will be created detailing the step by step process of building the pipeline. For a quick overview, the below diagram shows the different tools employed in architecting this pipeline.

- Jupyter: Initial data exploratory data analysis (EDA) was performed using Jupyter Notebooks and the production grade code refactored into .py files.

- Scikit-learn: A One-Vs-Rest Classifier model was built using Scikit-learn. The data transformation steps and final estimator were pickled using Sklearn Pipeline. The feature-engine package was used for the data cleaning and transformation, this allows for a much more cleaner code.

- MLflow: This open source package by Databricks was used for model management, experiment tracking and model versioning.

- Docker: Used for containerizing the front end and back end applications. Also used for testing the applications locally before deployment into the cloud. This is done using the docker-compose file in the root directory of this repo.

- Github: Used for code versioning and CI/CD deployment trigger.

- Travis-CI: This is the Continuous Integration & Continuous Deployment tool which gets triggered based on a push to the branch. Travis-CI pushes the code into Dockerhub where the docker images are built.

- Dockerhub: registry where the docker images are built and waiting for it to be picked up by AWS service.

- AWS Elastic Beanstalk: Web application deployment service in AWS. AWS looks for the Dockerrun.aws.json file in the root directory of the repo and uses the instructions contained in it to ochestrate the deployment of the docker images into containers in AWS ECS.


![image](https://user-images.githubusercontent.com/94634439/197063998-722863a1-2f14-47f5-b36b-1386abcb4f16.png)
