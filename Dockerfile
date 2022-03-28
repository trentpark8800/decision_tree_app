FROM python:3.8.8-slim
EXPOSE 80
WORKDIR /usr/src/decision_tree_app
COPY ["decision_tree_app.py", "requirements.txt", "config.toml", "credentials.toml",  "./"]
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
COPY data_management ./data_management
COPY data_modelling ./data_modelling
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run"]
CMD ["decision_tree_app.py"]
