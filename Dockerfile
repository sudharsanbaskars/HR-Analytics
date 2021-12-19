FROM python:3.7.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5001
ENTRYPOINT ["python"]
CMD ["app.py"]
