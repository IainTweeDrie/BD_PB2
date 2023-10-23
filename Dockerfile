FROM python:3.8
# CREATE APP DIRECTIVE
WORKDIR /app
# ADD SCRIPT TO APP DIRECTIVE
COPY bdt-sentiment.py .
# INSTALL REQUIREMENTS
RUN pip install --no-cache-dir transformers google-cloud-storage
# RUN SCRIPT
CMD ["python", "bdt-sentiment.py"]
