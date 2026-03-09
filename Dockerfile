# Use a specific Python base image (e.g., Python 3.12-slim)
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Set up environment variables to prevent Python from writing pyc files to disc 
# and to ensure stdout and stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nltk
RUN python -m nltk.downloader punkt stopwords punkt_tab wordnet omw-1.4

# Copy the rest of your application code to the container
COPY . .

# Expose the port Railway will use (handled by the $PORT env var at runtime)
EXPOSE 8000

# Command to run the application with 

CMD ["gunicorn", "clarityscan.wsgi:application", "--bind", "0.0.0.0:8000", "--log-file", "-"]
