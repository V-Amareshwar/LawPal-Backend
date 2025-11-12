# Use a lightweight Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (required by Hugging Face)
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
