# Step 1: start from a lightweight Python image
FROM python:3.9-slim

# Step 2: install system build tools (for FAISS, etc.)
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Step 3: set your working directory inside the container
WORKDIR /app

# Step 4: copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: copy your application code & FAISS index
COPY . .

# Step 6: expose the port Streamlit uses
EXPOSE 8501

# Step 7: configure Streamlit to run headless
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=false

# Step 8: the command to launch your app
CMD ["streamlit", "run", "app.py"]
