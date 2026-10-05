# Dockerfile - packages the RAG service into a portable image.
#
# An IMAGE is a blueprint, built once. A CONTAINER is a running instance of it.
# The whole point: this runs identically on your Mac, a teammate's Linux box, or
# a cloud server, with no "works on my machine" surprises.

# 1. Base image: a minimal Python that matches your local version (3.14).
#    'slim' means small, without build tooling we do not need at runtime.
FROM python:3.14-slim

# 2. The working directory inside the container. All later paths are relative to it.
WORKDIR /app

# 3. Install dependencies FIRST, before copying the app code.
#    Docker caches every instruction as a layer. By copying ONLY requirements.txt
#    at this point, the slow 'pip install' layer is reused on future builds unless
#    requirements.txt itself changes. Editing your .py files will NOT re-run pip.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Now copy the rest of the application code into the image.
COPY . .

# 5. Document the port the app listens on (informational only; -p does the real work).
EXPOSE 8000

# 6. The command run when the container starts.
#    --host 0.0.0.0 is REQUIRED inside a container: 127.0.0.1 would only be
#    reachable from within the container itself. 0.0.0.0 lets the port mapping
#    (docker run -p) reach it from your Mac.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
