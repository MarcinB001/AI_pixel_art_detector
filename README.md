# AI_pixel_art_detector

# Project Setup and Run Instructions

## Requirements
- Python 3 installed on your system
- `uvicorn` installed (`pip install uvicorn`)

## Running the Project

### Start a simple HTTP server to serve the frontend:
   ```bash
   python3 -m http.server 5500
   ```
### In another terminal, run the backend with Uvicorn:

  ```bash
  uvicorn main:app --reload
  ```
### Open the project in your browser:

http://127.0.0.1:5500/index.html
