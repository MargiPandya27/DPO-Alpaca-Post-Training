# Quick Start: DPO Training API

## 1️⃣ Install Dependencies

```bash
# Install API dependencies
pip install -r api/requirements.txt

# Or if using the main requirements.txt (which includes FastAPI)
pip install -r requirements.txt
```

## 2️⃣ Start the Server

```bash
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
INFO:     Initializing DPO Service...
INFO:     ✓ DPO Service initialized successfully
```

## 3️⃣ Access API Documentation

Open your browser:
- **Interactive Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Docs (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 4️⃣ Test the API

### Option A: Using Swagger UI
1. Go to [http://localhost:8000/docs](http://localhost:8000/docs)
2. Click "Try it out" on any endpoint
3. Fill in request parameters
4. Click "Execute"

### Option B: Using Python Client

```python
from api.client_example import DPOAPIClient

client = DPOAPIClient("http://localhost:8000")

# Test generation
prompts = ["What is machine learning?"]
completions = client.generate(prompts)
print(completions)

# Test judging
pairs = [("ML is good", "ML is bad")]
ranks = client.judge(prompts, pairs)
print(f"Winner: {'A' if ranks[0] == 0 else 'B'}")
```

### Option C: Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is AI?"], "temperature": 0.7}'

# Judge
curl -X POST http://localhost:8000/judge \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is AI?"], "pairs": [["AI is smart", "AI is dumb"]]}'
```

## 5️⃣ Start Training

```python
client.start_training(
    prompts=["Prompt 1", "Prompt 2"],
    chosen=["Good answer 1", "Good answer 2"],
    rejected=["Bad answer 1", "Bad answer 2"]
)

# Monitor progress
import time
while True:
    status = client.get_training_status()
    print(f"Status: {status['status']} | Progress: {status['progress']}%")
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(10)
```

## 📚 Full Documentation

See `api/README.md` for complete API documentation, deployment options, and advanced usage.

## 🔧 Troubleshooting

**Models not loading?**
- Check `config/config.yaml` paths
- Verify HuggingFace model IDs are correct
- Check GPU memory: `nvidia-smi`

**"Service not initialized" error?**
- Wait for startup messages in terminal
- Check for errors in logs
- Ensure GPU is available

**Generation is slow?**
- This is normal for LLMs on smaller GPUs
- Reduce `max_new_tokens` or `batch_size`
- Consider quantized models

## 🎯 Next Steps

1. **Read** `api/README.md` for full API reference
2. **Explore** `api/client_example.py` for usage patterns
3. **Customize** `api/dpo_service.py` for your workflow
4. **Deploy** to production using Docker or Gunicorn
