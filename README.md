# Multimodal RAG (Crop Disease Assistant)

This repository implements a **Multimodal Retrieval-Augmented Generation (RAG)** system using FastAPI, CLIP embeddings, Pinecone vector database, and Google Gemini.  
It allows querying with both **text** and **images** to detect crop diseases and provide recommendations.  

---

## 🚀 Getting Started

### 1. Fork the Repository

1. Go to [alumnx-ai-labs/mutimodal_RAG](https://github.com/alumnx-ai-labs/mutimodal_RAG).
2. Click **Fork** to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/<your-username>/mutimodal_RAG.git
cd mutimodal_RAG
````

### 3. Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables

Create a `.env` file in the root folder and add:

```env
GEMINI_API_KEY=your_google_generative_ai_key
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=crop-disease-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 6. Run the Server

```bash
uvicorn app.main:app --reload
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) to check if it’s running.

* Root endpoint: `/`
* Health check: `/health`

---

## 🔄 Workflow for Contributions

We follow a **fork → feature branch → PR → dev** workflow.

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Commit & Push Changes

```bash
git add .
git commit -m "Add new feature"
git push origin feature/my-new-feature
```

### 3. Raise a Pull Request

* Open a PR from your fork → `feature/my-new-feature` → **`dev` branch** of parent repo (`alumnx-ai-labs/mutimodal_RAG`).
* Do **not** raise PRs directly to `main`.

---

## 📥 Pulling the Latest Code

Always keep your fork updated with the upstream repo:

### 1. Add Upstream Remote (one-time setup)

```bash
git remote add upstream https://github.com/alumnx-ai-labs/mutimodal_RAG.git
```

### 2. Pull Latest `dev` Branch

```bash
git checkout dev
git pull upstream dev
git push origin dev
```

### 3. Pull Latest `main` Branch

```bash
git checkout main
git pull upstream main
git push origin main
```

---

## 📌 Notes

* The embedding model used is **CLIP ViT-B-32** .
* Vector storage is handled via **Pinecone** .
* API endpoints are exposed via **FastAPI** .
* Dependencies are listed in [`requirements.txt`](requirements.txt) .

---

## 🤝 Contributing

1. Fork → Feature Branch → PR to `dev`.
2. Ensure your code passes linting/tests.
3. Keep commits clean and atomic.

---

### 👨‍🌾 Example Use Cases

* Upload crop disease images with metadata.
* Query by text description or image.
* Get actionable advice (symptoms, treatment, prevention).
