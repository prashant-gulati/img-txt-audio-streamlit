# Image to Audio Story

> Upload an image. Get a story. Hear it spoken.

A Streamlit web app that turns any image into a short audio story using a fully local vision model, GPT-3.5-turbo, and OpenAI TTS — all chained together in a single pipeline.

**Live demo:** [img-txt-audio-app-3seto3syd2pzlql4yyflah.streamlit.app](https://img-txt-audio-app-3seto3syd2pzlql4yyflah.streamlit.app/)

---

## How It Works

```
Image → [BLIP] → Caption → [GPT-3.5-turbo] → Story → [OpenAI TTS] → Audio
```

| Step | Model | What it does |
|------|-------|--------------|
| 1. Caption | Salesforce BLIP | Describes what's in the image |
| 2. Story | GPT-3.5-turbo via LangChain | Turns the caption into a short creative story |
| 3. Audio | OpenAI `gpt-4o-mini-tts` | Narrates the story as a FLAC audio file |

---

## Features

- **No captions required** — the model reads and describes the image itself
- **Creative storytelling** — LLM prompt tuned for short, imaginative narratives (≤200 words)
- **High-quality audio** — lossless FLAC output with OpenAI's latest TTS model
- **Fully interactive UI** — upload, preview the image, expand to read the caption and story, then play the audio

---

## Stack

- [Streamlit](https://streamlit.io/) — UI
- [Hugging Face Transformers](https://huggingface.co/Salesforce/blip-image-captioning-base) — BLIP image captioning (runs locally)
- [LangChain](https://www.langchain.com/) — LLM orchestration
- [OpenAI API](https://platform.openai.com/) — Story generation + text-to-speech
- [PyTorch](https://pytorch.org/) — Backend for the BLIP model

---

## Getting Started

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# Clone the repo
git clone https://github.com/prashant-gulati/img-txt-audio-streamlit.git
cd img-txt-audio-streamlit

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file with your API key:

```toml
OPENAI_API_KEY = "sk-..."
```

> This file is gitignored and never committed.

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. Upload a `.jpg`, `.jpeg`, or `.png` image
2. Wait a few seconds while the pipeline runs
3. Expand **Image Caption** to see what BLIP described
4. Expand **Story** to read the generated narrative
5. Hit play on the audio player to hear it

---

## Project Structure

```
img-txt-audio/
├── app.py               # Entire application — pipeline + UI
├── requirements.txt     # Python dependencies
└── .streamlit/
    └── secrets.toml     # API key (not committed)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `transformers` | BLIP model |
| `torch` | PyTorch backend |
| `pillow` | Image loading |
| `langchain-openai` | GPT-3.5-turbo integration |
| `langchain-core` | Prompt templates and chains |
| `requests` | HTTP utilities |

---

## Notes

- The BLIP model is downloaded automatically on first run and cached in `~/.cache/huggingface/hub`
- Audio is saved as `audio.flac` in the project root
- Must be run with `streamlit run app.py`, not `python app.py`

---

**Github**
```
git init && git add .
git commit -m "$(cat <<'EOF'
initial commit
EOF
)"
git remote add origin https://github.com/prashant-gulati/img-txt-audio-streamlit.git
git push -u origin main
```
