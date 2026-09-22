# IT GRC Copilot

This project is a browser-based IT GRC assistant built with plain HTML, CSS, and JavaScript.

## What it does

- Loads a local GRC knowledge base from the `grc-knowledge` folder
- Searches the knowledge base for the most relevant control, policy, and compliance content
- Uses WebLLM in the browser to generate responses locally
- Supports microphone input for voice-based queries
- Focuses on IT governance, risk, compliance, and control mapping workflows

## Project structure

- `index.html` — app shell and UI
- `app.js` — app logic, knowledge search, and model loading
- `styles.css` — styling
- `grc-knowledge/` — compliance and GRC knowledge content
- `vercel.json` — Vercel hosting config (if used)

## Run locally

Open the project in a browser via a local static server, for example:

```bash
python -m http.server 8000
```

Then visit:

```text
http://localhost:8000/
```

## Notes

- WebLLM requires a browser with WebGPU support.
- Chrome or Edge is recommended.
- The app is intended for local browser-based AI inference and does not require a backend server.
