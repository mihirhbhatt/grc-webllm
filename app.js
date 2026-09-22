import { CreateMLCEngine } from "@mlc-ai/web-llm";

// ---- CONFIGURATION ----
const config = {
  model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC',
  fallbackModels: [
    'Llama-3.2-1B-Instruct-q4f32_1-MLC',
    'Llama-3.2-3B-Instruct-q4f32_1-MLC',
    'Llama-3.1-8B-Instruct-q4f32_1-MLC',
    'Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC',
    'Phi-3.5-mini-instruct-q4f32_1-MLC',
    'Phi-3-mini-4k-instruct-q4f32_1-MLC',
    'phi-2-q4f32_1-MLC',
    'phi-1_5-q4f32_1-MLC',
    'gemma-2-2b-it-q4f32_1-MLC',
    'gemma-2b-it-q4f32_1-MLC',
    'Mistral-7B-Instruct-v0.3-q4f32_1-MLC',
    'Hermes-2-Pro-Mistral-7B-q4f16_1-MLC',
    'NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC',
    'OpenHermes-2.5-Mistral-7B-q4f16_1-MLC',
    'Qwen3-0.6B-q4f32_1-MLC',
    'Qwen3-1.7B-q4f32_1-MLC',
    'Qwen3-4B-q4f32_1-MLC',
    'Qwen3-8B-q4f32_1-MLC',
    'Qwen3.5-2B-q4f32_1-MLC',
    'Qwen3.5-4B-q4f32_1-MLC'
  ],
  knowledgeRoot: './grc-knowledge',
  knowledgeManifest: './grc-knowledge/index.json',
  maxContext: 4096,
  topK: 10,
  temperature: 0.2,
  chunkSize: 6,
  localOnly: true,
  streaming: true,
  enableWebLlm: true,
  autoStartWebLlm: false,
  allowBackgroundInit: false,
  strictCitations: true,
  maxOutputTokens: 2048,
};
const inputTokenBudget = config.maxContext - config.maxOutputTokens - 300;

// ---- SYSTEM PROMPT ----
const SYSTEM_PROMPT = `You are a senior IT GRC analyst and GRC manager copilot for this lab only.
Cover every module in this lab: assets, vendors, policies, risks, controls, treatments, evidence, testing, findings, requirements, exceptions, audits, KRI, BIA/BCM, crosswalk, traceability, reports, projects, and missions.
This is an IT-only lab. Use only IT GRC guidance aligned to this lab data.
Always include: Executive Assessment, Risk View, Control Mapping with control IDs, Evidence Requirements, Treatment Plan, and next operational actions.
Use precise compliance language for auditors, ISSOs, and compliance engineers.
${config.strictCitations
    ? 'When answering, cite only the provided knowledge sources using their SOURCE ID in square brackets, e.g., [POL-001§2]. Do not invent sources. If the provided sources do not cover the question, state that explicitly.'
    : ''}`;

// ---- DOM ELEMENTS ----
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const modelStatus = document.getElementById('model-status');
const kbStatus = document.getElementById('kb-status');
const micStatus = document.getElementById('mic-status');
const loadModelBtn = document.getElementById('load-model-btn');
const loadKbBtn = document.getElementById('load-kb-btn');
const micBtn = document.getElementById('mic-btn');
const clearBtn = document.getElementById('clear-btn');
const modelSelect = document.getElementById('model-select');
const statusEl = document.getElementById('status');
const progressContainer = document.getElementById('model-progress-container');
const progressBar = document.getElementById('model-progress-bar');
const progressText = document.getElementById('model-progress-text');

// ---- STATE ----
let engine = null;
let persistentMessages = [{ role: 'system', content: SYSTEM_PROMPT }];
let chunks = [];
let chunkVectors = [];
let termToChunks = new Map();
let isListening = false;
let recognition = null;

// ---- UTILITY FUNCTIONS ----
function estimateTokens(text) {
  return Math.ceil((text || '').length / 4);
}

function stripMarkdown(md) {
  return (md || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitIntoSentences(text) {
  const normalized = text.replace(/\s+/g, ' ');
  const matches = normalized.match(/[^.!?]+[.!?]+["']*(?=\s+|$)/g) || [normalized];
  return matches.map(s => s.trim()).filter(s => s.length > 8);
}

function chunkText(text, maxSentences) {
  const sentences = splitIntoSentences(text);
  const results = [];
  let current = [];
  for (const sentence of sentences) {
    current.push(sentence);
    if (current.length >= maxSentences) {
      results.push(current.join(' '));
      current = [];
    }
  }
  if (current.length > 0) results.push(current.join(' '));
  if (results.length === 0 && text.trim()) results.push(text.trim());
  return results;
}

function tokenize(text) {
  return (text || '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
}

function computeIdf(term, totalDocs) {
  const df = termToChunks.get(term)?.size || 0;
  return Math.log((totalDocs + 1) / (df + 1)) + 1;
}

function tfidfVector(tokens, totalDocs) {
  const termFreq = new Map();
  for (const token of tokens) {
    termFreq.set(token, (termFreq.get(token) || 0) + 1);
  }
  const vec = new Map();
  for (const [term, freq] of termFreq) {
    vec.set(term, freq * computeIdf(term, totalDocs));
  }
  return vec;
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0, magA = 0, magB = 0;
  for (const [term, weight] of vecA) {
    if (vecB.has(term)) dot += weight * vecB.get(term);
    magA += weight * weight;
  }
  for (const weight of vecB.values()) magB += weight * weight;
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

const MODEL_OPTIONS = [
  { value: 'Llama-3.1-8B-Instruct-q4f32_1-MLC', label: 'Llama 3.1 8B' },
  { value: 'Llama-3.2-1B-Instruct-q4f32_1-MLC', label: 'Llama 3.2 1B' },
  { value: 'Llama-3.2-3B-Instruct-q4f32_1-MLC', label: 'Llama 3.2 3B' },
  { value: 'Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC', label: 'Hermes 2 Pro Llama 3 8B' },
  { value: 'Phi-3.5-mini-instruct-q4f32_1-MLC', label: 'Phi 3.5 Mini' },
  { value: 'Phi-3-mini-4k-instruct-q4f32_1-MLC', label: 'Phi 3 Mini' },
  { value: 'gemma-2-2b-it-q4f32_1-MLC', label: 'Gemma 2 2B' },
  { value: 'Mistral-7B-Instruct-v0.3-q4f32_1-MLC', label: 'Mistral 7B v0.3' },
  { value: 'Hermes-2-Pro-Mistral-7B-q4f16_1-MLC', label: 'Hermes 2 Pro Mistral 7B' },
  { value: 'Qwen3-1.7B-q4f32_1-MLC', label: 'Qwen3 1.7B' },
  { value: 'Qwen3-4B-q4f32_1-MLC', label: 'Qwen3 4B' },
  { value: 'Qwen3-8B-q4f32_1-MLC', label: 'Qwen3 8B' },
];

function populateModelOptions() {
  if (!modelSelect) return;
  const existingValues = new Set(Array.from(modelSelect.options).map((opt) => opt.value));
  MODEL_OPTIONS.forEach(({ value, label }) => {
    if (existingValues.has(value)) return;
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    if (value === config.model) option.selected = true;
    modelSelect.appendChild(option);
  });
  if (!modelSelect.value) {
    modelSelect.value = config.model;
  }
}

// ---- MODEL PROGRESS UI ----
function showProgressBar() {
  progressContainer.classList.add('active');
  updateProgress(0);
}

function hideProgressBar() {
  progressContainer.classList.remove('active');
}

function updateProgress(progress) {
  const percentage = Math.round(progress * 100);
  progressBar.style.width = `${percentage}%`;
  progressText.textContent = `${percentage}%`;
}

// ---- KNOWLEDGE BASE LOADING & INDEXING ----
function buildIndex() {
  termToChunks.clear();
  chunkVectors = [];
  const totalDocs = chunks.length;
  chunks.forEach((chunk, idx) => {
    const tokens = tokenize(chunk.text);
    const vec = tfidfVector(tokens, totalDocs);
    chunkVectors.push(vec);
    for (const token of tokens) {
      if (!termToChunks.has(token)) termToChunks.set(token, new Set());
      termToChunks.get(token).add(idx);
    }
  });
}

async function loadKnowledgeBase() {
  kbStatus.textContent = 'Knowledge: loading…';
  kbStatus.className = 'badge';
  loadKbBtn.disabled = true;
  try {
    const res = await fetch(config.knowledgeManifest);
    if (!res.ok) throw new Error(`Manifest fetch failed: ${res.status}`);
    const manifest = await res.json();
    const docs = Array.isArray(manifest) ? manifest : manifest.documents;
    if (!docs) throw new Error('Manifest missing documents array');

    chunks = [];
    for (const doc of docs) {
      let text = '';
      if (doc.content) {
        text = doc.content;
      } else if (doc.path) {
        const url = new URL(doc.path, new URL(config.knowledgeRoot + '/', window.location.href)).href;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`Document ${doc.id} fetch failed: ${r.status}`);
        text = await r.text();
      } else {
        continue;
      }
      text = stripMarkdown(text);
      const subChunks = chunkText(text, config.chunkSize);
      subChunks.forEach((tc, i) => {
        chunks.push({
          id: `${doc.id}§${i}`,
          title: doc.title || doc.id,
          docId: doc.id,
          chunkIndex: i,
          text: tc,
        });
      });
    }

    if (chunks.length === 0) throw new Error('No usable knowledge chunks found');
    buildIndex();
    kbStatus.textContent = `Knowledge: ${chunks.length} chunks`;
    kbStatus.classList.add('ready');
    statusEl.textContent = 'Knowledge base ready. Load the WebLLM model to begin.';
  } catch (e) {
    kbStatus.textContent = 'Knowledge: error';
    kbStatus.classList.add('error');
    statusEl.textContent = 'KB error: ' + e.message;
    console.error(e);
    alert('Knowledge base load error:\n' + e.message);
  } finally {
    loadKbBtn.disabled = false;
  }
}

function search(query, topK, tokenBudget) {
  if (chunks.length === 0) return [];
  const queryTokens = tokenize(query);
  const queryVec = tfidfVector(queryTokens, chunks.length);
  const scored = [];
  for (let i = 0; i < chunks.length; i++) {
    const score = cosineSimilarity(queryVec, chunkVectors[i]);
    if (score > 0) scored.push({ idx: i, score });
  }
  scored.sort((a, b) => b.score - a.score);

  const selected = [];
  let usedTokens = 0;
  const limit = tokenBudget || inputTokenBudget;
  for (const item of scored.slice(0, topK)) {
    const chunk = chunks[item.idx];
    const chunkTokens = estimateTokens(chunk.text) + estimateTokens(chunk.title) + 12;
    if (usedTokens + chunkTokens > limit && selected.length > 0) break;
    selected.push(chunk);
    usedTokens += chunkTokens;
    if (usedTokens >= limit) break;
  }
  return selected;
}

// ---- WEBLLM MODEL LOADING ----
async function createEngineWithModel(modelId) {
  console.log(`Attempting to load model: ${modelId}`);
  showProgressBar();
  updateProgress(0);
  const engineInstance = await CreateMLCEngine(modelId, {
    initProgressCallback: (report) => {
      const progress = report.progress ?? 0;
      updateProgress(progress);
      const pct = (progress * 100).toFixed(1);
      modelStatus.textContent = `Model: loading ${pct}%`;
    },
  });
  hideProgressBar();
  console.log(`Model ${modelId} loaded successfully`);
  return engineInstance;
}

async function loadModel() {
  if (engine) return;
  if (!('gpu' in navigator)) {
    alert('WebGPU is required. Use Chrome/Edge 113+.');
    return;
  }

  const selectedModel = modelSelect.value || config.model;
  const modelsToTry = [selectedModel, ...config.fallbackModels.filter((id) => id !== selectedModel)];

  modelStatus.textContent = 'Model: preparing…';
  modelStatus.className = 'badge';
  loadModelBtn.disabled = true;

  let lastError = null;

  for (const modelId of modelsToTry) {
    try {
      engine = await createEngineWithModel(modelId);
      break;  // success
    } catch (e) {
      console.error(`Failed to load ${modelId}:`, e);
      lastError = e;
      modelStatus.textContent = `Model: failed (${modelId})`;
      hideProgressBar();
    }
  }

  if (!engine) {
    modelStatus.textContent = 'Model: error';
    modelStatus.classList.add('error');
    statusEl.textContent = `Model load error: ${lastError?.message || 'Unknown error'}. Please check supported models.`;
    loadModelBtn.disabled = false;
    sendBtn.disabled = true;
    micBtn.disabled = true;
    return;
  }

  modelStatus.textContent = 'Model: ready';
  modelStatus.classList.add('ready');
  sendBtn.disabled = false;
  micBtn.disabled = false;
  statusEl.textContent = 'Model ready. Ask a GRC question or use the microphone.';
}

// ---- CHAT LOGIC ----
function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = content;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function prunePersistentHistory() {
  const maxPersistTokens = Math.min(1200, Math.floor(inputTokenBudget * 0.6));
  while (persistentMessages.length > 1) {
    let total = estimateTokens(SYSTEM_PROMPT);
    for (let i = 1; i < persistentMessages.length; i++) {
      total += estimateTokens(persistentMessages[i].content);
    }
    if (total <= maxPersistTokens) break;
    persistentMessages.splice(1, 1);
    if (persistentMessages.length > 1 && persistentMessages[1]?.role === 'assistant') {
      persistentMessages.splice(1, 1);
    }
  }
}

async function sendMessage() {
  const userText = inputEl.value.trim();
  if (!userText || !engine) return;

  inputEl.value = '';
  appendMessage('user', userText);
  sendBtn.disabled = true;

  const relevant = search(userText, config.topK, inputTokenBudget - estimateTokens(userText) - 100);
  let contextText = '';
  if (relevant.length > 0) {
    contextText = 'Relevant knowledge sources (use these for your answer):\n';
    relevant.forEach(chunk => {
      contextText += `\n[SOURCE ID: ${chunk.id}] TITLE: ${chunk.title}\n${chunk.text}\n`;
    });
    contextText += `\nUser question: ${userText}`;
  } else {
    contextText = userText;
  }

  const fullMessages = [
    ...persistentMessages,
    { role: 'user', content: contextText },
  ];

  const assistantDiv = appendMessage('assistant', '');
  let fullResponse = '';

  try {
    const stream = await engine.chat.completions.create({
      messages: fullMessages,
      stream: config.streaming,
      temperature: config.temperature,
      max_tokens: config.maxOutputTokens,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content
        ?? chunk.choices?.[0]?.text
        ?? '';
      if (delta) {
        fullResponse += delta;
        assistantDiv.textContent = fullResponse;
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    }

    if (!fullResponse) fullResponse = '(No response generated.)';
    assistantDiv.textContent = fullResponse;

    persistentMessages.push({ role: 'user', content: userText });
    persistentMessages.push({ role: 'assistant', content: fullResponse });
    prunePersistentHistory();
  } catch (e) {
    console.error(e);
    assistantDiv.textContent = '⚠️ Error generating response: ' + e.message;
  } finally {
    sendBtn.disabled = false;
  }
}

// ---- MICROPHONE (Web Speech API) ----
function initSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    micStatus.textContent = 'Mic: not supported';
    micStatus.classList.add('error');
    micBtn.disabled = true;
    return;
  }
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isListening = true;
    micStatus.textContent = 'Mic: listening…';
    micBtn.classList.add('active');
  };
  recognition.onend = () => {
    isListening = false;
    micStatus.textContent = 'Mic: idle';
    micBtn.classList.remove('active');
  };
  recognition.onerror = (event) => {
    isListening = false;
    micStatus.textContent = 'Mic: error ' + event.error;
    micBtn.classList.remove('active');
  };
  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    inputEl.value = transcript.trim();
  };
}

function toggleMic() {
  if (!recognition || !engine) return;
  if (isListening) {
    recognition.stop();
  } else {
    try {
      recognition.start();
    } catch (e) {
      console.error(e);
      alert('Could not start microphone: ' + e.message);
    }
  }
}

// ---- EVENT LISTENERS ----
sendBtn.addEventListener('click', sendMessage);
loadModelBtn.addEventListener('click', loadModel);
loadKbBtn.addEventListener('click', loadKnowledgeBase);
micBtn.addEventListener('click', toggleMic);
clearBtn.addEventListener('click', () => {
  chatEl.innerHTML = '';
  persistentMessages = [{ role: 'system', content: SYSTEM_PROMPT }];
  appendMessage('system', 'Chat cleared. Conversation history has been reset.');
});
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

// ---- INITIALIZATION ----
(function init() {
  populateModelOptions();
  initSpeechRecognition();
  loadKnowledgeBase();

  if (!('gpu' in navigator)) {
    modelStatus.textContent = 'Model: WebGPU missing';
    modelStatus.classList.add('error');
    loadModelBtn.disabled = true;
    statusEl.textContent = 'WebGPU is not supported in this browser. WebLLM cannot run.';
  }
  appendMessage('system', 'Welcome. Load the WebLLM model to begin. Use the mic button for voice input.');
})();