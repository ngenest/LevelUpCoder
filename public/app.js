import { challenges, referenceSolutions, quotes } from './data.js';

const langSelect = document.getElementById('lang');
const resetBtn = document.getElementById('reset');
const levelDisplay = document.getElementById('level-display');
const meter = document.getElementById('meter');
const challengeTitle = document.getElementById('challenge-title');
const challengeTask = document.getElementById('challenge-task');
const challengeExample = document.getElementById('challenge-example');
const levelNumber = document.getElementById('level-number');
const answerBox = document.getElementById('answer');
const submitBtn = document.getElementById('submit');
const viewRefBtn = document.getElementById('view-ref');
const feedback = document.getElementById('feedback');
const referenceArea = document.getElementById('reference');
const refTitle = document.getElementById('ref-title');
const copyBtn = document.getElementById('copy-ref');
const ext = document.getElementById('ext');

const cookieKey = 'levelupcoder-progress';
const langExt = { python: 'py', javascript: 'js', java: 'java' };

let state = loadState();
let refVisible = false;

document.addEventListener('DOMContentLoaded', () => {
  langSelect.value = state.language;
  render();
});

langSelect.addEventListener('change', () => {
  state.language = langSelect.value;
  saveState();
  renderReference();
  ext.textContent = langExt[state.language];
});

resetBtn.addEventListener('click', () => {
  state.level = 1;
  state.language = 'python';
  state.answers = {};
  saveState();
  langSelect.value = 'python';
  render();
});

submitBtn.addEventListener('click', async () => {
  const answer = answerBox.value.trim();
  if (!answer) return showFeedback('Drop some code in first!', 'fail');
  const ref = getReferenceSnippet();
  const localJudge = similarityScore(answer, ref) > 0.55;
  let pass = localJudge;
  let judgeNote = 'Semantic matcher';
  if (!pass) {
    const chat = await askChatGPT(answer, ref, state.language);
    if (chat?.verdict === 'PASS') {
      pass = true;
      judgeNote = 'ChatGPT sanity check';
    } else if (chat?.verdict === 'FAIL') {
      judgeNote = 'ChatGPT nudge';
    }
  }

  if (pass) {
    showFeedback(`PASS! ${judgeNote} says you’re close enough. ${randomQuote()}`, 'pass');
    state.level = Math.min(50, state.level + 1);
    saveState();
    renderProgress();
    renderChallenge();
  } else {
    showFeedback(`Not yet. ${judgeNote} thinks you can tighten it a bit. ${guidanceHint()} ${randomQuote()}`, 'fail');
  }
});

viewRefBtn.addEventListener('click', () => {
  renderReference(!refVisible);
});

copyBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(referenceArea.textContent);
    copyBtn.textContent = 'Copied!';
    setTimeout(() => (copyBtn.textContent = 'Copy snippet'), 1400);
  } catch (e) {
    copyBtn.textContent = 'Clipboard unavailable';
    setTimeout(() => (copyBtn.textContent = 'Copy snippet'), 1400);
  }
});

answerBox.addEventListener('input', () => {
  state.answers[state.level] = answerBox.value;
  saveState();
});

function render() {
  renderProgress();
  renderChallenge();
  refVisible = false;
  renderReference(false);
  ext.textContent = langExt[state.language];
}

function renderProgress() {
  levelDisplay.textContent = state.level;
  levelNumber.textContent = state.level;
  const progress = ((state.level - 1) / 50) * 100;
  meter.style.width = `${Math.max(4, progress)}%`;
}

function renderChallenge() {
  const challenge = challenges.find(c => c.level === state.level) ?? challenges[challenges.length - 1];
  challengeTitle.textContent = challenge.title;
  challengeTask.textContent = challenge.task;
  challengeExample.textContent = challenge.example || '';
  answerBox.value = state.answers[state.level] ?? '';
  refTitle.textContent = `${challenge.title} · ${state.language}`;
}

function renderReference(reveal = refVisible) {
  refVisible = !!reveal;
  const ref = getReferenceSnippet();
  referenceArea.textContent = refVisible ? ref : 'Hidden until you peek or pass.';
  viewRefBtn.textContent = refVisible ? 'Hide reference' : 'Peek reference snippet';
}

function getReferenceSnippet() {
  const langRefs = referenceSolutions[state.language];
  return langRefs?.[state.level] ?? 'Reference missing for this level.';
}

function saveState() {
  const serialized = JSON.stringify(state);
  document.cookie = `${cookieKey}=${encodeURIComponent(serialized)}; path=/; max-age=${60 * 60 * 24 * 365};`;
}

function loadState() {
  const cookie = document.cookie.split(';').map(v => v.trim()).find(v => v.startsWith(`${cookieKey}=`));
  if (cookie) {
    try {
      return JSON.parse(decodeURIComponent(cookie.split('=')[1]));
    } catch (e) {
      return defaultState();
    }
  }
  return defaultState();
}

function defaultState() {
  return { level: 1, language: 'python', answers: {} };
}

function showFeedback(message, type) {
  feedback.textContent = message;
  feedback.classList.remove('hidden', 'pass', 'fail');
  feedback.classList.add(type);
  feedback.animate([
    { transform: 'scale(0.98)', opacity: 0.6 },
    { transform: 'scale(1)', opacity: 1 }
  ], { duration: 350, easing: 'ease-out' });
}

function similarityScore(a, b) {
  const clean = str => str.replace(/\s+/g, ' ').toLowerCase();
  const s1 = clean(a);
  const s2 = clean(b);
  const dist = levenshtein(s1, s2);
  const maxLen = Math.max(s1.length, s2.length) || 1;
  return 1 - dist / maxLen;
}

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp = new Array(n + 1).fill(0).map((_, i) => i);
  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const temp = dp[j];
      if (a[i - 1] === b[j - 1]) dp[j] = prev;
      else dp[j] = 1 + Math.min(prev, dp[j], dp[j - 1]);
      prev = temp;
    }
  }
  return dp[n];
}

function randomQuote() {
  const q = quotes[Math.floor(Math.random() * quotes.length)];
  return `Keep going! ${q}`;
}

function guidanceHint() {
  const challenge = challenges.find(c => c.level === state.level);
  if (!challenge) return 'Focus on clarity and edge cases.';
  return `Hint: ${challenge.task}`;
}

async function askChatGPT(answer, reference, language) {
  const key = window.OPENAI_API_KEY || window.env?.OPENAI_API_KEY;
  if (!key) return null;
  try {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are a strict but encouraging code reviewer that returns PASS or FAIL only.' },
          { role: 'user', content: `Language: ${language}. Reference solution: ${reference}. User answer: ${answer}. Is the user answer close enough to solve the same task? Respond with PASS or FAIL only.` }
        ],
        temperature: 0
      })
    });
    const json = await res.json();
    const content = json?.choices?.[0]?.message?.content?.trim();
    const verdict = content && content.toUpperCase().includes('PASS') ? 'PASS' : 'FAIL';
    return { verdict, raw: content };
  } catch (e) {
    return null;
  }
}
