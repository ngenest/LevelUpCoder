# LevelUpCoder

A lightweight, Brilliant-style coding dojo for the CodeBoxx x eBay “Level-Up” challenges. It ships with 50 progressively harder prompts, soft auto-grading, and client-side cookies so learners can return right where they left off.

## Quickstart

```bash
npm install
npm start
```

Then open http://localhost:3000.

### Docker

```bash
docker build -t levelupcoder .
docker run -p 3000:3000 levelupcoder
```

The container serves the static app via Express. It’s small, stateless, and ready for a Droplet.

## Features
- Choose Python, JavaScript, or Java; reference snippets swap automatically.
- Forgiving similarity check with optional ChatGPT fallback when `OPENAI_API_KEY` is available in the browser environment.
- Celebratory/encouraging feedback with CodeBoxx-inspired gradients.
- Client-side cookie persistence for level + draft answers.
- Prize milestone reminders at levels 30 and 50.

## Notes
- Reference solutions come from the included challenge documents (Python/Java/JavaScript).
- ChatGPT calls are optional; without a key, the app uses semantic similarity.
