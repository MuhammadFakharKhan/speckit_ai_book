---
id: 1
title: AI-driven Technical Book Constitution
stage: constitution
date: 2025-12-22
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: Claude Code
command: /sp.constitution
labels: [constitution,ai-book,rag-chatbot]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

Project: AI-Driven Technical Book with Embedded RAG Chatbot
Core principles:
Accuracy: all factual claims verified against primary sources (official docs, repos).
Clarity: content written for developers and ML practitioners.
Reproducibility: every step (content generation, build, RAG indexing) must be repeatable from the repo.
Interactivity: a working RAG chatbot embedded directly in the book.
Transparency: prompts and AI outputs are versioned and auditable.
Key standards:
Use Spec-Kit Plus and Claude Code for drafting and structuring the book.
Use Docusaurus to build the site and deploy to GitHub Pages.
Every factual claim includes a source link or citation.
The RAG chatbot must support two modes:
Global mode: answers questions using the full book.
Selection-only mode: answers strictly from user-selected text.
All content chunks must include provenance metadata (page, heading, offsets).
All secrets managed via environment variables only.
Constraints:
Book length: 20,000–40,000 words across 8–14 chapters.
Single GitHub repository containing book, prompts, RAG backend, and CI.
RAG stack: FastAPI backend, Neon Serverless Postgres (metadata), Qdrant Cloud (vectors), OpenAI Agents/ChatKit SDKs.
Must operate within free-tier limits and provide a local fallback.
No hard-coded keys; no persistent user data without opt-in.
Success criteria:
Docusaurus site builds locally and deploys to GitHub Pages via CI.
Claude Code prompts and outputs are included and reviewed by a human.
Embedded RAG chatbot works on the live site.
Selection-only retrieval is enforced and verifiable with citations.
A fresh clone of the repo can reproduce the site and a local RAG demo end-to-end.

## Response snapshot

Updated the project constitution at .specify/memory/constitution.md with specific principles and guidelines for the AI-Driven Technical Book with Embedded RAG Chatbot project. The new constitution includes:

1. Core Principles:
   - Accuracy and Verification: All factual claims must be verified against primary sources
   - Reproducibility and Transparency: Every step must be repeatable and auditable
   - Interactive Experience: Working RAG chatbot with global and selection-only modes
   - Technical Excellence: Using Spec-Kit Plus, Claude Code, Docusaurus, and specified RAG stack
   - Security and Privacy: Proper secret management and privacy protection
   - Compliance and Constraints: Following SDD methodology within specified constraints

2. Additional sections for constraints, development workflow, and governance

The constitution now properly reflects the project requirements and will guide all future development decisions.

## Outcome

- ✅ Impact: Updated constitution with project-specific principles and guidelines
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with AI book specific content
- 🔁 Next prompts: Ready to proceed with feature specification and planning
- 🧠 Reflection: Constitution now properly aligned with AI-driven technical book project requirements

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - Constitution properly updated with all required principles
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin feature specification based on new constitution