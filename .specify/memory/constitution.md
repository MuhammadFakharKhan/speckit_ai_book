<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.0
Modified principles: None (completely new constitution)
Added sections: All sections with AI-Driven Technical Book specific content
Removed sections: Template placeholders
Templates requiring updates: N/A (first version)
Follow-up TODOs: None
-->
# AI-Driven Technical Book with Embedded RAG Chatbot Constitution

## Core Principles

### I. Accuracy and Verification
All factual claims must be verified against primary sources (official documentation, repositories) with source links or citations included for every claim. Content must be written for developers and ML practitioners with clear, precise information.

### II. Reproducibility and Transparency
Every step (content generation, build, RAG indexing) must be repeatable from the repo. Prompts and AI outputs are versioned and auditable. All content chunks must include provenance metadata (page, heading, offsets).

### III. Interactive Experience
A working RAG chatbot must be embedded directly in the book supporting two modes: Global mode (answers questions using the full book) and Selection-only mode (answers strictly from user-selected text with verifiable citations).

### IV. Technical Excellence
Use Spec-Kit Plus and Claude Code for drafting and structuring the book. Use Docusaurus to build the site and deploy to GitHub Pages. The RAG stack uses FastAPI backend, Neon Serverless Postgres (metadata), Qdrant Cloud (vectors), OpenAI Agents/ChatKit SDKs.

### V. Security and Privacy
All secrets managed via environment variables only. No hard-coded keys. No persistent user data without opt-in. Must operate within free-tier limits and provide a local fallback.

### VI. Compliance and Constraints
Book length must be 20,000–40,000 words across 8–14 chapters. Single GitHub repository containing book, prompts, RAG backend, and CI. All development follows the Spec-Driven Development methodology with proper task breakdown and verification.

## Constraints and Boundaries

The project operates within strict constraints: single GitHub repository containing book, prompts, RAG backend, and CI pipeline. Book length limited to 20,000–40,000 words across 8–14 chapters. RAG stack must operate within free-tier limits with local fallback capability. No hard-coded keys or persistent user data without explicit opt-in. The project must support both global and selection-only RAG modes with verifiable citations.

## Development Workflow

Development follows the Spec-Kit Plus methodology with clear separation of concerns: Specification → Planning → Task breakdown → Implementation → Verification. All content generation uses Claude Code with human review. Docusaurus site builds locally and deploys to GitHub Pages via CI. All changes must be tested for both build integrity and RAG functionality. Every implementation must include proper error handling, observability, and documentation.

## Governance

This constitution supersedes all other development practices. Amendments require explicit documentation of changes, approval from project stakeholders, and migration plan for existing artifacts. All pull requests and reviews must verify compliance with these principles. Code complexity must be justified with clear benefits. Use this constitution as the primary guidance for all development decisions.

**Version**: 1.1.0 | **Ratified**: 2025-12-22 | **Last Amended**: 2025-12-22
