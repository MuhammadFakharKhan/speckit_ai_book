# ADR-004: Future RAG Architecture for Educational Content

## Status
Proposed

## Date
2025-12-23

## Context
We plan to integrate Retrieval-Augmented Generation (RAG) capabilities into the educational book to provide an interactive experience with a chatbot that can answer questions about the content. This will enhance the learning experience by allowing students to ask questions and get context-aware responses.

## Decision
We will implement RAG functionality using the following technology stack:
- **Backend API**: FastAPI - Modern Python web framework with excellent async support
- **Database**: Neon Serverless Postgres - Serverless PostgreSQL for structured data storage
- **Vector Database**: Qdrant Cloud - Specialized vector database for semantic search
- **Integration**: Embedded directly in the book with Global mode and Selection-only mode
- **Citations**: Verifiable citations to ensure accuracy and source attribution

## Consequences
### Positive
- Rich interactive learning experience
- Accurate, context-aware responses based on book content
- Scalable architecture with serverless components
- Proper citation system for educational integrity
- Multiple interaction modes (global and selection-based)
- Industry-standard tools with good support

### Negative
- Additional complexity in deployment and maintenance
- Potential cost considerations for cloud services
- Need for content chunking and indexing processes
- Additional security considerations
- Dependency on external cloud services

## Alternatives Considered
- **Self-hosted solutions**: More complex deployment and maintenance
- **Different vector databases**: Qdrant was chosen for its cloud offering and Python SDK
- **Different backend frameworks**: FastAPI chosen for its async support and documentation
- **Simple search**: Would not provide the rich interactive experience needed
- **External chat service**: Would not be as well integrated with the book content

## References
- plan.md: Technical Context and Gate 3 (Interactive Experience)
- spec.md: Requirements for interactive learning experience