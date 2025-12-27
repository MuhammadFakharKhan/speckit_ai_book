# ADR-001: Frontend Technology Stack for Educational Book

## Status
Accepted

## Date
2025-12-23

## Context
We need to create an educational book for ROS 2 and humanoid robotics that will be accessible to students and developers. The frontend needs to support rich documentation with embedded code examples, be easy to maintain, and potentially integrate with future RAG capabilities.

## Decision
We will use the following technology stack for the frontend:
- **Framework**: Docusaurus v3 - Static site generator optimized for documentation
- **UI Library**: React - Component-based UI framework
- **Runtime**: Node.js - JavaScript runtime for development and build tools
- **Package Manager**: pnpm - Fast, disk space efficient package manager
- **Deployment**: GitHub Pages - Static hosting solution

## Consequences
### Positive
- Docusaurus provides excellent documentation features out of the box
- Rich Markdown support with embedded code examples
- Built-in search and navigation capabilities
- Strong community support and ecosystem
- Easy for technical writers to contribute content
- Fast loading static pages

### Negative
- Learning curve for contributors unfamiliar with React/JS ecosystem
- Potential complexity when integrating with future RAG features
- Dependency on Node.js ecosystem

## Alternatives Considered
- **GitBook**: More limited customization options, less active development
- **Sphinx + Read the Docs**: Python-focused, would require different skill set
- **Custom React app**: More complex to implement documentation features from scratch
- **Hugo + Academic theme**: Less integrated with JavaScript/ROS 2 ecosystem

## References
- plan.md: Technical Context section
- spec.md: Requirements for educational content delivery