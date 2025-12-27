# ADR-003: Repository Structure and Monorepo Approach

## Status
Accepted

## Date
2025-12-23

## Context
We need to organize the codebase for an educational book that includes static site content, ROS 2 examples, AI integration components, and documentation. The structure should support maintainability, clear separation of concerns, and ease of development.

## Decision
We will use a monorepo approach with the following directory structure:
- **site/**: Docusaurus static site with documentation content
- **examples/**: ROS 2 example code and workspace
- **prompts/**: Claude Code prompts and outputs
- **history/**: Prompt History Records and development artifacts
- **.rag/**: Future RAG configuration and assets
- **.specify/**: Spec-Kit Plus configuration and templates

This approach centralizes all related code in a single repository while maintaining clear separation of concerns through directory structure.

## Consequences
### Positive
- Single source of truth for all related components
- Easier coordination between documentation and examples
- Simplified dependency management
- Clear visibility of all project components
- Simplified CI/CD setup
- Easier to maintain consistency across components

### Negative
- Larger repository size
- Potential complexity in access control if different teams handle different components
- Single point of failure for the entire project
- Potentially longer clone times

## Alternatives Considered
- **Multi-repo approach**: Would complicate coordination between documentation and examples
- **Separate documentation repo**: Would create disconnect between content and examples
- **Single flat structure**: Would lack clear organization and separation of concerns
- **Component-based repos**: Would increase complexity of cross-cutting changes

## References
- plan.md: Project Structure section
- spec.md: Requirements for integrated educational experience