---
title: Documentation Standards and Validation
description: Standards and validation procedures for Isaac ecosystem documentation following Docusaurus formatting guidelines
sidebar_position: 14
tags: [documentation, standards, docusaurus, formatting]
---

# Documentation Standards and Validation

## Introduction

This document outlines the standards and validation procedures for Isaac ecosystem documentation. All documentation must follow Docusaurus formatting guidelines and include proper frontmatter to ensure consistent rendering and organization.

## Docusaurus Documentation Standards

### Frontmatter Requirements

Every documentation file must include proper frontmatter with the following required fields:

```yaml
---
title: Page Title
description: Brief description of the page content
sidebar_position: Position number in sidebar
tags: [list, of, relevant, tags]
---
```

#### Required Frontmatter Fields

1. **title**: Clear, descriptive title for the page
2. **description**: Brief summary of the page content (under 160 characters)
3. **sidebar_position**: Numeric position for sidebar organization
4. **tags**: Comma-separated list of relevant tags for searchability

#### Optional Frontmatter Fields

```yaml
---
title: Page Title
description: Brief description
sidebar_position: 1
tags: [tag1, tag2, tag3]
# Optional fields:
slug: custom-url-slug
hide_table_of_contents: true  # Hide TOC for this page
---

```

### Content Structure

#### Standard Page Structure

```markdown
---
title: Page Title
description: Brief description of the page content
sidebar_position: Position
tags: [list, of, relevant, tags]
---

# Main Title

## Section 1

Content for section 1...

### Subsection 1.1

Detailed content for subsection...

## Section 2

Content for section 2...

### Code Examples

```python
# Example code block
def example_function():
    return "Hello World"
```

## Conclusion

Summary of the page content...
```

### Markdown Formatting Standards

#### Headers

- Use `#` for main title (only one per page)
- Use `##` for main sections
- Use `###` for subsections
- Use `####` for sub-subsections (avoid if possible)

#### Text Formatting

- Use **bold** for important terms and emphasis
- Use *italics* for technical terms and definitions
- Use `inline code` for code snippets and file names
- Use ```code blocks``` for multi-line code examples

#### Lists

Use consistent formatting for lists:

**Unordered lists:**
```markdown
- First item
- Second item
  - Nested item
- Third item
```

**Ordered lists:**
```markdown
1. First step
2. Second step
   1. Sub-step
   2. Another sub-step
3. Third step
```

### Code Block Standards

#### Language Specification

Always specify the programming language for syntax highlighting:

````markdown
```python
def example_function():
    """Example function with docstring."""
    return "Hello, World!"
```

```yaml
key: value
nested:
  - item1
  - item2
```

```bash
# Example bash command
ros2 run package executable
```
````

#### Code Block Best Practices

- Include descriptive comments in code examples
- Use realistic variable names that reflect actual usage
- Include error handling examples where appropriate
- Add context for complex code examples

## Validation Procedures

### Automated Validation Script

```python
#!/usr/bin/env python3
"""
Validation script for Isaac ecosystem documentation
"""
import os
import re
import yaml
from pathlib import Path

class DocumentationValidator:
    def __init__(self, docs_dir):
        self.docs_dir = Path(docs_dir)
        self.errors = []
        self.warnings = []

    def validate_all_docs(self):
        """
        Validate all documentation files in the directory
        """
        md_files = list(self.docs_dir.rglob("*.md"))
        total_files = len(md_files)
        valid_files = 0

        print(f"Validating {total_files} documentation files...")

        for md_file in md_files:
            if self.validate_file(md_file):
                valid_files += 1
            else:
                print(f"  ❌ {md_file}")

        print(f"\nValidation complete: {valid_files}/{total_files} files valid")
        if self.errors:
            print(f"Found {len(self.errors)} errors")
        if self.warnings:
            print(f"Found {len(self.warnings)} warnings")

        return valid_files == total_files

    def validate_file(self, file_path):
        """
        Validate a single documentation file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split frontmatter and content
            parts = self.split_frontmatter(content)
            if not parts:
                self.errors.append(f"{file_path}: No frontmatter found")
                return False

            frontmatter, body = parts

            # Validate frontmatter
            frontmatter_valid = self.validate_frontmatter(frontmatter, file_path)

            # Validate content
            content_valid = self.validate_content(body, file_path)

            return frontmatter_valid and content_valid

        except Exception as e:
            self.errors.append(f"{file_path}: Error reading file - {str(e)}")
            return False

    def split_frontmatter(self, content):
        """
        Split content into frontmatter and body
        """
        lines = content.split('\n')
        if len(lines) < 2 or lines[0] != '---':
            return None

        # Find end of frontmatter
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                frontmatter_str = '\n'.join(lines[1:i])
                body = '\n'.join(lines[i+1:])
                return frontmatter_str, body

        return None

    def validate_frontmatter(self, frontmatter_str, file_path):
        """
        Validate frontmatter structure and required fields
        """
        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            self.errors.append(f"{file_path}: Invalid YAML in frontmatter - {str(e)}")
            return False

        # Check required fields
        required_fields = ['title', 'description', 'sidebar_position', 'tags']
        missing_fields = [field for field in required_fields if field not in frontmatter]

        if missing_fields:
            self.errors.append(f"{file_path}: Missing required fields: {', '.join(missing_fields)}")
            return False

        # Validate field types
        if not isinstance(frontmatter['title'], str):
            self.errors.append(f"{file_path}: Title must be a string")
            return False

        if not isinstance(frontmatter['description'], str):
            self.errors.append(f"{file_path}: Description must be a string")
            return False

        if not isinstance(frontmatter['sidebar_position'], (int, float)):
            self.errors.append(f"{file_path}: Sidebar position must be a number")
            return False

        if not isinstance(frontmatter['tags'], list):
            self.errors.append(f"{file_path}: Tags must be a list")
            return False

        # Validate description length
        if len(frontmatter['description']) > 160:
            self.warnings.append(f"{file_path}: Description is too long ({len(frontmatter['description'])} chars, max 160)")

        # Validate tags
        for tag in frontmatter['tags']:
            if not isinstance(tag, str):
                self.errors.append(f"{file_path}: All tags must be strings")
                return False

        return True

    def validate_content(self, content, file_path):
        """
        Validate content structure and formatting
        """
        lines = content.split('\n')
        valid = True

        # Check for main title (only one H1)
        h1_count = 0
        for line in lines:
            if line.strip().startswith('# ') and not line.strip().startswith('##'):
                h1_count += 1

        if h1_count != 1:
            self.errors.append(f"{file_path}: Must have exactly one main title (H1), found {h1_count}")
            valid = False

        # Check header hierarchy
        if not self.validate_header_hierarchy(lines, file_path):
            valid = False

        # Check for proper code block formatting
        if not self.validate_code_blocks(lines, file_path):
            valid = False

        return valid

    def validate_header_hierarchy(self, lines, file_path):
        """
        Validate that headers follow proper hierarchy (H1 -> H2 -> H3)
        """
        current_level = 0
        valid = True

        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Count header level
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break

                # Validate hierarchy
                if level > current_level + 1:
                    self.errors.append(f"{file_path}: Line {i+1}: Header level {level} skips level {current_level + 1}")
                    valid = False
                elif level <= current_level:
                    current_level = level

        return valid

    def validate_code_blocks(self, lines, file_path):
        """
        Validate code block formatting
        """
        in_code_block = False
        valid = True

        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                else:
                    # Start of code block - check for language specification
                    parts = line.strip().split('```')
                    if len(parts) > 1 and parts[1].strip():
                        # Has language specification
                        pass
                    else:
                        self.warnings.append(f"{file_path}: Line {i+1}: Code block should specify language")
                    in_code_block = True

        return valid

def main():
    """
    Main function to run validation
    """
    docs_dir = "site/docs"  # Adjust path as needed

    if not os.path.exists(docs_dir):
        print(f"Error: Documentation directory {docs_dir} does not exist")
        return 1

    validator = DocumentationValidator(docs_dir)
    is_valid = validator.validate_all_docs()

    if not is_valid:
        print("\n❌ Documentation validation failed")
        return 1
    else:
        print("\n✅ All documentation is valid")
        return 0

if __name__ == "__main__":
    exit(main())
```

### Manual Validation Checklist

#### Frontmatter Validation

- [ ] Title is clear and descriptive
- [ ] Description is under 160 characters
- [ ] Sidebar position is set appropriately
- [ ] Tags are relevant and properly formatted
- [ ] No extra fields that shouldn't be there

#### Content Validation

- [ ] Only one main title (H1) per page
- [ ] Headers follow proper hierarchy (H1 → H2 → H3)
- [ ] Code blocks specify language for syntax highlighting
- [ ] Images have alt text
- [ ] Links are properly formatted
- [ ] No hard tabs (use spaces instead)
- [ ] Consistent indentation

#### Technical Content Validation

- [ ] Code examples are accurate and tested
- [ ] Commands are complete and executable
- [ ] File paths are correct
- [ ] Technical terms are defined or linked
- [ ] Cross-references are accurate

## Common Issues and Fixes

### Frontmatter Issues

#### Missing Frontmatter

**Problem:**
```markdown
# Page Title

Content here...
```

**Fix:**
```markdown
---
title: Page Title
description: Brief description of the page content
sidebar_position: 1
tags: [tag1, tag2, tag3]
---

# Page Title

Content here...
```

#### Invalid YAML

**Problem:**
```yaml
---
title: My Title with "quotes"
description: Description with special characters: &
sidebar_position: 1
tags: [list, of, tags]
---
```

**Fix:**
```yaml
---
title: "My Title with quotes"
description: "Description with special characters: &"
sidebar_position: 1
tags: [list, of, tags]
---
```

### Content Structure Issues

#### Improper Header Hierarchy

**Problem:**
```markdown
# Main Title

#### Subsection (skips H3)
```

**Fix:**
```markdown
# Main Title

## Section

### Subsection
```

#### Multiple H1 Headers

**Problem:**
```markdown
# Main Title

# Another Title (incorrect)
```

**Fix:**
```markdown
# Main Title

## Section Title
```

## Best Practices

### Writing Standards

1. **Clarity**: Use clear, concise language
2. **Consistency**: Maintain consistent terminology
3. **Completeness**: Provide complete examples
4. **Accuracy**: Ensure all information is correct
5. **Accessibility**: Write for different skill levels

### Technical Documentation Standards

1. **Examples**: Include working code examples
2. **Context**: Provide sufficient context for examples
3. **Error Handling**: Include error handling information
4. **Dependencies**: Document any dependencies
5. **Versioning**: Specify version requirements when relevant

### Formatting Standards

1. **Consistency**: Use consistent formatting throughout
2. **Readability**: Prioritize readability over complexity
3. **Structure**: Follow logical document structure
4. **Navigation**: Ensure easy navigation between related topics
5. **Searchability**: Use appropriate tags for searchability

## Validation Tools

### Built-in Docusaurus Validation

Docusaurus provides built-in validation when building the site:

```bash
# Build the site to validate documentation
cd site
npm run build

# Serve locally to check rendering
npm run serve
```

### Linting Tools

Consider using markdown linting tools:

```bash
# Install markdownlint
npm install -g markdownlint-cli

# Validate all markdown files
markdownlint site/docs/**/*.md
```

## Conclusion

Following these documentation standards ensures consistent, high-quality documentation for the Isaac ecosystem. Regular validation helps maintain quality and prevents common formatting issues. By adhering to these standards, contributors can create documentation that is both user-friendly and technically accurate.