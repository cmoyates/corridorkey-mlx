---
name: btca-local
description: Query local btca resources via CLI for source-first answers
---

## What I do

- Query locally-cloned resources using the btca CLI
- Provide source-first answers about technologies stored in .btca/ or ~/.local/share/btca/

## When to use me

Use this skill when you need information about technologies stored in the project's local btca resources.

## Getting resources

Check `btca.config.jsonc` for the list of available resources in this project.

## Commands

Ask a question about one or more resources:

```bash
# Single resource
btca ask --resource <resource> --question "<question>"

# Multiple resources
btca ask --resource svelte --resource effect --question "How do I integrate Effect with Svelte?"

# Using @mentions in the question
btca ask --question "@svelte @tailwind How do I style components?"
```

## Managing Resources

```bash
# Add a git resource
btca add https://github.com/owner/repo

# Add a local directory
btca add ./docs

# Remove a resource
btca remove <name>
```
