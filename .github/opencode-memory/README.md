# Agent Memory

This directory contains context and instructions for the Claude Code autonomous agent that maintains this repository.

## Purpose

The agent periodically scans [Replicate Explore](https://replicate.com/explore) to discover new popular models and creates corresponding NodeTool nodes for them.

## Files

- `README.md` - This file
- `repository-context.md` - Overview of the repository structure
- `node-creation-guide.md` - Step-by-step instructions for creating new nodes
- `features.md` - Log of features added by the agent

## Related Workflows

- `.github/workflows/opencode-replicate-sync.yml` - Scheduled workflow that runs the Claude Code agent
