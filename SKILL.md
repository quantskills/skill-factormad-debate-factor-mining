---
name: my-skill-name
description: "One sentence on what this skill does, with concrete capabilities. Use when an agent needs to <trigger scenario 1>, <trigger scenario 2>, or <trigger scenario 3> on portable agent platforms such as Claude Code, OpenClaw, or Codex-style skill systems."
quantSkills:
  organization: https://github.com/quantskills
  repository: quantskills/skill-my-skill-name
  repository_url: https://github.com/quantskills/skill-my-skill-name
  project_type: skill
  collection: <collection-name>
  license: GPL-3.0
  category: tooling            # trader-research / factor / data-api / replication / monitor / analyst / tooling
  tags: [tag-one, tag-two]     # 小写连字符,1-10 个
  platforms: [claude-code, codex, openclaw]
  language: zh-en
  status: draft                # draft / stable / deprecated
  validation_level: listed     # listed / runnable / verified(社区三级验证体系)
  maintainer_type: community   # official / community
  requires: []                 # dependent sibling skill-* or agent-* repository names
  summary_zh: 一句话中文简介(8-120 字符)
  summary_en: One-line English summary (8-200 chars)
---

# My Skill Name

Use this skill to <核心用途一句话>.

## Core Workflow

1. Step one.
2. Step two.

## Output Contract

Produce:

- `<output_file_1>`
- a concise report

## References

Use `references/source_boundary.md`.
