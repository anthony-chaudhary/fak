#!/usr/bin/env python3
"""Behavior tests for Markdown-to-JSON-LD extraction and idempotent injection."""

import gen_structured_data as structured


def test_faq_pairs_flattens_markdown_and_skips_non_questions():
    faq = """---
title: FAQ
---
## What does fak protect?

It protects **tool calls** and links to [the policy](https://example.com) while
preserving `snake_case` identifiers.

## Background

This section is deliberately not a question and must not become FAQ schema.

## Can it run offline?

- Yes, the policy floor and deterministic agent demo run without a model key.
"""

    pairs = structured.faq_pairs(faq)

    assert [question for question, _ in pairs] == [
        "What does fak protect?",
        "Can it run offline?",
    ]
    assert "**" not in pairs[0][1] and "](" not in pairs[0][1]
    assert "tool calls" in pairs[0][1] and "the policy" in pairs[0][1]
    schema = structured.build_faqpage(faq)
    assert schema["@type"] == "FAQPage"
    assert len(schema["mainEntity"]) == 2
    assert structured.faqpage_artifacts(faq) == []


def test_inject_replaces_existing_block_without_duplication():
    begin, end = "<!-- BEGIN -->", "<!-- END -->"
    original = f"# Page\n\n{begin}\nold\n{end}\n\nBody\n"
    block = f"{begin}\nnew\n{end}"

    once = structured.inject(original, begin, end, block, after_front_matter=True)
    twice = structured.inject(once, begin, end, block, after_front_matter=True)

    assert once == twice
    assert once.count(begin) == 1 and "old" not in once and "new" in once


def test_inject_places_new_markdown_block_after_front_matter():
    begin, end = "<!-- B -->", "<!-- E -->"
    text = "---\ntitle: X\n---\n# Heading\nBody\n"
    block = f"{begin}\npayload\n{end}"
    out = structured.inject(text, begin, end, block, after_front_matter=True)
    assert out.startswith("---\ntitle: X\n---\n\n" + block)

