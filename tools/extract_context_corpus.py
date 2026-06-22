#!/usr/bin/env python3
"""
extract_context_corpus.py — turn a Claude Code session JSONL transcript into a
corpus the fak security gates can run over.

A transcript-derived corpus captures what actually entered a model context window:
  - RESULT side  : every tool_result payload (the bytes the ctxmmu write-time
                   admission gate is designed to adjudicate).
  - CALL side    : every tool_use args object (what the preflight rung ladder
                   adjudicates before a call fires).

Schema reused from tools/session_audit.py:
  type=assistant : message.content = [ {type: tool_use, id, name, input}, ... ]
  type=user      : message.content = [ {type: tool_result, tool_use_id, content}, ... ]
                   where content is a str OR a list of {type:text, text:...} blocks.

Usage:
  python extract_context_corpus.py <session.jsonl> [--out corpus.json]
  python extract_context_corpus.py --glob "<dir>/*.jsonl" [--out corpus.json]   # aggregate
"""
import os
import json
import glob
import argparse
import collections


def text_of(content):
    """Flatten a tool_result `content` field to its text bytes."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
                elif "text" in blk and isinstance(blk["text"], str):
                    parts.append(blk["text"])
                elif blk.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join(parts)
    if isinstance(content, dict):
        return text_of(content.get("content"))
    return str(content)


def extract(paths):
    id_to_tool = {}        # tool_use_id -> tool name (built from assistant records)
    calls = []             # {tool, args}
    pending_results = []   # {tool_use_id, payload} resolved against id_to_tool after the pass
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msg = rec.get("message") or {}
                content = msg.get("content")
                if rec.get("type") == "assistant" and isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            tool = blk.get("name", "?")
                            tid = blk.get("id", "")
                            if tid:
                                id_to_tool[tid] = tool
                            args = blk.get("input", {})
                            try:
                                args_s = json.dumps(args, ensure_ascii=False)
                            except Exception:
                                args_s = "{}"
                            calls.append({"tool": tool, "args": args_s})
                elif rec.get("type") == "user" and isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_result":
                            tid = blk.get("tool_use_id", "")
                            payload = text_of(blk.get("content"))
                            pending_results.append({"tid": tid, "payload": payload})
    results = []
    for i, pr in enumerate(pending_results):
        tool = id_to_tool.get(pr["tid"], "unknown")
        payload = pr["payload"]
        results.append({
            "name": f"r{i:03d}-{tool}",
            "tool": tool,
            "payload": payload,
            "bytes": len(payload.encode("utf-8", "replace")),
        })
    return calls, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", help="path to one session.jsonl")
    ap.add_argument("--glob", help="glob over many transcripts (aggregate)")
    ap.add_argument("--out", default="corpus.json")
    args = ap.parse_args()

    if args.glob:
        paths = sorted(glob.glob(args.glob))
    elif args.session:
        paths = [args.session]
    else:
        ap.error("need a session path or --glob")

    calls, results = extract(paths)

    tool_mix = collections.Counter(c["tool"] for c in calls)
    total_result_bytes = sum(r["bytes"] for r in results)
    corpus = {
        "_provenance": "extracted from private Claude Code session transcript(s); review before committing.",
        "sources": [os.path.basename(p) for p in paths],
        "summary": {
            "n_calls": len(calls),
            "n_results": len(results),
            "total_result_bytes": total_result_bytes,
            "tool_mix": dict(tool_mix.most_common()),
        },
        "calls": calls,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=0)
    print(f"sources       : {corpus['sources']}")
    print(f"calls         : {len(calls)}")
    print(f"results       : {len(results)}")
    print(f"result bytes  : {total_result_bytes:,}")
    print(f"tool mix      : {dict(tool_mix.most_common(12))}")
    print(f"wrote         : {args.out}")


if __name__ == "__main__":
    main()
