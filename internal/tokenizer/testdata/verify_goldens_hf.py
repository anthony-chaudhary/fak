#!/usr/bin/env python
"""Independent HF-oracle re-derivation of the in-tree encode goldens (#488).

The tokenizer leaf's acceptance gate is an offline corpus witness: for every
text, tokenizer.Encode must equal the HF fast tokenizer's input_ids id-for-id.
The Go tests assert Go.Encode == in-tree goldens against a real tokenizer.json,
but that alone cannot tell a *true HF oracle* from a circular Go==Go check if the
goldens were ever pasted from the Go encoder's own output. This script closes
that gap: it re-encodes the SAME corpora with the real HuggingFace fast tokenizer
(`tokenizers.Tokenizer`) and asserts id-for-id equality, so the goldens are
provably HF-derived — the one corruption the issue warns must never be faked.

It is a manual reproducer, NOT wired into `go test`. Run it on a host with the
HF model caches and `pip install tokenizers`:

    python internal/tokenizer/testdata/verify_goldens_hf.py

Corpora checked:
  * SmolLM2-135M-Instruct   — parsed straight out of tokenizer_test.go
                              (TestOptionalSmolLM2EncodeMatchesHFCorpus)
  * Qwen2.5-1.5B-Instruct   — testdata/qwen25_golden.tsv (base64(text)\\tids)

Exit 0 if every present corpus matches (or its tokenizer is absent — nothing to
check); exit 1 on any id-for-id mismatch.
"""
import base64, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_GO = os.path.join(HERE, "..", "tokenizer_test.go")
QWEN_TSV = os.path.join(HERE, "qwen25_golden.tsv")


def hub_glob(*parts):
    home = os.path.expanduser("~")
    m = glob.glob(os.path.join(home, ".cache", "huggingface", "hub", *parts))
    return m[0] if m else None


def go_unquote(s):
    # The corpus strings use only \\n \\t \\r \\" \\\\ plus literal UTF-8;
    # codecs.unicode_escape would mangle multibyte UTF-8, so unescape manually.
    out, i = [], 0
    simple = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(simple.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def parse_smollm2_cases():
    text = open(TEST_GO, encoding="utf-8").read()
    start = text.index("func TestOptionalSmolLM2EncodeMatchesHFCorpus")
    body = text[start:text.index("\n}", start)]
    pat = re.compile(r'\{"((?:[^"\\]|\\.)*)",\s*\[\]int\{([^}]*)\}\}')
    return [(go_unquote(m.group(1)),
             [int(x) for x in m.group(2).replace(",", " ").split()])
            for m in pat.finditer(body)]


def parse_qwen_cases():
    cases = []
    for line in open(QWEN_TSV, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        b64, ids = line.split("\t", 1)
        cases.append((base64.b64decode(b64).decode("utf-8"),
                      [int(x) for x in ids.split()]))
    return cases


def check(name, tokenizer_json, cases):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_json)
    bad = 0
    for txt, want in cases:
        got = tok.encode(txt, add_special_tokens=False).ids
        if got != want:
            bad += 1
            sys.stdout.write(f"  MISMATCH {txt!r}\n    hf  ={got}\n    gold={want}\n")
    print(f"{name}: {len(cases) - bad}/{len(cases)} match HF fast tokenizer "
          f"({os.path.relpath(tokenizer_json, os.path.expanduser('~'))})")
    return bad


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # CJK/emoji reprs on any console
    targets = [
        ("SmolLM2", hub_glob("models--HuggingFaceTB--SmolLM2-135M-Instruct",
                             "snapshots", "*", "tokenizer.json"), parse_smollm2_cases),
        ("Qwen2.5", hub_glob("models--Qwen--Qwen2.5-1.5B-Instruct",
                             "snapshots", "*", "tokenizer.json"), parse_qwen_cases),
    ]
    bad = 0
    checked = 0
    for name, tj, load in targets:
        if not tj:
            print(f"{name}: tokenizer.json not cached — skipping")
            continue
        checked += 1
        bad += check(name, tj, load())
    if checked == 0:
        print("no HF tokenizer caches present; nothing to verify")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
