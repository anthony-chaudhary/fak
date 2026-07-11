"""
fak — Hugging Face Space demo (Gradio, Docker SDK).

This app is a THIN WRAPPER over three commands that fak already ships and that
CLAIMS.md witnesses — nothing here computes a result of its own, so the demo
cannot over-claim. Each tab prints the literal command it ran, the raw stdout,
and the one honest fence that bounds what that output proves:

  1. Policy floor      → `fak preflight --policy examples/... --tool <t> --args {}`
                         DENY (POLICY_BLOCK) for refund_payment; ALLOW for search_kb.
                         Witness: CLAIMS.md "Deployable capability floor".
  2. Provable deletion → `deletioncert -selfcheck`
                         evicted == never-saw (max|Δ|=0), cert minted + tamper-rejected.
                         Witness: CLAIMS.md "Provable-deletion certificate".
  3. Turn tax          → `turntaxdemo -print -suite turntax-airline`
                         a tuned 2026 SOTA agent pays 5 forced round-trips; fak stays 0.
                         Witness: cmd/turntaxdemo, testdata/turntax/*.json.

All three run offline: no API key, no model weights, no GPU. The heavy proof
(the HuggingFace-oracle parity behind the deletion claim) lives in the repo's
`go test ./internal/model`; this Space is the 60-second front door to it.
"""

import os
import shlex
import subprocess

import gradio as gr

# The app's working directory inside the container. The Dockerfile copies the
# fak binaries, the policy `examples/` dir, and `testdata/turntax/` here, so the
# relative paths below resolve exactly as they do from a repo checkout.
APP_DIR = os.path.dirname(os.path.abspath(__file__))

POLICY = "examples/customer-support-readonly-policy.json"

# One-line honest fences, shown verbatim under each result so the bound travels
# WITH the output and is never buried in prose.
FENCE_POLICY = (
    "Fence: this is the structural admission floor — a declarative, version-tagged "
    "JSON policy loaded at runtime (`--policy FILE`), not a compiled-in rule. It "
    "decides WHICH tools may be called; it is not a prompt-injection classifier "
    "(that detector is ~100% evadable by design and is explicitly not the floor)."
)
FENCE_DELETION = (
    "Fence: the certificate is a self-signed v1 receipt — it attests the integrity "
    "of the recorded facts, not independence from the recorder; `evicted_count` is a "
    "self-report; the bound max|Δ|=0 is a signed string here. The numeric "
    "bit-exactness is proven separately against a HuggingFace oracle in "
    "`go test ./internal/model` (cos=1.000000, KV-evict max|Δ|=0)."
)
FENCE_TURNTAX = (
    "Fence: the honest headline is fak vs a TUNED 2026 SOTA agent (the forced "
    "round-trips it cannot elide), never a naive baseline. The safety floor (poison "
    "paged out, destructive op refused) is a SEPARATE axis, never folded into the "
    "turn count. fak is not a faster token engine — the win is deleted round-trips."
)


def _run(argv, extra_env=None):
    """Run a command in the app dir, return 'cmd\\n\\n<combined output>'.

    stdout+stderr are merged so a non-zero exit still shows its diagnostic. The
    printed command line is the exact thing a reader can run from a checkout.
    """
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    shown = " ".join(shlex.quote(a) for a in argv)
    try:
        proc = subprocess.run(
            argv,
            cwd=APP_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return f"$ {shown}\n\n(command not found — is the binary built into the image?)"
    except subprocess.TimeoutExpired:
        return f"$ {shown}\n\n(timed out after 60s)"
    out = (proc.stdout or "") + (proc.stderr or "")
    return f"$ {shown}\n\n{out.rstrip()}"


def preflight(tool):
    # `{}` args: the deny is structural (tool identity vs the policy), not argument-shaped.
    body = _run(["./fak", "preflight", "--policy", POLICY, "--tool", tool, "--args", "{}"])
    return f"{body}\n\n{FENCE_POLICY}"


def deletion():
    body = _run(["./deletioncert", "-selfcheck"])
    return f"{body}\n\n{FENCE_DELETION}"


def turntax():
    # NO_COLOR so the terminal two-column diff renders as clean text in the browser.
    body = _run(["./turntaxdemo", "-print", "-suite", "turntax-airline"], {"NO_COLOR": "1"})
    return f"{body}\n\n{FENCE_TURNTAX}"


INTRO = """
# fak — the agent kernel, proven offline

`fak` treats every tool call like a **syscall**: it is adjudicated at a real
boundary before it runs, and the kernel owns the KV cache so a poisoned result
can be *provably* evicted. This Space runs three of fak's own witnesses live —
**no API key, no model weights, no GPU.**

Every command below is one `CLAIMS.md`-witnessed command; the tab prints the
literal invocation, the raw output, and the honest fence that bounds it.
The full HuggingFace-oracle parity proof (cos=1.000000, final-logits
max|Δ|≈4.4e-5, KV-evict max|Δ|=0) is in `go test ./internal/model`.

Repo: https://github.com/anthony-chaudhary/fak ·
Notebooks: [Colab quickstart](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb)
"""

with gr.Blocks(title="fak — agent kernel, proven offline") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("1 · Policy floor"):
        gr.Markdown(
            "A read-only customer-support policy. `refund_payment` is denied by "
            "**structure** (`DENY (POLICY_BLOCK)`); `search_kb` is allowed — the same "
            "policy file, two verdicts."
        )
        with gr.Row():
            deny_btn = gr.Button("refund_payment → expect DENY", variant="stop")
            allow_btn = gr.Button("search_kb → expect ALLOW", variant="primary")
        policy_out = gr.Textbox(label="fak preflight", lines=14, show_copy_button=True)
        deny_btn.click(lambda: preflight("refund_payment"), outputs=policy_out)
        allow_btn.click(lambda: preflight("search_kb"), outputs=policy_out)

    with gr.Tab("2 · Provable deletion"):
        gr.Markdown(
            "Evict a secret span from the kernel-owned KV cache, then prove the "
            "surviving context is **byte-identical** to a run that never saw it "
            "(max|Δ|=0) — mint a tamper-evident certificate and watch verification "
            "**fail closed** when the cert or its journal is forged."
        )
        del_btn = gr.Button("Run deletioncert -selfcheck", variant="primary")
        del_out = gr.Textbox(label="deletioncert -selfcheck", lines=22, show_copy_button=True)
        del_btn.click(deletion, outputs=del_out)

    with gr.Tab("3 · Turn tax"):
        gr.Markdown(
            "Replay one airline-support tool-call trace through the real kernel. A "
            "tuned 2026 SOTA agent is **forced** into 5 recovery round-trips; fak "
            "resolves each condition inside the syscall it arrived on and stays flat "
            "at **0** extra round-trips."
        )
        tt_btn = gr.Button("Run turntaxdemo -print", variant="primary")
        tt_out = gr.Textbox(label="turntaxdemo -print", lines=22, show_copy_button=True)
        tt_btn.click(turntax, outputs=tt_out)

if __name__ == "__main__":
    # HF Docker Spaces expose port 7860; bind all interfaces (containers need
    # 0.0.0.0, not loopback).
    demo.launch(server_name="0.0.0.0", server_port=7860)
