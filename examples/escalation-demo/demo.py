#!/usr/bin/env python3
"""
fak kernel — human-in-the-loop escalation demo
==============================================

`adjudication-demo/` shows that a denied call is refused at the boundary. This demo shows
what a well-built harness does *after* that refusal: a denied call must not dead-end with a
bare "no" — it should **escalate to a human queue** along the path the policy itself declares
(`safe_sinks`). Every shipped template carries a `safe_sinks` field; this is the example that
finally exercises it.

    harness ──POST /v1/fak/adjudicate {refund_payment}──▶  fak serve  (the kernel; capability floor)
        │                                                      │  adjudicate(policy, call):
        │   ◀── DENY (POLICY_BLOCK / TERMINAL) ────────────────┘    refund_payment in deny-map
        │
        ├─ catch the verdict (don't dead-end)
        ├─ resolve the declared safe_sink from the SAME policy  (transfer_to_human_agents)
        ├─ build a structured ticket: original call · reason code · REDACTED args
        ├──POST /v1/fak/adjudicate {transfer_to_human_agents}──▶ kernel ── ALLOW ──┐
        │     (the escalation route is itself adjudicated — it is part of the policy,
        │      not a side-channel; an un-sanctioned "human queue" tool would be denied too)
        ▼
    user sees: "I can't refund this myself — I've routed it to a human agent with ticket #…"

Why the fak-native `/v1/fak/adjudicate` path (no model) and not the OpenAI wire: a denied
call is refused *before the kernel decodes its arguments*, so the OpenAI-wire deny never hands
the args back to the client. The escalation ticket needs the original args (to redact them),
so the faithful harness is the "client runs its own tools" path: it holds the call it
proposed, submits it for a verdict, and on DENY still has the call to redact and escalate.
That also makes this demo deterministic and CI-usable — the deny is a pure function of
(policy, call), independent of any model. `adjudication-demo/` covers the real-model proposal
side; this covers the graceful-deny side.

The discipline this asserts (README.md): redaction applies to the *escalation payload too*.
The kernel refuses a denied call before decoding it, so it does NOT redact a denied call's
args — the harness must, using the SAME `redact_fields` the policy already declares. No secret
may leak through the escalation path.

Usage:
    examples/escalation-demo/run.sh                 # one command: build + serve + run
    python3 demo.py [--kernel URL] [--policy PATH] [--no-color]

Exit code: 0 iff all four steps hold (deny → catch+route → redacted ticket → helpful reply);
1 otherwise. CI-usable. Honors NO_COLOR (https://no-color.org).
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY = os.path.normpath(os.path.join(HERE, "..", "customer-support-readonly-policy.json"))

# A realistic agent turn: the customer asks for a refund, and the harness assembles the call
# it would make — carrying secret-shaped fields (ssn, token) alongside benign ones. These two
# sentinel values must NEVER appear in the escalation ticket or the user-facing reply.
DENIED_TOOL = "refund_payment"
SECRET_SSN = "123-45-6789"
SECRET_TOKEN = "EXAMPLE_FAKE_TOKEN_redact_me_not_a_real_key"
PROPOSED_ARGS = {
    "order_id": "ORD-4471",
    "amount_usd": 500,
    "customer_email": "dana@example.com",
    "ssn": SECRET_SSN,        # redact_fields → must be stripped before it leaves the harness
    "token": SECRET_TOKEN,    # redact_fields → ditto
}


def color(enabled):
    if not enabled:
        return {k: "" for k in ("g", "r", "y", "b", "d", "x")}
    return {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m",
            "b": "\033[36m", "d": "\033[2m", "x": "\033[0m"}


def post_json(url, obj):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def adjudicate(base, tool, arguments):
    """Submit one proposed call to the kernel and return its WireVerdict dict."""
    resp = post_json(base + "/v1/fak/adjudicate", {"tool": tool, "arguments": arguments})
    return resp.get("verdict", {}) or {}


def redact(args, fields):
    """Strip secret-shaped fields, mirroring the kernel's own TRANSFORM marker ([REDACTED]).
    Uses the policy's declared redact_fields — the escalation path obeys the SAME rule the
    call boundary does, so no secret rides out through the human queue."""
    out = dict(args)
    for f in fields:
        if f in out:
            out[f] = "[REDACTED]"
    return out


def main():
    ap = argparse.ArgumentParser(description="fak kernel human-in-the-loop escalation demo")
    base = os.environ.get("FAK_DEMO_KERNEL", "http://127.0.0.1:8080")
    ap.add_argument("--kernel", default=base)
    ap.add_argument("--policy", default=os.environ.get("FAK_DEMO_POLICY", DEFAULT_POLICY))
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    # UTF-8 the streams so ✓/✗/→ survive a Windows code-page console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    c = color(not args.no_color and not os.environ.get("NO_COLOR") and sys.stdout.isatty())
    base = args.kernel.rstrip("/")

    # The escalation policy is read from the SAME manifest the kernel is enforcing — the sink
    # and the redaction rule are not invented here, they are the policy's own declarations.
    try:
        with open(args.policy, encoding="utf-8") as f:
            policy = json.load(f)
    except OSError as e:
        print(f"{c['r']}✗ cannot read policy {args.policy}: {e}{c['x']}")
        return 1
    safe_sinks = policy.get("safe_sinks") or []
    redact_fields = policy.get("redact_fields") or []
    allow = set(policy.get("allow") or [])

    print(f"{c['b']}fak kernel — human-in-the-loop escalation demo{c['x']}  "
          f"{c['d']}kernel={base}  policy={os.path.basename(args.policy)}{c['x']}")
    print(f"{c['d']}  a denied call must not dead-end — it escalates along the policy's declared "
          f"safe_sink{c['x']}\n")

    if not safe_sinks:
        print(f"{c['r']}✗ policy declares no safe_sinks — nothing to escalate to{c['x']}")
        return 1

    fails = []

    # ── Step 1 — DENY: the kernel refuses the call at the capability boundary. ──────────────
    try:
        vd = adjudicate(base, DENIED_TOOL, PROPOSED_ARGS)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"{c['r']}✗ could not reach the kernel at {base}: {e}{c['x']}")
        print(f"{c['d']}  start it first, or use run.sh which builds + serves + runs.{c['x']}")
        return 1
    reason, disp = vd.get("reason"), vd.get("disposition")
    deny_ok = vd.get("kind") == "DENY" and reason == "POLICY_BLOCK"
    if deny_ok:
        print(f"  {c['g']}✓{c['x']} 1 kernel DENIED the call            "
              f"{c['d']}{DENIED_TOOL} → DENY ({reason}/{disp}){c['x']}")
    else:
        fails.append(f"step 1: expected {DENIED_TOOL} → DENY/POLICY_BLOCK, got {vd}")
        print(f"  {c['r']}✗ 1 expected DENY/POLICY_BLOCK, got {vd}{c['x']}")

    # ── Step 2 — CATCH + ROUTE: pick the declared safe_sink and prove the kernel SANCTIONS it.
    # The escalation route is part of the policy, not a side-channel: an un-sanctioned "human
    # queue" tool would itself be denied. We build the ticket first (step 3) so the route
    # carries the redacted payload.
    sink = safe_sinks[0]
    redacted_args = redact(PROPOSED_ARGS, redact_fields)
    ticket_id = f"HUMAN-{PROPOSED_ARGS['order_id']}"
    ticket = {
        "ticket_id": ticket_id,
        "queue": sink,
        "original_call": {"tool": DENIED_TOOL, "arguments": redacted_args},
        "reason_code": reason,                 # the kernel's named refusal code, carried verbatim
        "kernel_disposition": disp,
        "summary": "Kernel refused a payment-mutating call at the capability boundary; "
                   "a human teammate must decide.",
    }
    try:
        sink_vd = adjudicate(base, sink, ticket)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"{c['r']}✗ could not reach the kernel for the escalation call: {e}{c['x']}")
        return 1
    route_ok = sink in safe_sinks and sink in allow and sink_vd.get("kind") == "ALLOW"
    if route_ok:
        print(f"  {c['g']}✓{c['x']} 2 harness routed to the safe_sink    "
              f"{c['d']}{sink} → ALLOW (declared safe_sink + on the allow-list){c['x']}")
    else:
        fails.append(f"step 2: safe_sink {sink!r} must be declared, allow-listed, and ALLOW; "
                     f"got allow={sink in allow} verdict={sink_vd}")
        print(f"  {c['r']}✗ 2 escalation route not sanctioned: allow={sink in allow} "
              f"verdict={sink_vd}{c['x']}")

    # ── Step 3 — REDACTED TICKET: the human queue receives the original call + reason code,
    # with every secret stripped. We prove it by scanning the serialized ticket for the
    # sentinel secret values — they must be absent.
    serialized = json.dumps(ticket)
    leaked = [s for s in (SECRET_SSN, SECRET_TOKEN) if s in serialized]
    ticket_ok = (
        ticket["original_call"]["tool"] == DENIED_TOOL
        and ticket["reason_code"] == "POLICY_BLOCK"
        and redacted_args.get("ssn") == "[REDACTED]"
        and redacted_args.get("token") == "[REDACTED]"
        and redacted_args.get("order_id") == "ORD-4471"   # benign fields are preserved
        and not leaked
    )
    human_queue = []   # the "human queue": a structured ticket lands here, not a bare refusal
    if ticket_ok:
        human_queue.append(ticket)
        print(f"  {c['g']}✓{c['x']} 3 human queue got a redacted ticket "
              f"{c['d']}{ticket_id}: original={DENIED_TOOL} reason={reason} "
              f"ssn/token=[REDACTED]{c['x']}")
    else:
        fails.append(f"step 3: ticket malformed or leaked a secret (leaked={leaked})")
        print(f"  {c['r']}✗ 3 ticket malformed or leaked a secret: leaked={leaked}{c['x']}")

    # ── Step 4 — HELPFUL REPLY: the user gets routed, not stonewalled, and no secret leaks. ──
    user_msg = (f"I can't issue that refund myself — payment changes are routed to a human "
                f"teammate for approval. I've opened ticket {ticket_id} and a support agent "
                f"will follow up with you shortly.")
    reply_ok = (
        ticket_id in user_msg
        and "ticket" in user_msg.lower()
        and SECRET_SSN not in user_msg
        and SECRET_TOKEN not in user_msg
    )
    if reply_ok:
        print(f"  {c['g']}✓{c['x']} 4 user-facing reply stays helpful   {c['d']}\"{user_msg[:64]}…\"{c['x']}")
    else:
        fails.append("step 4: user reply missing ticket id, or leaked a secret")
        print(f"  {c['r']}✗ 4 user reply unhelpful or leaked a secret{c['x']}")

    print()
    if human_queue:
        print(f"{c['y']}ticket delivered to the human queue:{c['x']}")
        print(f"{c['d']}{json.dumps(human_queue[0], indent=2)}{c['x']}\n")

    if fails:
        print(f"{c['b']}summary:{c['x']} {c['r']}ESCALATION TEST FAILED{c['x']}  ·  " + "  ·  ".join(fails))
        return 1
    print(f"{c['b']}summary:{c['x']} {c['g']}escalation test passed{c['x']}  ·  "
          f"deny → catch → route → redacted ticket → helpful reply, all 4 steps held")
    print(f"{c['d']}  the load-bearing result: the kernel decided the deny; the harness routed it to the\n"
          f"  policy's DECLARED safe_sink; the same redact_fields scrubbed the escalation payload, so\n"
          f"  a refused call became a graceful human handoff with no secret leaked.{c['x']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
