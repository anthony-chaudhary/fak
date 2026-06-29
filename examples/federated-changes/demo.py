#!/usr/bin/env python3
"""demo.py - the federated "what changed" feed + revoke walkthrough (#340).

Drives TWO independent `fak serve` instances (A and B) through the cross-agent
trust surface:

    GET  /v1/fak/changes   the "what changed" feed: a cursor-drained stream of
                           typed write Mutations + Revocations on THIS instance.
    POST /v1/fak/revoke    refute a poisoned external witness: every pooled entry
                           admitted under it is stranded, future re-admission under
                           it is refused, and the refutation lands on the feed with
                           an advanced trust_epoch (the integrity clock).

The load-bearing, honest picture (verified against the v0.34.0 routes):

  * Each `fak serve` observes its OWN process-global coherence bus. A write
    driven through A appears on A's feed; it does NOT auto-gossip to B. The feed
    is a per-instance, cursor-drained ledger, NOT a network consensus protocol.
  * "Federation" is the host (or a peer) DRAINING one instance's feed by cursor
    and REPLAYING the relevant event onto another instance. This demo does that
    explicitly: it propagates a refutation A -> B so the SAME revocation appears
    in BOTH feeds. That is the peer-trust property the routes give you, stated
    without overclaiming a consensus layer fak does not implement.

No model is needed: the feed and the refutation are pure kernel state, so the
load-bearing assertions fire with `--engine mock` alone. The exit code gates on
the kernel observations (the mutation appears, the revocation propagates).

Usage (normally via run.sh, which starts/stops the two kernels):
    FAK_A=http://127.0.0.1:8431 FAK_B=http://127.0.0.1:8432 python3 demo.py
    python3 demo.py --no-color
"""
import json
import os
import sys
import urllib.error
import urllib.request

A = os.environ.get("FAK_A", "http://127.0.0.1:8431")
B = os.environ.get("FAK_B", "http://127.0.0.1:8432")
COLOR = "--no-color" not in sys.argv and sys.stdout.isatty()


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if COLOR else s


def post(base, path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return r.status, json.loads(r.read().decode())


def changes(base, since=0):
    _, body = get(base, f"/v1/fak/changes?since={since}")
    return body


def syscall(base, tool, args, witness="", read_only=False):
    req = {"tool": tool, "arguments": args}
    if witness:
        req["witness"] = witness
    if read_only:
        req["read_only"] = True
    _, body = post(base, "/v1/fak/syscall", req)
    return body


def revoke(base, witness):
    _, body = post(base, "/v1/fak/revoke", {"witness": witness})
    return body


def find(events, kind, **match):
    for e in events:
        if e.get("kind") == kind and all(e.get(k) == v for k, v in match.items()):
            return e
    return None


def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        mark = c("32", "✓") if cond else c("31", "✗")
        print(f"  {mark} {label}{('   ' + detail) if detail else ''}")
        ok = ok and cond

    print(c("36;1", "fak federated 'what changed' feed + revoke walkthrough"))
    print(f"  instance A = {A}")
    print(f"  instance B = {B}\n")

    # 1) A drives a write-shaped tool. It COMPLETES (policy-allowed) -> a mutation
    #    lands on A's "what changed" feed, naming the tool and the world-version clock.
    wit_a = "commit-aaa111"
    va = syscall(A, "update_booking", {"id": "SFO-JFK-7", "seat": "14C"}, witness=wit_a)
    a_feed = changes(A)["events"]
    mut = find(a_feed, "mutation", tool="update_booking")
    check(
        "A: a write lands on A's change feed",
        va["verdict"]["kind"] == "ALLOW" and mut is not None,
        f"update_booking -> mutation seq={mut['seq'] if mut else '-'} world_ver={mut['world_ver'] if mut else '-'}",
    )

    # 2) B's feed is INDEPENDENT: A's mutation is NOT on it (no auto-gossip).
    b_feed_pre = changes(B)["events"]
    check(
        "B: A's write did NOT auto-propagate (per-instance feed)",
        find(b_feed_pre, "mutation", tool="update_booking") is None,
        f"B feed has {len(b_feed_pre)} event(s), none of them A's write",
    )

    # 3) B drives a DIFFERENT write under its own witness -> B's own feed entry.
    wit_b = "commit-bbb222"
    vb = syscall(B, "cancel_order", {"id": "ORD-99"}, witness=wit_b)
    b_feed = changes(B)["events"]
    check(
        "B: B's own write lands on B's feed",
        vb["verdict"]["kind"] == "ALLOW" and find(b_feed, "mutation", tool="cancel_order") is not None,
        "cancel_order -> mutation on B",
    )

    # 4) A REFUTES B's witness (commit-bbb222): a peer-signed refutation. It lands on
    #    A's feed as a revocation and advances A's trust_epoch (the integrity clock).
    rva = revoke(A, wit_b)
    a_feed2 = changes(A)["events"]
    rev_a = find(a_feed2, "revocation", witness=wit_b)
    check(
        "A: revoke refutes the witness on A (trust_epoch advances)",
        rev_a is not None and rva["trust_epoch"] >= 1,
        f"evicted={rva['evicted']} trust_epoch={rva['trust_epoch']}",
    )

    # 5) FEDERATION: the host drains A's feed, finds the revocation, and REPLAYS it
    #    onto B. The SAME refutation now appears in BOTH feeds -> the peer-trust
    #    property. (This replay is the federation mechanism; the routes do not gossip
    #    on their own.)
    rvb = revoke(B, rev_a["witness"])
    b_feed2 = changes(B)["events"]
    rev_b = find(b_feed2, "revocation", witness=wit_b)
    check(
        "B: the refutation propagates A -> B (revocation in BOTH feeds)",
        rev_b is not None and rvb["trust_epoch"] >= 1,
        f"B trust_epoch={rvb['trust_epoch']}",
    )

    # 6) Cursor drain: draining B PAST the revocation's seq returns nothing new for it
    #    (the at-least-once, monotone-cursor delivery contract).
    cur = changes(B)["cursor"]
    after = changes(B, since=cur)
    check(
        "cursor drain is monotone (nothing re-delivered past the cursor)",
        find(after["events"], "revocation", witness=wit_b) is None,
        f"drained B since cursor={cur} -> {len(after['events'])} new event(s)",
    )

    # 7) An empty witness is refused at the wire (fail-closed input check).
    status, body = post(A, "/v1/fak/revoke", {"witness": ""})
    check(
        "empty-witness revoke is refused (400)",
        status == 400,
        f"HTTP {status}: {body.get('error', {}).get('message', '')}",
    )

    print()
    if ok:
        print(c("32;1", "summary: federated change-feed + revoke walkthrough passed"))
        return 0
    print(c("31;1", "summary: FAILED - a federation assertion did not hold"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
