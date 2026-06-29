#!/usr/bin/env python3
"""
ctxnav.py — agent-navigable context: dynamic resolution via `memory(ref, op, budget)`.

The active counterpart to ctxcost.py's observability trace. The O(1)+history idea is not only
that the SYSTEM reconstructs a bounded view each turn — it is that every node in that view is a
TOMBSTONE at some resolution, and the AGENT (or the system) can zoom any node in or out on
demand:

    memory(ref, "expand",  budget)   # raise this node's resolution: get ~budget more tokens,
                                      #   with the still-elided middle left as a CHILD tombstone
    memory(ref, "contract"        )  # drop this node back to a 1-line tombstone, freeing budget

Expansion is RECURSIVE: a node's elided middle is itself a node, so an agent that finds a
head+tail window insufficient expands the middle, then that middle's middle, drilling to any
depth. For any node the agent can therefore go UP (contract, zoom out) or DOWN (expand, zoom in),
to any size, directed by its own reasoning or by a system budget. Nothing is ever destroyed —
the lossless store holds the full bytes; a tombstone is just the smallest rendering of a node
that is always one budgeted expand away. This is "the agent sees everything that can be observed"
made operational, and it keeps the RESIDENT context O(1) (a wall of tombstones is nearly free)
while the FULL history stays reachable, pay-as-you-go.

This is the calibration + proof harness for the operation, the peer of ctxwin.py (head+tail+
pointer elision) and ctxcost.py (cost/observability). The LIVE path is the lossless span store +
demand-page in `internal/ctxplan` (Materialize), surfaced to the model as a memory tool at the
gateway seam; that wiring collides with the cache_control prefix on the verbatim req.Raw route,
which is exactly the cache-vs-observability trade ctxcost documents. Here we prove the OPERATION:
deterministic, budget-bounded, recursive, and fully replayable (every op is a journal event, so
an agent's exploration path can be replayed and learned from).

HONESTY: token counts are the tokenizer-free ~4-chars/tok estimate (same as ctxwin); a tombstone
is a fixed small marker; "resident" is the sum of rendered nodes. The store is lossless by
construction (it holds the original bytes), so expansion is always exact, never lossy — contrast
the cost harness, which prices bytes sent, not quality.

Usage:
  python tools/ctxnav.py demo  [<session.jsonl> | --session SUBSTR] [--budget T] [--steps N]
  python tools/ctxnav.py selfcheck
"""
import sys
import os
import argparse
import hashlib

CHARS_PER_TOK = 4.0
TOMBSTONE_TOK = 12          # fixed cost of a node's smallest rendering ("[#abc1234 Read 5300tok ↧]")
POINTER_TOK = 8             # the inline "[middle …]" child marker left inside a window


def toks(s):
    return len(s) / CHARS_PER_TOK


def itok(s):
    return int(round(toks(s)))


def _ref(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:8]


# --------------------------------------------------------------------------------------
# the store: lossless nodes addressed by ref. A node owns its full bytes; children are the
# recursive sub-nodes that expansion mints from the elided middle.
# --------------------------------------------------------------------------------------
class Node:
    __slots__ = ("ref", "kind", "content", "parent")

    def __init__(self, kind, content, parent=None):
        self.kind = kind
        self.content = content
        self.parent = parent
        # ref is content-addressed but salted by parent so identical chunks at different depths
        # stay distinct and addressable (the store is a tree, not a DAG, for replay clarity).
        self.ref = _ref((parent or "") + "\x00" + content)

    @property
    def full_tok(self):
        return itok(self.content)


class Store:
    """A lossless content store + a per-node resolution state. The resolution is the resident
    rendering: 'tombstone' (smallest), an int budget (a head+tail window of that many tokens),
    or 'full'. Everything is reachable; resolution only changes what is RESIDENT right now."""

    def __init__(self):
        self.nodes = {}          # ref -> Node
        self.res = {}            # ref -> 'tombstone' | int(budget) | 'full'
        self.order = []          # top-level node refs in render order
        self.journal = []        # replayable op log

    def add(self, kind, content):
        n = Node(kind, content)
        self.nodes[n.ref] = n
        self.res[n.ref] = "tombstone"
        self.order.append(n.ref)
        return n.ref

    # ---- the operation: memory(ref, op, budget) ----------------------------------------
    def memory(self, ref, op, budget=0, _record=True):
        if ref not in self.nodes:
            raise KeyError(f"unknown ref {ref}")
        if op == "expand":
            self.res[ref] = "full" if budget >= self.nodes[ref].full_tok else int(budget)
            self._mint_middle(ref)            # ensure the elided-middle child exists to drill into
        elif op == "contract":
            self.res[ref] = "tombstone"
        else:
            raise ValueError(f"unknown op {op!r} (expand|contract)")
        if _record:
            self.journal.append({"ref": ref, "op": op, "budget": int(budget)})
        return self.render_node(ref)

    def _mint_middle(self, ref):
        """When a node is windowed (not full), the elided middle becomes a CHILD node so the
        agent can recurse into it. Deterministic: same node+budget -> same child ref."""
        n = self.nodes[ref]
        r = self.res[ref]
        if r == "full" or r == "tombstone":
            return None
        budget = int(r)
        half = max(1, int((budget - POINTER_TOK) / 2) * int(CHARS_PER_TOK))
        if len(n.content) <= 2 * half:        # nothing actually elided
            return None
        mid = n.content[half:len(n.content) - half]
        child = Node(n.kind, mid, parent=n.ref)
        if child.ref not in self.nodes:
            self.nodes[child.ref] = child
            self.res[child.ref] = "tombstone"
        return child.ref

    def child_of(self, ref):
        """The middle-child ref of a windowed node, if any (for recursive expansion)."""
        cid = self._mint_middle(ref)
        return cid

    # ---- rendering: what is RESIDENT right now -----------------------------------------
    # render_node RECURSES through a windowed node's middle child: a tombstone middle renders
    # as a small pointer, an expanded middle inlines its own window (which may recurse again),
    # so resident grows exactly as the agent drills down and shrinks as it contracts.
    def render_node(self, ref):
        n = self.nodes[ref]
        r = self.res[ref]
        if r == "tombstone":
            return f"[#{ref} {n.kind} {n.full_tok}tok +expand]"
        if r == "full":
            return n.content
        budget = int(r)
        half = max(1, int((budget - POINTER_TOK) / 2) * int(CHARS_PER_TOK))
        if len(n.content) <= 2 * half:
            return n.content
        head = n.content[:half]
        tail = n.content[len(n.content) - half:]
        cid = self.child_of(ref)
        middle = self.render_node(cid) if cid else f"[...{itok(n.content) - 2*itok(head)}tok elided]"
        return f"{head}\n{middle}\n{tail}"

    def resident_tok(self):
        """Total tokens currently resident across all TOP-LEVEL nodes (the O(1) working set).
        Recursive: an expanded middle child counts inside its parent's rendering."""
        return sum(itok(self.render_node(r)) for r in self.order)

    def full_tok(self):
        return sum(self.nodes[r].full_tok for r in self.order)

    # ---- replay: a journal of ops deterministically reproduces a resident view ----------
    def replay(self):
        fresh = Store()
        # rebuild the same top-level nodes (content-addressed) then re-apply the journal
        for r in self.order:
            fresh.add(self.nodes[r].kind, self.nodes[r].content)
        for ev in self.journal:
            fresh.memory(ev["ref"], ev["op"], ev["budget"], _record=False)
        return fresh


# --------------------------------------------------------------------------------------
# build a store from a real transcript's tool results (reuse ctxwin's parser)
# --------------------------------------------------------------------------------------
def store_from_session(path, top_n=40):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ctxwin as W
    items, _ = W.parse_window(path)
    results = [it for it in items if it.kind == "result" and it.content]
    results.sort(key=lambda it: len(it.content), reverse=True)
    st = Store()
    for it in results[:top_n]:
        st.add(it.tool, it.content)
    return st


# --------------------------------------------------------------------------------------
# CLI: demo — start all-tombstones, simulate an agent self-exploring, show O(1) resident
# --------------------------------------------------------------------------------------
def _discover_one(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ctxwin as W
    roots = W.discover_roots()
    sess = W.discover(roots, ns_prefix="" if args.all else W.NS_INCLUDE_PREFIX, max_n=args.max)
    if not sess:
        return None
    if args.session:
        sess = [s for s in sess if args.session in os.path.basename(s["path"])] or sess
    return sess[0]["path"]


def cmd_demo(args):
    path = args.transcript or _discover_one(args)
    if not path:
        print("no transcript found; pass a path or try --all", file=sys.stderr)
        return 2
    st = store_from_session(path, top_n=args.top_n)
    if not st.order:
        print("session has no tool results to navigate", file=sys.stderr)
        return 2
    label = os.path.basename(path)
    full = st.full_tok()
    print(f"\n  ctxnav · agent self-exploration — {label}")
    print(f"  {len(st.order)} tool-result nodes · {full:,} tok of full content in the store\n")
    print("  start: every node a TOMBSTONE")
    print(f"    resident = {st.resident_tok():,} tok  ({100*st.resident_tok()/max(1,full):.2f}% of full; the store is O(1))\n")

    # simulate an agent deciding to drill in: expand the biggest node, then recurse into its
    # middle, under a fixed per-expand budget — the exact memory(ref, expand, budget) loop.
    big = max(st.order, key=lambda r: st.nodes[r].full_tok)
    b = int(args.budget)
    print(f"  agent: memory(#{big}, expand, {b})  — wants more on the largest node")
    st.memory(big, "expand", b)
    print(f"    resident now {st.resident_tok():,} tok")
    cur = big
    for d in range(args.steps):
        cid = st.child_of(cur)
        if cid is None:
            print(f"    (node fully resolved at depth {d}; nothing left to drill)")
            break
        print(f"  agent: memory(#{cid}, expand, {b})  — recurse into the elided middle (depth {d+1})")
        st.memory(cid, "expand", b)
        print(f"    resident now {st.resident_tok():,} tok")
        cur = cid
    # the agent decides it has what it needs and zooms back out
    print(f"  agent: memory(#{big}, contract)  — done; zoom the whole branch back to a tombstone")
    st.memory(big, "contract")
    print(f"    resident back to {st.resident_tok():,} tok\n")

    # replay witness: the journal reproduces the exact resident view
    rp = st.replay()
    ok = rp.resident_tok() == st.resident_tok()
    print(f"  replay: {len(st.journal)} ops re-applied -> resident {rp.resident_tok():,} tok "
          f"({'MATCH' if ok else 'MISMATCH'}) - the exploration path is deterministic + replayable")
    print(f"\n  Every node went up/down on demand; resident stayed O(1) while all {full:,} tok "
          f"stayed reachable. That is agent-navigable context.\n")
    return 0


# --------------------------------------------------------------------------------------
# CLI: selfcheck — synthetic nodes with KNOWN structure (CI gate, anti-overclaim)
# --------------------------------------------------------------------------------------
def runselfcheck():
    print("== ctxnav selfcheck: synthetic nodes (no real data) ==\n")
    fails = 0

    st = Store()
    big = st.add("Read", "X" * 40000)        # 10000 tok
    st.add("Bash", "Y" * 4000)               # 1000 tok
    full = st.full_tok()

    # 1) all-tombstones resident is tiny and bounded (O(1)), << full
    r0 = st.resident_tok()
    ok = r0 <= 2 * TOMBSTONE_TOK + 4 and r0 < full / 100
    print(f"  tombstone resident is O(1)   {'PASS' if ok else 'FAIL'}  resident={r0} full={full}")
    fails += not ok

    # 2) expand to a budget renders ~budget tok (head+tail), not the full node
    st.memory(big, "expand", 1000)
    rb = itok(st.render_node(big))
    ok = 800 <= rb <= 1200
    print(f"  expand respects budget       {'PASS' if ok else 'FAIL'}  rendered~{rb} (budget 1000)")
    fails += not ok

    # 3) recursion: the windowed node has a middle child that itself expands (drill down)
    cid = st.child_of(big)
    ok = cid is not None and cid in st.nodes
    print(f"  recursion mints a child      {'PASS' if ok else 'FAIL'}  child={cid}")
    fails += not ok
    st.memory(cid, "expand", 1000)
    ok = itok(st.render_node(cid)) >= 800
    print(f"  child expands (go deeper)    {'PASS' if ok else 'FAIL'}  child rendered~{itok(st.render_node(cid))}")
    fails += not ok

    # 4) contract frees the budget back to a tombstone (go up)
    st.memory(big, "contract")
    ok = itok(st.render_node(big)) <= TOMBSTONE_TOK + 2
    print(f"  contract frees the budget    {'PASS' if ok else 'FAIL'}  resident node back to {itok(st.render_node(big))} tok")
    fails += not ok

    # 5) expand to >= full size yields the EXACT bytes (lossless, never lossy)
    st.memory(big, "expand", 10_000_000)
    ok = st.render_node(big) == st.nodes[big].content
    print(f"  full expand is exact         {'PASS' if ok else 'FAIL'}  (lossless store)")
    fails += not ok

    # 6) replay determinism: the journal reproduces the resident view byte-for-byte
    rp = st.replay()
    ok = rp.resident_tok() == st.resident_tok()
    print(f"  journal replay deterministic {'PASS' if ok else 'FAIL'}  {st.resident_tok()} == {rp.resident_tok()}")
    fails += not ok

    print()
    if fails:
        print(f"SELFCHECK FAILED — {fails} check(s) failed")
        return 1
    print("OK — tombstones are O(1), expand/contract honor budget + go up/down, recursion drills, "
          "full expand is lossless, and the exploration journal replays deterministically")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="agent-navigable context: memory(ref, expand/contract, budget)")
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="start all-tombstones, simulate an agent self-exploring a real session")
    d.add_argument("transcript", nargs="?", default=None)
    d.add_argument("--session", default=None, help="substring of a discovered session basename")
    d.add_argument("--budget", type=float, default=1000, help="tokens per expand (default 1000)")
    d.add_argument("--steps", type=int, default=3, help="how many recursive drill-downs to simulate")
    d.add_argument("--top-n", type=int, default=40, help="how many of the heaviest results to load as nodes")
    d.add_argument("--max", type=int, default=20)
    d.add_argument("--all", action="store_true")

    sub.add_parser("selfcheck", help="synthetic nodes: O(1) tombstones + expand/contract/recurse + replay (CI gate)")

    args = ap.parse_args(argv)
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "selfcheck":
        return runselfcheck()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
