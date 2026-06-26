#!/usr/bin/env python3
"""Invariant tests for the committed bench-node registry template (#922).

`tools/bench_node.sh` onboards every bench node off a registry. The REAL registry
(`tools/bench_nodes.json`) carries live tailnet identity and is gitignored; the
COMMITTED surface is the public-safe template `tools/bench_nodes.example.json` plus
its `_onboarding` roster (the sanitized "which node is onboarded vs pending, and
why" view the #922 umbrella tracks).

Nothing previously witnessed that committed surface at the content level:
`check_committed_files_test.py` only path-classifies the example file (it would not
notice a real tailnet IP or hostname pasted into the template, nor a roster that
drifted out of sync with the node list). The README warns this file "has been
clobbered by a regen before". This test is that missing witness — it locks two
invariants the whole onboarding design rests on:

  1. roster <-> nodes bijection: every `_onboarding.roster` entry names a node in
     `nodes[]` and vice versa, with a matching `sanitized_name`. So onboarding a
     node (or retiring one) cannot silently desync the status roster from the
     resolvable node list.
  2. placeholder-only safety: every committed node uses an RFC-5737 documentation
     IP (never the Tailscale 100.64/10 CGNAT range a real node lives on), a
     `.example.ts.net` MagicDNS name, the literal ssh user `user`, and no pinned
     `host_key`. This is exactly the leak a careless copy of the gitignored real
     registry over the template would introduce — and the shape-only scrub audit
     would NOT catch it.

Host-free by construction: it parses committed JSON + reads `git ls-files`; it
never touches a node or the network.

Run: `python tools/bench_node_registry_test.py`  (exit 0 = all pass),
or `python -m pytest tools/bench_node_registry_test.py -q`.
"""
from __future__ import annotations

import ipaddress
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "tools" / "bench_nodes.example.json"
REAL = "tools/bench_nodes.json"

# RFC 5737 documentation ranges — the only IPs a public-safe template may carry.
_DOC_NETS = [
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
]
# Tailscale hands real nodes addresses out of the 100.64.0.0/10 CGNAT range — a
# template IP landing here is a live-identity leak, not a placeholder.
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _load() -> dict:
    with open(EXAMPLE, encoding="utf-8") as fh:
        return json.load(fh)


def test_example_is_tracked_and_real_registry_is_not() -> None:
    """The template is the committed surface; the real-identity registry must stay
    out of the tree (the safety premise of the whole runner)."""
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                       capture_output=True, text=True)
    assert r.returncode == 0, "git ls-files failed"
    tracked = set(r.stdout.split())
    assert "tools/bench_nodes.example.json" in tracked, "template not tracked"
    assert REAL not in tracked, (
        f"{REAL} carries live identity and MUST stay gitignored — it is tracked")


def test_schema_and_blocks_present() -> None:
    d = _load()
    assert d.get("schema") == "fleet-bench-nodes/1", d.get("schema")
    assert isinstance(d.get("nodes"), list) and d["nodes"], "no nodes[]"
    assert "_onboarding" in d and isinstance(d["_onboarding"].get("roster"), list), \
        "no _onboarding.roster"


def test_roster_nodes_bijection() -> None:
    """Every roster entry names a real node and every node has a roster status —
    no silent desync between the onboarding tracker and the resolvable nodes."""
    d = _load()
    node_names = [n["name"] for n in d["nodes"]]
    roster_names = [e["node"] for e in d["_onboarding"]["roster"]]
    assert len(node_names) == len(set(node_names)), f"duplicate node name: {node_names}"
    assert len(roster_names) == len(set(roster_names)), f"duplicate roster node: {roster_names}"
    assert set(node_names) == set(roster_names), (
        "roster/nodes desync — only in nodes: "
        f"{set(node_names) - set(roster_names)}; only in roster: "
        f"{set(roster_names) - set(node_names)}")


def test_sanitized_name_matches_across_roster_and_nodes() -> None:
    d = _load()
    by_node = {n["name"]: n.get("sanitized_name") for n in d["nodes"]}
    for e in d["_onboarding"]["roster"]:
        san = e.get("sanitized_name")
        assert san, f"roster entry {e['node']} has no sanitized_name"
        assert by_node[e["node"]] == san, (
            f"{e['node']}: roster sanitized_name {san!r} != node "
            f"{by_node[e['node']]!r}")


def test_committed_template_is_placeholders_only() -> None:
    """The safety invariant: no live tailnet identity may reach the committed
    template. Catches a regen/copy of the gitignored real registry over it."""
    d = _load()
    for n in d["nodes"]:
        name = n["name"]
        ip = ipaddress.ip_address(n["tailnet_ip"])
        assert any(ip in net for net in _DOC_NETS), (
            f"{name}: tailnet_ip {ip} is not an RFC-5737 documentation address")
        assert ip not in _CGNAT, (
            f"{name}: tailnet_ip {ip} is in the Tailscale CGNAT range — live leak")
        assert str(n["magicdns"]).endswith(".example.ts.net"), (
            f"{name}: magicdns {n['magicdns']!r} is not a .example.ts.net placeholder")
        assert n["ssh_user"] == "user", (
            f"{name}: ssh_user {n['ssh_user']!r} is not the placeholder 'user'")
        assert not n.get("host_key"), (
            f"{name}: host_key is pinned in the committed template — leak")


def test_roster_status_is_well_formed() -> None:
    """An onboarded node carries a verified date; a pending node names its
    blocker — so the roster always says where each node actually stands."""
    d = _load()
    for e in d["_onboarding"]["roster"]:
        st = e.get("status")
        assert st in ("onboarded", "pending"), f"{e['node']}: bad status {st!r}"
        if st == "onboarded":
            assert e.get("verified"), f"{e['node']}: onboarded but no verified date"
        else:
            assert e.get("blocker"), f"{e['node']}: pending but no blocker named"


def _run() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
