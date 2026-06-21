#!/usr/bin/env python3
"""Receive, import, and verify a private benchmark-node result bundle.

Expected archive names:

    <host>-YYYYMMDD-HHMMSS.tgz
    <host>-turn-agent-<profile>-YYYYMMDD-HHMMSS.tgz

This script can pull pending Taildrop files into a repo-local inbox, safely
extract the newest matching archive into fak/experiments/fleet-nodes/<host>/,
then verify it. Broad node bundles run the Phase 0 gate and cross-node
comparison; turn-agent bundles check the sweep manifest and JSON.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "remote-node"
DEFAULT_INBOX = ROOT / "tools" / "_registry" / "taildrop-inbox"
DEFAULT_NODES = ROOT / "fak" / "experiments" / "fleet-nodes"


def find_tailscale():
    exe = shutil.which("tailscale")
    if exe:
        return exe
    windows_default = Path(r"C:\Program Files\Tailscale\tailscale.exe")
    if windows_default.exists():
        return str(windows_default)
    return None


def run(cmd, *, check=True):
    print("+ " + " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def receive_taildrop(inbox, wait):
    inbox.mkdir(parents=True, exist_ok=True)
    tailscale = find_tailscale()
    if not tailscale:
        print("tailscale CLI not found; skipping Taildrop receive", file=sys.stderr)
        return
    cmd = [tailscale, "file", "get", "--conflict=rename"]
    if wait:
        cmd.append("--wait")
    cmd.append(str(inbox))
    run(cmd, check=False)


def archive_candidates(inbox, host):
    patterns = (f"{host}-*.tgz", f"{host}-*.tar.gz", f"{host}.tgz", f"{host}.tar.gz")
    paths = []
    for pattern in patterns:
        paths.extend(inbox.glob(pattern))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)


def validate_archive(tar, host):
    members = tar.getmembers()
    if not members:
        raise ValueError("archive is empty")
    for member in members:
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"unsafe archive member path: {member.name}")
        if not name.parts or name.parts[0] != host:
            raise ValueError(f"archive member is not under {host}/: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported archive member type: {member.name}")
    return members


def extract_archive(archive, nodes_dir, host, replace):
    dest = nodes_dir / host
    if dest.exists() and not replace:
        raise FileExistsError(f"{dest} already exists; rerun with --replace to overwrite it")
    nodes_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    with tarfile.open(archive, mode="r:*") as tar:
        members = validate_archive(tar, host)
        tar.extractall(nodes_dir, members=members)
    if not dest.is_dir():
        raise FileNotFoundError(f"archive did not create {dest}")
    return dest


def verify(node_dir, min_speedup):
    gate = ROOT / "tools" / "fak_phase0_gate.py"
    compare = ROOT / "tools" / "fak_node_compare.py"
    run([sys.executable, str(gate), str(node_dir), "--clean-node", "--min-speedup", str(min_speedup)])
    run([sys.executable, str(compare)])


def looks_like_turn_agent(node_dir):
    return any(node_dir.glob("turn-agent-*/turn-agent-sweep-manifest.json"))


def verify_turn_agent(node_dir):
    candidates = sorted(node_dir.glob("turn-agent-*/turn-agent-sweep-manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"no turn-agent manifest under {node_dir}")
    manifest_path = candidates[-1]
    result_path = manifest_path.parent / "turn-agent-fak-q8.json"
    stderr_path = manifest_path.parent / "turn-agent-fak-q8.err.txt"
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if not result_path.exists():
        detail = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        raise FileNotFoundError(f"missing {result_path}; stderr:\n{detail[-2000:]}")
    with result_path.open(encoding="utf-8") as f:
        result = json.load(f)
    points = result.get("points") or []
    if not points:
        raise ValueError(f"{result_path} contains no points")
    required = {"turns", "concurrency", "reuse_total_ms", "reuse_agent_turns_per_sec"}
    for i, point in enumerate(points):
        missing = sorted(required - set(point))
        if missing:
            raise ValueError(f"{result_path} point {i} missing fields: {', '.join(missing)}")
    print(f"turn-agent profile: {manifest.get('profile')} ({len(points)} points)")
    for point in points:
        print(
            "  T={turns} A={concurrency} fak={reuse_agent_turns_per_sec:.3f} turns/s total={reuse_total_ms:.0f}ms".format(
                **point
            )
        )


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--nodes-dir", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--wait", action="store_true", help="wait for one Taildrop file if the inbox is empty")
    parser.add_argument("--skip-taildrop", action="store_true", help="do not run tailscale file get first")
    parser.add_argument("--archive", type=Path, help="specific .tgz/.tar.gz bundle to import")
    parser.add_argument("--replace", action="store_true", help="replace an existing node artifact directory")
    parser.add_argument("--turn-agent", action="store_true", help="verify a turn-agent sweep bundle instead of the broad node gate")
    parser.add_argument("--min-speedup", type=float, default=45.0)
    args = parser.parse_args(argv)

    if not args.skip_taildrop:
        receive_taildrop(args.inbox, args.wait)

    archive = args.archive
    if archive is None:
        candidates = archive_candidates(args.inbox, args.host)
        if not candidates:
            print(f"no {args.host} bundle found under {args.inbox}", file=sys.stderr)
            print(f"expected a file like {args.host}-YYYYMMDD-HHMMSS.tgz", file=sys.stderr)
            print(f"or {args.host}-turn-agent-PROFILE-YYYYMMDD-HHMMSS.tgz", file=sys.stderr)
            return 1
        archive = candidates[0]

    node_dir = extract_archive(archive.resolve(), args.nodes_dir.resolve(), args.host, args.replace)
    if args.turn_agent or looks_like_turn_agent(node_dir):
        verify_turn_agent(node_dir)
    else:
        verify(node_dir, args.min_speedup)
    print(f"imported and verified {node_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
