#!/usr/bin/env python3
"""Unit test for the fleet-reuse demo's accounting (no kernel, no model, no GPU).

This is the runnable gate for examples/fleet-reuse-demo: it pins the exact reuse curve
the demo prints, so a regression in the bytes/turns accounting fails here instead of
silently shipping a wrong number. Run: python3 demo_test.py  (exit 0 = pass).
"""
import sys

import demo


def approx(label, got, want):
    if got != want:
        print(f"FAIL {label}: got {got!r}, want {want!r}")
        return False
    return True


def main():
    ok = True
    shared = len(demo.SHARED_PROMPT.encode("utf-8"))

    # N = 1: nothing to reuse yet — naive and fak must be byte-identical, and the curve
    # must NOT pretend there is a win here.
    r1 = demo.reuse_curve(1)
    ok &= approx("N=1 naive==fak bytes", r1["naive_bytes"], r1["fak_bytes"])
    ok &= approx("N=1 bytes_saved", r1["bytes_saved"], 0)
    ok &= approx("N=1 fak setup turns", r1["fak_setup_turns"], 1)
    ok &= approx("N=1 naive setup turns", r1["naive_setup_turns"], 1)
    ok &= approx("N=1 shared bytes", r1["shared_bytes"], shared)

    # N = 5: fak re-sends the shared prefix once; the naive loop re-sends it 5×.
    r5 = demo.reuse_curve(5)
    d = [len(demo.WORKER_SUBTASKS[i % len(demo.WORKER_SUBTASKS)].encode("utf-8")) for i in range(5)]
    want_naive = sum(shared + di for di in d)
    want_fak = (shared + d[0]) + sum(d[1:])
    ok &= approx("N=5 naive bytes", r5["naive_bytes"], want_naive)
    ok &= approx("N=5 fak bytes", r5["fak_bytes"], want_fak)
    ok &= approx("N=5 bytes_saved", r5["bytes_saved"], want_naive - want_fak)
    ok &= approx("N=5 fak setup turns", r5["fak_setup_turns"], 1)
    ok &= approx("N=5 naive setup turns", r5["naive_setup_turns"], 5)
    ok &= approx("N=5 injection seen naive", r5["injection_seen_naive"], 5)
    ok &= approx("N=5 injection seen fak", r5["injection_seen_fak"], 1)

    # The fak prefix saving for N>1 is exactly (N-1) copies of the shared prefix.
    ok &= approx("N=5 saving == 4×shared", r5["bytes_saved"], 4 * shared)

    # Monotonic reuse curve: bytes saved grows with N (the whole point of "fleet").
    saved = [demo.reuse_curve(n)["bytes_saved"] for n in (1, 2, 5)]
    if not (saved[0] < saved[1] < saved[2]):
        print(f"FAIL reuse curve not increasing: {saved}")
        ok = False

    # n < 1 is a usage error, not a silent zero.
    try:
        demo.reuse_curve(0)
        print("FAIL reuse_curve(0) should raise")
        ok = False
    except ValueError:
        pass

    print("ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
