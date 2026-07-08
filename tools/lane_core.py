"""Python mirror of the Go core-source / trust-critical lane predicates.

This module is the LIVE-PYTHON twin of ``internal/dispatchtick/selfmodify.go``:
the same closed set of tree predicates the native ``fak dispatch`` path uses to
decide (a) which lanes a GUARDED worker may ship to -- the SELF_MODIFY hold,
narrowed to the trust-critical referee set -- and (b) which lanes are fak's own
guard-shippable CORE engineering (the "default the wave toward core forward
progress" bias). It is duplicated here rather than shelled to Go so the module
the live Scheduled Tasks actually invoke (``tools/issue_resolve_dispatch.py``)
ranks lanes by the SAME rule the Go path does -- the identical mirror discipline
``internal/dispatchtick/priority.go`` keeps with ``tools/issue_triage.py``'s
PRIORITY map.

Keep these tables in lockstep with ``selfmodify.go``: ``SelfSourceTreePrefixes``,
``TrustCriticalTreePrefixes``, ``TrustCriticalFileGlobs``, ``normalizeTree``,
``IsSelfSourceTree``, ``IsTrustCriticalTree``, ``IsCoreSourceLaneTree``,
``LaneDispatchableUnderGuard``. A drift between the two is a dispatch-parity bug.

Why the hold is NARROW (trust-critical only, not all self-source): a guarded
self-improving worker must never SHIP an edit to the referee that grades its own
homework -- the ABI/kernel/adjudicator/policy/registrations loader and the
architest/shipgate witness gates -- the RSI hazard #1397 protects against.
Everything ELSE under ``cmd/**``/``internal/**`` (gateway, agent, compute, ...)
is guard-shippable and deliberately NOT held; concurrent core work on the shared
trunk is kept build-safe by the push-seam TRUNK_WOULD_NOT_COMPILE gate
(``cmd/fak/prepush_build.go`` via ``fak hooks pre-push``), not by holding the
whole self-source tree. Holding all of self-source starved the guarded picker of
the ~85% of the backlog the guard actually permits, leaving only the coarse
docs/tools buckets (#1338/#1397).

Read-only, stdlib-only, FAIL-OPEN: an unknown/blank tree is never CORE and is
DISPATCHABLE (never spuriously held), so a taxonomy the router under-declares
degrades to today's behavior rather than vanishing from the surface.
"""

from __future__ import annotations

from typing import Any

# Mirror selfmodify.go SelfSourceTreePrefixes: fak's own running source -- the
# BROAD "compiles into the binary" predicate.
SELF_SOURCE_TREE_PREFIXES = ("cmd/", "internal/")

# Mirror selfmodify.go TrustCriticalTreePrefixes: the referee / witness machinery
# a guarded worker must never SHIP an edit to (a strict subset of self-source).
TRUST_CRITICAL_TREE_PREFIXES = (
    "internal/abi/",
    "internal/kernel/",
    "internal/adjudicator/",
    "internal/policy/",
    "internal/registrations/",
    "internal/architest/",
    "internal/shipgate/",
)

# Mirror selfmodify.go TrustCriticalFileGlobs: the FILE-level members of the same
# trust set (lane taxonomy + stamp grammar, policy manifest, version stamp).
TRUST_CRITICAL_FILE_GLOBS = ("dos.toml", ".dos/", "policy.json", "VERSION")


def _string_globs(tree: Any) -> list[str]:
    """Coerce a lane tree (list of globs, a bare string, or None) to a list of
    strings. Anything else yields ``[]`` so a malformed tree is simply non-core
    and dispatchable (fail-open), never an exception."""
    if tree is None:
        return []
    if isinstance(tree, str):
        return [tree]
    if isinstance(tree, (list, tuple)):
        return [str(g) for g in tree]
    return []


def normalize_tree(glob: str) -> str:
    """Mirror selfmodify.go normalizeTree: strip a leading ``./`` or ``fak/``
    module prefix, normalize backslashes to forward slashes, and trim whitespace,
    so a Windows-authored or module-prefixed glob matches the same as a POSIX
    one."""
    g = str(glob).strip().replace("\\", "/")
    if g.startswith("./"):
        g = g[2:]
    if g.startswith("fak/"):
        g = g[4:]
    return g


def is_self_source_tree_glob(glob: str) -> bool:
    """Mirror selfmodify.go IsSelfSourceTree: one glob rooted in ``cmd/**`` or
    ``internal/**``."""
    return normalize_tree(glob).startswith(SELF_SOURCE_TREE_PREFIXES)


def is_trust_critical_tree(glob: str) -> bool:
    """Mirror selfmodify.go IsTrustCriticalTree: one glob rooted in the
    trust-critical referee/witness machinery (a strict subset of self-source --
    the predicate the SELF_MODIFY hold keys on)."""
    g = normalize_tree(glob)
    if not g:
        return False
    if g.startswith(TRUST_CRITICAL_TREE_PREFIXES):
        return True
    for f in TRUST_CRITICAL_FILE_GLOBS:
        if g == f or g.startswith(f):
            return True
    return False


def is_core_source_lane_tree(tree: Any) -> bool:
    """Mirror selfmodify.go IsCoreSourceLaneTree: True iff EVERY glob is
    guard-shippable core -- self-source AND not trust-critical -- and the tree is
    non-empty. An empty tree, any non-self-source glob, or any trust-critical glob
    yields False, so a mixed lane never masquerades as core and the held referee
    set is never PREFERRED (a guarded worker aimed there only wastes a slot on a
    doomed SELF_MODIFY-held pick)."""
    globs = _string_globs(tree)
    if not globs:
        return False
    for g in globs:
        if (not g.strip()
                or not is_self_source_tree_glob(g)
                or is_trust_critical_tree(g)):
            return False
    return True


def lane_dispatchable_under_guard(guarded: bool, tree: Any) -> bool:
    """Mirror selfmodify.go LaneDispatchableUnderGuard: whether a GUARDED worker
    may ship to a lane -- i.e. the lane is NOT rooted in the trust-critical
    machinery. Unguarded → always dispatchable (the operator/worktree escape
    #1334). A lane with NO declared tree is dispatchable (fail-open): an empty
    tree carries no trust-critical witness to hold on, so failing OPEN keeps a
    lane the taxonomy under-declares from silently vanishing from the surface."""
    if not guarded:
        return True
    for g in _string_globs(tree):
        if is_trust_critical_tree(g):
            return False
    return True
