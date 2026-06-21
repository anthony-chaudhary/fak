# Expire by Default — moved

*This post was restructured into a modular set on 2026-06-20: a tight core plus three
companion notes, so each piece can be revised, re-pathed, or dropped independently. The
single-file version it replaced over-framed its own claim and litigated its concessions
twice; the split fixes both.*

**Read the core here → [`expire-by-default.md`](expire-by-default.md).**

Companions:
- [Prior art, and where the gap really is](_prior-art.md)
- [The enforcement-topology argument](_enforcement-topology.md)
- [What fell under adversarial review](_adversarial-review.md)

The thesis, in one line: trust decides whether a value may enter memory; *durability*
decides whether it should, and for how long — a fail-closed tag on a write-time gate that
already ships (`internal/ctxmmu`), with the honest default *persist is never right for an
unclassified span*. The contribution is an enforcement-topology claim, not a new memory
semantics — and it is honestly small, riding a gate that is real.
