# Indexed document sets

Any maintained Markdown source longer than **150 lines** (approximately three 50-line pages) must be split into bounded child pages and replaced by a concise index. Mark an index with `<!-- fak:document-set -->`. The threshold is source-line based so local and CI results are deterministic.

The migration is content-preserving: move prose rather than rewrite it, retain stable titles, and repair inbound links. Generated corpora and completed-plan archives are audited by their generators/archive policy rather than this live-doc ratchet.

`CLAIMS.md` is the reference implementation: every second-level claim is an individual page under `docs/claims/`.

## Backfill priority

Direct page-view analytics are not available in this public checkout. The checked-in `document-access-top30.json` therefore labels and records a reproducible **inbound-link count proxy** over committed Markdown: documents linked most often are treated as most accessed. This is observed repository reach, not human traffic.
