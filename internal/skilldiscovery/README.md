# Pinned remote skill catalogs (#10712)

An **optional distribution module**, not a trust engine or automatic activation:

~~~text
fak skill pull --source https://HOST/CATALOG/ --index-sha256 TRUSTED_64_HEX --cache-root EXISTING_CANONICAL_DIRECTORY --json
~~~

Uppercase placeholders are required operator inputs, not an executable recipe.
The cache root must already exist, be a real canonical directory and be writable.
No persistent setting, implicit home cache, provider credential, proxy credential,
launch hook or remote skill activation is created.

## Acquisition and normal consumer

1. Obtain the exact index SHA256 through an independently trusted channel. A hash
   computed from the same untrusted download is **not** an authenticity decision.
2. Explicitly invoke the pull command. It accepts only HTTPS, no URL userinfo,
   query or fragment, no redirects, cookies, automatic retry or environment proxy.
   The source URL is hashed in the receipt, not logged.
3. The index, names/paths/version labels and every declared file size and digest
   are validated. Data is bounded before immutable publication.
4. The receipt names one complete skills_root and reports activated:false.
   Review the pinned skill content. Only then deliberately supply that root to
   the existing fak agent --skills-dir ROOT or fak chat --skills-dir ROOT.
5. Those paths already feed ArmCodeToolsWithOptions → DiscoverSkills and the
   native skill tool. Pull does not arm this registry, change the tool permission
   floor, execute scripts, authorize model instructions or grant publisher trust.

The native parser validates each SKILL.md with an empty filename, preventing
filesystem canonical-resolution during acquisition. A mismatched name or nonempty
metadata.canonical is refused: a self-contained bundle cannot later import an
unrelated local file through that mechanism. Normal local-skill parsing and
permissions are unchanged. Skill instructions remain data; requested actions
must still pass normal authorization.

## Catalog profile

The donor's skills[].name/version/files shape is retained, with mandatory
additive sha256 and sizes maps keyed by every file path:

~~~json
{"skills":[{"name":"example","version":"v1","files":["SKILL.md","references/guide.md"],"sha256":{"SKILL.md":"<64 lowercase hex>","references/guide.md":"<64 lowercase hex>"},"sizes":{"SKILL.md":123,"references/guide.md":456}}]}
~~~

This strict integrity-bearing profile does **not** claim universal support for
unpinned OpenCode catalogs or the separate Agent Skills discovery RFC. The
placeholder digests above are intentionally invalid. SKILL.md is required because
it is the actual FAK entrypoint. Unknown fields, missing hashes, bad sizes,
duplicate/case-colliding identities and file/directory collisions refuse as a
whole; unsafe members are never silently dropped.

Paths use portable ASCII letters/digits plus ._- segments, at most eight
segments/512 characters. Absolute, empty, traversal, backslash, percent-encoded,
URL/query/fragment and Windows device-name paths are refused. Names and version
labels are bounded safe segments. This is deliberately narrower than arbitrary
platform filenames.

Limits: 256KiB index, 16 skills, 64 files, 1MiB per file and 8MiB aggregate file
bytes. Integer/schema overflow refuses. Actual response reads are bounded even
when Content-Length is absent or deceptive. Files are fetched sequentially with
30s per-request and a 2min HTTP-phase deadline. Cancellation prevents publication.
There is no inferred throughput improvement.

## Immutable cache and failure truth

Each acquisition creates a private 0700 generation inside the explicit cache
root; files/records are 0600, never executable. Root-relative Go filesystem
operations confine traversal. Symlink cache parents/generations and unlisted
content are refused. Cached bytes are revalidated against the trusted index and
all file digests before a hit is returned.

The visibility point is a **create-only hard link of a complete manifest** into
the source+index-keyed records directory. This is the atomic publication unit:
no mutable current pointer, directory replacement or old-version deletion.
Concurrent pulls publish once or verify the already-published winner. A cache
mismatch refuses without deleting, repairing or fetching over it. Failed new
versions remove only their own unpublished generation and leave every prior
record/root intact, so in-flight registries retain their original paths.

Same-filesystem hard links are required; an unsupported filesystem refuses,
without a non-atomic copy fallback. Files are flushed before publication, but
power-loss durability and hostile same-user filesystem mutation are not claimed.
Killed processes can leave unreferenced private generations; there is no
automatic garbage collection or authority to delete another generation.

## Provenance and adaptations

**ADAPT / OPTIONAL-MODULE:** anomalyco/opencode at
4eb29a64f0054672950acf789f2b09487ebfbb20
(commit event 2026-09-02T20:33:15Z), MIT:
- packages/core/src/skill/discovery.ts, blob a192ba1023b1658b182f03f7a39aa31341fd8dee;
- packages/core/test/skill-discovery.test.ts, blob e875beb931b2241f111dc17c92e7d92755fe28fd;
- LICENSE, blob 6439474beed8e0271df9862eff97ffd70ec2464c, retained here.

Observed 2026-11-02. The donor supplies index discovery, safe path/origin checks,
version caching and stage/backup/rollback. It does not supply the trusted
file-digest profile, byte bounds, immutable retention, FAK parser boundary or
native receipt: these are FAK adaptations, not inherited qualification. Its
4-skill/8-file concurrency is not claimed. The algorithm and path fixtures are
adapted to Go rather than importing the Effect/Bun runtime.

Existing local discovery and separate inert OCI transport are unchanged; this
does not complete OCI activation. Existing #10712 is reused. Donor follow-up
#39106/#39107 describes silent stale-version fallback after refresh failure;
this implementation instead returns a typed failure. RFC/schema proposals are
not shipped compatibility evidence. Refresh on donor schema/consumer or local
trust changes.

## Focused witnesses and limits

~~~text
go test ./internal/skilldiscovery -run '^TestRemoteSkillCatalog$' -count=1 -v
go test ./cmd/fak -run '^TestSkillPullCommand$' -count=1 -v
~~~

These are qualification commands, not execution receipts. Fixtures cover real
registry discovery/harmless loading, retained versions, cleanup, cache tampering,
pins, paths/symlinks, actual byte bounds, concurrent publication, redaction,
redirects and cancellation. Cost markers are declared estimates. Compiler,
fixtures, installed controller, live remote acquisition and end-to-end operator
use remain UNRUN until separately witnessed. Authoring this source downloads,
installs and activates no remote skills.
