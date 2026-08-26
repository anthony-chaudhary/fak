# Repository-index backend decision — 2026-08-25

Issue: [#8937](https://github.com/anthony-chaudhary/fak/issues/8937)

Status: **decided** — keep the exhaustive-study spine local, deterministic, and backend-free; document external indexes as opt-in evidence sources rather than silent prerequisites.

## Decision in one screen

For FAK contributors who need an exhaustive, reproducible repository study, the problem is establishing what was actually inspected without making a hosted search service, language indexer, or proprietary corpus a prerequisite. Today `fak study-inventory` records the pinned checkout's files, language mix, size bands, generated/vendor/testdata classification, and suggested review batches; `rg`/`git grep` then search that same checkout. This is better than selecting one universal index because each candidate below covers a different axis and can be stale, language-limited, unavailable, or detached from the studied revision. The witness is a committed inventory plus literal commands and source pins for every optional backend.

**Native default:** `fak study-inventory` remains the authoritative tree-coverage layer. Local `rg` or `git grep` remains the first text-search recipe. Checked-in Markdown/JSON remains the candidate-decision record. These work offline, bind directly to a revision, expose omissions, and add no daemon, credentials, or index freshness state.

**Optional recipes:** use GitHub CLI/API for issue, PR, and release history; GitHub code search or Zoekt for remote/cross-repository text discovery; SCIP, Kythe, or OpenGrok when definition/reference navigation is worth an index build; CodeQL only for build-aware semantic/security questions. Sourcegraph is an optional integrated host for search plus precise navigation, not a FAK dependency. Glean is excluded from an implementation recipe until an owned deployment and reproducible export exist.

No backend is adopted by default, and no backend result proves exhaustive coverage by itself. A study must still name repository, immutable revision, backend/version, index time or freshness evidence, exact query, and local read-back of every cited file.

## Layer decision matrix

Legend: **D** default, **O** optional, **—** not the tool's role. “Candidate storage” means the durable matrix of systems or findings selected for follow-up, not an engine's internal posting list.

| Backend / layer | Local text | Symbols / definitions / references | Build-aware semantic facts | Issue / PR / release history | Candidate storage | Decision and principal failure mode |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `fak study-inventory` + `rg`/`git grep` + checked-in Markdown/JSON | **D** | — | — | — | **D** | Keep native. Revision-bound and inspectable; literal search misses semantic relationships and untracked remote history. |
| GitHub CLI/API | O | — | — | **D** | O | Default history recipe when GitHub is the forge. Pagination, permissions, deleted objects, rate limits, and mutable search indexes can make an unrecorded query incomplete. |
| GitHub code search | O | O | — | — | — | Optional zero-operator discovery across GitHub. Indexed-default-branch scope, permissions, search syntax/limits, and index freshness mean it cannot certify the local pinned tree. |
| Zoekt | O | — | — | — | — | Preferred self-hosted fast text/regex candidate if repeated multi-repo search justifies an index. It is code search, not semantic reference analysis; branches and shards can be stale or omitted. |
| SCIP indexers / SCIP data | — | O | O | — | — | Preferred portable symbol/reference interchange when a supported language indexer exists. SCIP is a protocol/data model, not one universal indexer; precision and build fidelity inherit each indexer and invocation. |
| Kythe | — | O | O | — | — | Optional graph for organizations already producing compilation-aware Kythe entries. Extractor/language coverage and serving infrastructure are substantial prerequisites; incomplete compilations create incomplete graphs. |
| OpenGrok | O | O | — | — | — | Optional self-hosted browser for many source trees. Analyzer support varies by language, history integration needs repository metadata, and reindex lag can hide the studied revision. |
| Sourcegraph Search + precise code navigation | O | O | O | — | — | Optional integrated deployment when cross-repo search and SCIP-backed navigation are both already operated. Instance access, repository inclusion, ranking, index freshness, and precise-navigation uploads are separate states. |
| CodeQL CLI/database | O | O | **O** | — | — | Use only for a concrete data-flow, control-flow, call-graph, or security question. Database creation may require a successful language build; supported-language and extractor behavior limit coverage, and setup is too heavy for baseline inventory. |
| Glean workplace search | O | O | — | O | O | **No FAK recipe now.** It may unify enterprise code and work artifacts, but connector permissions, proprietary ranking/schema, tenant availability, freshness, and non-reproducible exports prevent a public revision-bound witness. |

The rows are intentionally not a horse race. For example, Zoekt can improve candidate recall over many repositories without replacing SCIP references; CodeQL can answer a semantic query without replacing forge history; GitHub history can recover issue context without proving source-tree coverage.

## Reproducible recipes and exclusions

Run every recipe against an immutable checkout or record the remote index's exact repository/revision. Commands below are the first check, not a claim that a service is installed.

### 1. Native inventory, local text, and durable candidate matrix — required

```bash
git clone git@github.com:OWNER/REPO.git study-repo
cd study-repo
git checkout --detach <full-commit-sha>
fak study-inventory --root . --repository OWNER/REPO --revision <full-commit-sha> --json --out inventory.json
rg -n --hidden --glob '!.git/**' '<literal-or-regex>' .
git grep -n '<literal>' <full-commit-sha>
```

Commit the inventory (when it is a durable study artifact), the decision note, and any machine-readable candidate matrix together. Failure is explicit: unreadable paths, submodules/LFS objects not materialized, generated/vendor classification requiring review, or a query that needs semantics rather than text. The command contract is documented in [`fak study-inventory`](../cli-reference.md#fak-study-inventory); its report does not pretend to contain symbols or forge history.

### 2. GitHub history — default hosted-history recipe

Prerequisites: `gh`, authentication for non-public data, and explicit pagination/time bounds.

```bash
gh api --paginate repos/OWNER/REPO/issues -f state=all -f per_page=100 > issues-and-prs.json
gh api --paginate repos/OWNER/REPO/releases -f per_page=100 > releases.json
gh search prs --repo OWNER/REPO --match title,body,comments '<term>' --limit 1000
```

GitHub's issues endpoint includes pull requests, identified by `pull_request`; preserve raw pages or normalize that distinction. Record retrieval time and query. Search limits, inaccessible/private records, deleted records, and mutable indexing are failure modes. Primary sources: [REST issue listing](https://docs.github.com/en/rest/issues/issues#list-repository-issues), [REST releases](https://docs.github.com/en/rest/releases/releases#list-releases), and [`gh search prs`](https://cli.github.com/manual/gh_search_prs) (CLI observed as 2.79.0 on 2026-08-25).

### 3. GitHub code search — optional remote discovery

```bash
gh search code '<term>' --repo OWNER/REPO --limit 100 --json path,repository,sha,textMatches
```

Use it to discover candidates, then read them from the pinned local checkout. Exclude it as an exhaustiveness witness because result caps, access, searchable-branch rules, syntax, and index freshness are external. Primary sources: [GitHub code-search syntax](https://docs.github.com/en/search-github/github-code-search/understanding-github-code-search-syntax) and [`gh search code`](https://cli.github.com/manual/gh_search_code).

### 4. Zoekt — optional repeated cross-repository text search

Prerequisites: Go toolchain, enough disk/RAM for an index, and an owner for refresh/rebuild.

```bash
go install github.com/sourcegraph/zoekt/cmd/zoekt-git-index@6fa876bf5395a70bc2249128d0e3734d591047be
go install github.com/sourcegraph/zoekt/cmd/zoekt-webserver@6fa876bf5395a70bc2249128d0e3734d591047be
zoekt-git-index -index /tmp/zoekt-index /path/to/pinned/repo
zoekt-webserver -index /tmp/zoekt-index
```

Verify the indexed repository/branch and rebuild after revision changes. There is no symbol/semantic claim. Primary source pin: [`sourcegraph/zoekt@6fa876b`](https://github.com/sourcegraph/zoekt/tree/6fa876bf5395a70bc2249128d0e3734d591047be) (observed 2026-08-25).

### 5. SCIP — optional portable symbol/reference layer

Prerequisites: a language-specific SCIP indexer and that indexer's build/dependency requirements.

```bash
# Example only for a Go repository:
go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
scip-go
# Require the resulting index.scip and indexer version in the study evidence.
```

Do not install `@latest` in a durable witness: resolve and record the selected indexer's immutable version first. If no maintained indexer supports the language/build, exclude SCIP rather than representing text matches as references. Primary sources: [SCIP indexer overview](https://scip-code.org/), [SCIP protocol source at `sourcegraph/scip@a7b9c65`](https://github.com/sourcegraph/scip/tree/a7b9c65a8aa148a79b67cc7f6dafea154dbc63d0), and [SCIP Go](https://github.com/sourcegraph/scip-go).

### 6. Kythe — optional compilation-aware graph

Prerequisites: a supported language extractor, a real build whose compilation units can be captured, Kythe release tooling, storage, and serving ownership.

```bash
# After producing entries with the language's documented extractor:
write_entries < entries.entries
write_tables
```

Those command names are the architecture's first check, not a portable install recipe; follow the chosen release/container and language guide. Exclude Kythe when the real build cannot be captured or language support is absent. Primary sources: [Kythe overview](https://kythe.io/docs/kythe-overview.html), [getting started](https://kythe.io/getting-started/), and [`kythe/kythe@26056ed`](https://github.com/kythe/kythe/tree/26056edfc953b5d4ea0ed8e94db072caa7f7d4c7) (observed 2026-07-16).

### 7. OpenGrok — optional self-hosted source browser

Prerequisites: a supported Java runtime, Universal Ctags, OpenGrok distribution/container, source root, data/config directories, and a refresh owner.

```bash
docker run --rm -p 8080:8080 \
  -v /path/to/pinned/repos:/opengrok/src:ro \
  -v opengrok-data:/opengrok/data \
  opengrok/docker:1.14.17
```

Pin the container digest in real evidence; the tag above is the release first check. Analyzer limitations and index lag require local read-back. Primary sources: [OpenGrok setup](https://github.com/oracle/opengrok/wiki/How-to-setup-OpenGrok), [Docker image documentation](https://github.com/oracle/opengrok/tree/a0503de43e69f1f562a7133963e2ce903d42279b/docker), and [`oracle/opengrok` 1.14.17](https://github.com/oracle/opengrok/releases/tag/1.14.17).

### 8. Sourcegraph — optional integrated search/navigation host

Prerequisites: an owned Sourcegraph instance/account, repository synchronization, search-index freshness evidence, and separately configured precise code navigation/index uploads.

```bash
src login https://<sourcegraph-host> -token "$SRC_ACCESS_TOKEN"
src search -json 'repo:^github\.com/OWNER/REPO$ rev:<full-commit-sha> <term>'
```

If the requested revision is not indexed, stop and use the local checkout. Search availability does not imply SCIP-backed precise navigation availability. Primary sources: [Sourcegraph search query language](https://sourcegraph.com/docs/code-search/queries), [`src search`](https://sourcegraph.com/docs/cli/references/search), and [precise code navigation](https://sourcegraph.com/docs/code-navigation/precise_code_navigation).

### 9. CodeQL — optional build-aware semantic facts

Prerequisites: supported language, CodeQL CLI bundle, query packs, substantial disk/RAM, and—where required—a successful real build.

```bash
codeql version
codeql database create codeql-db --language=<language> --source-root=/path/to/pinned/repo --command='<real build command>'
codeql database analyze codeql-db <query-or-suite> --format=sarif-latest --output=results.sarif
```

Pin CLI and pack versions and preserve database-creation logs. Exclude CodeQL when no concrete semantic question exists, language support is absent, or the build cannot be faithfully reproduced. Primary sources: [CodeQL database creation](https://docs.github.com/en/code-security/codeql-cli/codeql-cli-manual/database-create), [database analysis](https://docs.github.com/en/code-security/codeql-cli/codeql-cli-manual/database-analyze), and [`codeql-cli-binaries` 2.26.3](https://github.com/github/codeql-cli-binaries/releases/tag/v2.26.3).

### 10. Glean — explicit exclusion pending an owner

There is no public, backend-free command that reproduces a tenant's connectors, ACL-filtered corpus, ranking, or answer provenance. The first check is therefore organizational: name the tenant owner, connector scope, export/API contract, immutable source identifiers, freshness SLA, and a scrubbed reproducible output. Until all exist, use GitHub history plus checked-in matrices. Primary source: [Glean connectors overview](https://docs.glean.com/connectors).

## Adoption gates and first implementation child

Do **not** add a backend integration merely because this matrix names it. File an implementation child only when a real study demonstrates all of:

1. the native inventory plus local search leaves a repeated, measured recall or operator-time gap;
2. one optional backend answers that exact layer, against a pinned repository and revision;
3. setup is reproducible with pinned tool/image/indexer versions and a named support owner;
4. stale, partial, permission-filtered, and unsupported-language states fail visibly;
5. output records backend/version, repository/revision, query, index timestamp/freshness, result limits, and local read-back status;
6. the integration remains opt-in and the offline native path stays green.

**First child, if those gates are met:** add `fak study-inventory --history <github-export.json>` as an offline join of explicitly captured issue/PR/release metadata, not a network client. History is the largest uncovered layer that has a stable public export and does not duplicate the existing text/index work. The witness should fixture a paginated export, prove PR-vs-issue classification and revision/time provenance, reject truncated/unversioned input, and show byte-for-byte deterministic output. Do not file or build this child until a study supplies the measured gap and owner required above.

Zoekt/SCIP/CodeQL integration children rank behind that join: their native command outputs can already be cited as optional artifacts, while embedding or operating their indexes would add lifecycle ownership without improving the authoritative local tree map.

## Witness checklist

- [x] Native default and every optional/excluded backend have a first checkable command or explicit exclusion.
- [x] Every backend row links primary documentation and, where a public source repository exists, an immutable source/release pin observed on 2026-08-25.
- [x] Prerequisites and partial/stale/failure modes are explicit.
- [x] The five requested layers are separated rather than conflated.
- [x] A conditional first implementation child has an observable contract.
- [x] This note is linked from [`docs/index.md`](../index.md).
