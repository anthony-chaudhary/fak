# GitHub Pages publishing

fak publishes reader-facing material from the tracked `docs/` source tree at
<https://anthony-chaudhary.github.io/fak/>. The publishing contract is intentionally
source-driven: edit or delete source, merge it to `main`, and the Pages deployment is
rebuilt from an empty output directory.

## What happens automatically

`.github/workflows/pages.yml` coalesces source churn on a 15-minute schedule, and also runs immediately when the publishing implementation changes. A per-doc push trigger is intentionally avoided: this shared trunk can land docs many times per minute, and GitHub Actions keeps only one pending run per concurrency group, so newer pushes otherwise starve every build before it starts:

1. `pagescheck source` rejects non-UTF-8 source before Jekyll can fail opaquely.
2. `pagescheck seo` scores the complete published source corpus and refuses regression below the checked-in score/debt/orphan baseline with a narrow cross-platform path-resolution allowance (85.0 / 424 / 50; measured source is 85.2 / 416 / 47); its full JSON witness is published at `/_proofs/seo-report.json`.
3. GitHub's supported Jekyll builder creates a fresh `_site` from `docs/`.
4. `pagescheck artifact` refuses a narrow or SEO-broken artifact. It requires at least
   1,000 HTML pages, a sitemap using the production base URL, the front page, and the
   rendered `awesome-token-efficiency.html` page with title, description, and canonical URL.
5. The check writes `_site/.pages-manifest.json`, an exact sorted list of deployed paths,
   byte sizes, and SHA-256 digests.
6. `deploy-pages` replaces the prior Pages artifact with that clean build. An in-progress build is never cancelled. The cadence coalesces rapid source updates, so the single pending concurrency slot advances instead of being replaced on every trunk commit.

The replacement model is the stale-page deletion mechanism: a file removed from `docs/`
cannot survive in `_site`, the manifest, or the next deployment. No generated site is
committed and no long-lived `gh-pages` branch accumulates obsolete files.

## Local preflight

The source check is dependency-free and uses the repository's Go toolchain:

```bash
go run ./cmd/pagescheck source --root docs
```

To audit a locally built Jekyll artifact:

```bash
go run ./cmd/pagescheck artifact --root _site \
  --base-url https://anthony-chaudhary.github.io/fak/ \
  --minimum-pages 1000 --require index.html \
  --require awesome-token-efficiency.html --write-manifest
```

After deployment, the authoritative checks are the workflow's green `deploy` job, the live
page, `sitemap.xml`, and `.pages-manifest.json`. Repository Pages settings must use **GitHub
Actions** as the build source; legacy `main:/docs` builds bypass this contract.

## CI/CD contract impact

- **Changed:** Pages moves from the legacy `main:/docs` builder to the explicit `pages.yml` artifact and deployment contract.
- **Consumers migrated:** the Pages environment receives `_site`; `robots.txt` continues to point crawlers at the generated production sitemap.
- **Cutover / rollback:** switch Pages `build_type` to `workflow`; rollback is restoring `legacy` with `main:/docs`, though that also restores opaque failures and stale-output risk.

