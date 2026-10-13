#!/usr/bin/env python3
"""Contract tests for the release-artifacts workflow + the install surfaces.

The release-artifacts workflow attaches cross-compiled `fak` binaries to the GitHub
release on a `v*` tag push (issue #133). These tests pin the contract the installer
and adopters rely on WITHOUT cross-compiling: the five targets, the static/no-cgo
build, the version stamp, and idempotent uploads. They also smoke-check that the
installer and Dockerfile stay consistent with the assets the workflow publishes.
"""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "release-artifacts.yml"
MACOS_WORKFLOW = ROOT / ".github" / "workflows" / "release-macos.yml"
CONTAINER_WORKFLOW = ROOT / ".github" / "workflows" / "release-container.yml"
CUDA_CONTAINER_WORKFLOW = ROOT / ".github" / "workflows" / "release-cuda-container.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
INSTALL_SH = ROOT / "install.sh"
DOCKERFILE = ROOT / "Dockerfile"
# The stamp recipe (-trimpath/-ldflags) is DRY'd into the one canonical build
# entrypoint (#3709); the workflow and Dockerfile route through it rather than
# carrying the recipe inline, so those assertions follow it here.
BUILD_SH = ROOT / "scripts" / "build.sh"
UPLOAD_ACTION = ROOT / ".github" / "actions" / "release-upload" / "action.yml"

# The exact target and asset sets published by the release workflow, plus the
# macOS artifacts uploaded by the separate release-macos workflow.
TARGETS = (
    ("linux", "amd64"),
    ("linux", "arm64"),
    ("darwin", "amd64"),
    ("darwin", "arm64"),
    ("windows", "amd64"),
)
ARCHIVES = tuple(
    f"fak_${{VERSION}}_{goos}_{goarch}{'.zip' if goos == 'windows' else '.tar.gz'}"
    for goos, goarch in TARGETS
) + (
    "fak_${VERSION}_darwin_universal.tar.gz",
    "fak_${VERSION}_darwin_arm64_metal.tar.gz",
    "fak_${VERSION}_linux_amd64_vulkan.tar.gz",
)
RELEASE_ASSETS = (*ARCHIVES, *(f"{archive}.sha256" for archive in ARCHIVES), "SHA256SUMS")
LDFLAG = "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion="


def yaml_matrix_targets(text: str) -> list[tuple[str, str]]:
    """Read the simple goos/goarch maps beneath the workflow's matrix include."""
    lines = text.splitlines()
    matrix_index = next(i for i, line in enumerate(lines) if line.strip() == "matrix:")
    matrix_indent = len(lines[matrix_index]) - len(lines[matrix_index].lstrip())
    include_index = next(
        i
        for i in range(matrix_index + 1, len(lines))
        if lines[i].strip() == "include:"
    )
    include_indent = len(lines[include_index]) - len(lines[include_index].lstrip())

    targets: list[tuple[str, str]] = []
    goos: str | None = None
    for line in lines[include_index + 1 :]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped and indent <= include_indent:
            break
        if stripped.startswith("- goos:"):
            goos = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("goarch:") and goos is not None:
            targets.append((goos, stripped.split(":", 1)[1].strip()))
            goos = None

    if matrix_indent >= include_indent:
        raise AssertionError("matrix include indentation is invalid")
    return targets


def matrix_legs(text: str) -> list[dict[str, str]]:
    """Read every `- goos:` map beneath the matrix include as key -> raw value."""
    lines = text.splitlines()
    include_index = next(i for i, line in enumerate(lines) if line.strip() == "include:")
    include_indent = len(lines[include_index]) - len(lines[include_index].lstrip())
    legs: list[dict[str, str]] = []
    for line in lines[include_index + 1 :]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped and indent <= include_indent:
            break
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            legs.append({})
            stripped = stripped[2:]
        key, _, value = stripped.partition(":")
        legs[-1][key.strip()] = value.strip()
    return legs


def bash_array_values(text: str, name: str) -> list[str]:
    """Read one shell array whose entries are one quoted value per line."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"{name}=(")
    values: list[str] = []
    for line in lines[start + 1 :]:
        value = line.strip()
        if value == ")":
            return values
        if value:
            values.append(value.removeprefix('"').removesuffix('"'))
    raise AssertionError(f"unterminated {name} array")


class ReleaseArtifactsWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(WORKFLOW.exists(), f"missing {WORKFLOW}")
        self.text = WORKFLOW.read_text(encoding="utf-8")

    def test_triggers_on_tag_and_dispatch(self) -> None:
        self.assertIn("push:", self.text)
        self.assertIn('tags: ["v*"]', self.text)
        self.assertIn("workflow_dispatch:", self.text)

    def test_builds_exact_five_target_matrix(self) -> None:
        self.assertCountEqual(yaml_matrix_targets(self.text), TARGETS)

    def test_verifies_exact_release_asset_payload(self) -> None:
        archives = bash_array_values(self.text, "expected_archives")
        checked_assets = (*archives, *(f"{archive}.sha256" for archive in archives), "SHA256SUMS")

        self.assertCountEqual(archives, ARCHIVES)
        self.assertCountEqual(checked_assets, RELEASE_ASSETS)
        self.assertIn('[ -s "${assets}/${archive}" ]', self.text)
        self.assertIn('[ -s "${assets}/${archive}.sha256" ]', self.text)
        self.assertIn('[ -s "${assets}/SHA256SUMS" ]', self.text)

    def test_waits_for_macos_gpu_checksum_before_aggregate(self) -> None:
        self.assertIn('fak_${VERSION}_darwin_universal.tar.gz.sha256', self.text)
        self.assertIn('fak_${VERSION}_darwin_arm64_metal.tar.gz.sha256', self.text)
        self.assertIn('fak_${VERSION}_linux_amd64_vulkan.tar.gz.sha256', self.text)
        self.assertIn('CUDA_IMAGE="ghcr.io/${OWNER}/fak:${VERSION}-cuda"', self.text)
        self.assertIn('docker manifest inspect "$CUDA_IMAGE"', self.text)
        self.assertIn('docker pull "$CUDA_IMAGE"', self.text)
        self.assertIn('"cuda" in (d.get("build_tags") or [])', self.text)
        self.assertIn("ldd /usr/local/bin/fak | grep -q libcudart", self.text)
        self.assertIn("CUDA image version mismatch", self.text)
        self.assertIn('cuda_digest=${cuda_digest}', self.text)
        self.assertIn("fold_max=120", self.text)
        self.assertIn('if [ "$fold_ok" -ne 1 ]', self.text)
        self.assertIn("refusing to publish incomplete SHA256SUMS", self.text)

    def test_macos_release_requires_native_metal_linkage(self) -> None:
        text = MACOS_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("otool -L ./dist/arm64-metal/fak", text)
        self.assertIn("/Metal.framework/", text)
        self.assertIn("/MetalPerformanceShaders.framework/", text)
        self.assertNotIn("dist/arm64-metal/fak backends | grep -qx metal", text)
        self.assertNotIn("metal backend probe not available yet", text)
        self.assertIn("dist/arm64-metal/fak dist/amd64/fak", text)
        self.assertLess(
            text.index("Validate Homebrew Formula and Executable Functionality"),
            text.index("Upload validated macOS binaries to GitHub Release"),
        )

    def test_static_no_cgo_build(self) -> None:
        # Static, reproducible, no cgo — the property that lets the binary run
        # anywhere and the distroless image stay tiny — on every leg EXCEPT
        # darwin/arm64, which scripts/build.sh refuses without cgo (Metal stub,
        # #12771/#12784) and which therefore builds natively on a macOS runner
        # (v0.56.0 release blocker). The -trimpath flag lives in the shared recipe.
        self.assertIn("CGO_ENABLED: ${{ matrix.cgo }}", self.text)
        self.assertIn("runs-on: ${{ matrix.runner }}", self.text)
        legs = matrix_legs(self.text)
        self.assertEqual(sorted((leg["goos"], leg["goarch"]) for leg in legs), sorted(TARGETS))
        for leg in legs:
            if (leg["goos"], leg["goarch"]) == ("darwin", "arm64"):
                self.assertEqual(leg["cgo"], '"1"')
                self.assertTrue(leg["runner"].startswith("macos-"), leg)
            else:
                self.assertEqual(leg["cgo"], '"0"', leg)
        self.assertIn("scripts/build.sh", self.text)
        self.assertIn("-trimpath", BUILD_SH.read_text(encoding="utf-8"))
        self.assertIn("REFUSED -- darwin/arm64 without cgo", BUILD_SH.read_text(encoding="utf-8"))

    def test_macos_release_never_builds_cgoless_arm64(self) -> None:
        # release-macos runs with job CGO_ENABLED=0 for the amd64 slice; any
        # darwin/arm64 build there must override it to 1 or build.sh refuses.
        text = MACOS_WORKFLOW.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "GOARCH=arm64" in line and "scripts/build.sh" in line:
                self.assertIn("CGO_ENABLED=1", line)

    def test_stamps_version_ldflag(self) -> None:
        # A shipped binary with no VERSION file alongside resolves its version from
        # this ldflag (internal/appversion.Current precedence). The stamp is DRY'd
        # into scripts/build.sh (#3709); the workflow inherits it by routing through
        # that script rather than carrying the ldflag inline.
        self.assertIn("scripts/build.sh", self.text)
        self.assertIn(LDFLAG, BUILD_SH.read_text(encoding="utf-8"))

    def test_uploads_idempotently_with_checksums(self) -> None:
        # v0.56.0: gh's internal retry re-POSTed an asset whose first attempt had
        # already landed, so GitHub answered HTTP 422 despite --clobber. Every
        # per-build upload goes through the release-upload action, which accepts a
        # same-size published asset or deletes and retries; tooling is taken from
        # the workflow revision so a dispatch backfill of an older tag still works.
        action = UPLOAD_ACTION.read_text(encoding="utf-8")
        self.assertIn("gh release upload", action)
        self.assertIn("--clobber", action)
        self.assertIn("gh release delete-asset", action)
        self.assertIn('.state == \\"uploaded\\"', action)
        self.assertIn("not visible yet", action)
        for text in (self.text, MACOS_WORKFLOW.read_text(encoding="utf-8")):
            self.assertIn("uses: ./.release-tooling/.github/actions/release-upload", text)
            self.assertIn("ref: ${{ github.workflow_sha }}", text)
            self.assertNotIn('gh release upload "$TAG" "$ARCHIVE"', text)
            self.assertNotIn('gh release upload "$TAG" "$UNIVERSAL_ARCHIVE"', text)
        self.assertIn("gh release upload", self.text)
        self.assertIn("--clobber", self.text)
        self.assertIn("sha256sum", self.text)
        self.assertIn("SHA256SUMS", self.text)

    def test_waits_for_release_page_before_upload(self) -> None:
        # Regression #369: the tag push fires THIS workflow and release-cadence's
        # page-creation step (release_publish.py) concurrently; uploading before
        # the release page exists 404s with "release not found". Every upload path
        # must poll `gh release view` until the page appears before touching it.
        self.assertIn("gh release view", self.text)
        self.assertIn("not visible yet", self.text)
        self.assertIn("gh release view", UPLOAD_ACTION.read_text(encoding="utf-8"))

    def test_verify_settles_asset_propagation_before_judging(self) -> None:
        # Regression (v0.37.0): the verify job downloaded assets once, immediately
        # after upload, and raced GitHub's non-read-your-writes asset store — a
        # still-propagating arm64 archive read "FAILED open or read" and quarantined
        # a release whose checksum matched on the next fetch. The verify step must
        # re-download until every expected asset is present and non-empty before
        # judging completeness, so a transient propagation gap cannot false-quarantine.
        self.assertIn("not fully propagated yet", self.text)
        self.assertIn("settle_ok", self.text)

    def test_promotes_verified_release_to_latest(self) -> None:
        # The front-door invariant: verification is symmetric. It quarantines a bad
        # release (prerelease=true, make_latest=false) AND promotes a good one
        # (prerelease=false, make_latest=true) so the latest verified stable cut is
        # what github.com/.../releases shows by default — never left lagging as a
        # prerelease behind an older tag.
        self.assertIn("Promote verified release to Latest", self.text)
        self.assertIn("-F prerelease=false", self.text)
        self.assertIn("-f make_latest=true", self.text)
        self.assertIn("CURRENT_LATEST=", self.text)
        self.assertIn("sort -V | tail -n1", self.text)
        self.assertIn("preserving the newer default", self.text)
        # And it still demotes a failed one.
        self.assertIn("Quarantine failed release", self.text)
        self.assertIn("-F prerelease=true", self.text)
        self.assertIn("-f make_latest=false", self.text)

    def test_failed_gpu_aggregation_still_quarantines_release(self) -> None:
        self.assertIn("if: ${{ always() }}", self.text)
        self.assertIn("needs.checksums.result != 'success'", self.text)
        self.assertIn("artifact aggregation failed", self.text)

    def test_mutable_oci_aliases_are_owned_by_serialized_verified_aggregate(self) -> None:
        cpu = CONTAINER_WORKFLOW.read_text(encoding="utf-8")
        cuda = CUDA_CONTAINER_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("steps.meta.outputs.version }}-cpu", cpu)
        self.assertNotIn("steps.meta.outputs.image }}:latest", cpu)
        self.assertIn("steps.meta.outputs.version }}-cuda", cuda)
        self.assertNotIn("steps.meta.outputs.image }}:cuda-latest", cuda)
        self.assertIn("release-promotion-${{ github.repository }}", self.text)
        self.assertIn('CPU_DIGEST="${{ needs.checksums.outputs.cpu_digest }}"', self.text)
        self.assertIn('CUDA_DIGEST="${{ needs.checksums.outputs.cuda_digest }}"', self.text)
        self.assertIn('imagetools create --tag "${IMAGE}:cpu-latest" "$CPU_DIGEST"', self.text)
        self.assertIn('imagetools create --tag "${IMAGE}:cuda-latest" "$CUDA_DIGEST"', self.text)
        self.assertIn('imagetools create --tag "${IMAGE}:latest" "$CUDA_DIGEST"', self.text)

    def test_checksums_job_resolves_repo_without_checkout(self) -> None:
        # Regression #369: the aggregate-SHA256SUMS job has no checkout, so `gh`
        # cannot infer the repo from a git remote ("fatal: not a git repository").
        # It pins the repo via GH_REPO instead of paying for a full checkout.
        self.assertIn("GH_REPO: ${{ github.repository }}", self.text)

    def test_uses_module_go_version(self) -> None:
        self.assertIn("go-version-file: go.mod", self.text)

    def test_write_permission_scoped_to_upload_jobs(self) -> None:
        # The top-level token is read-only; only the jobs that touch the release
        # escalate to contents: write.
        self.assertIn("permissions:\n  contents: read", self.text)
        self.assertIn("contents: write", self.text)

    def test_announces_release_after_assets_land(self) -> None:
        self.assertIn("announce:", self.text)
        self.assertIn("needs: checksums", self.text)
        self.assertIn("announce release to #releases", self.text)
        self.assertIn("FAK_SCOREBOARD_TOKEN: ${{ secrets.FAK_SCOREBOARD_TOKEN }}", self.text)
        # #releases folds onto the CI/CD reporting sink (C0BGQ411TCJ) by default, exactly
        # like every sibling feeder — the repo variable and the family-wide
        # FAK_CICD_REPORT_CHANNEL are overrides, not hard requirements, so a provisioned
        # repo posts release reports without operator setup.
        self.assertIn("FAK_RELEASES_CHANNEL: ${{ vars.FAK_RELEASES_CHANNEL || vars.FAK_CICD_REPORT_CHANNEL || 'C0BGQ411TCJ' }}", self.text)
        self.assertIn("go run ./cmd/fak scoreboard post", self.text)
        self.assertIn("--kpi release", self.text)
        self.assertIn('--value "$TAG"', self.text)
        self.assertIn('--detail "$NOTES_URL"', self.text)
        self.assertIn('NOTES_URL="https://github.com/${{ github.repository }}/releases/tag/${TAG}"', self.text)

    def test_release_announce_dry_runs_without_secret(self) -> None:
        self.assertIn("render release card (dry-run)", self.text)
        self.assertIn("--dry-run | tee release-card.txt", self.text)
        self.assertIn("GITHUB_STEP_SUMMARY", self.text)
        self.assertIn("post to #releases", self.text)
        # With a built-in default channel the post is gated on the token secret ALONE;
        # the channel can never be empty, so it is not part of the guard anymore.
        self.assertIn("if: ${{ env.FAK_SCOREBOARD_TOKEN != '' }}", self.text)
        self.assertNotIn("env.FAK_RELEASES_CHANNEL != ''", self.text)
        self.assertIn("Release announcement SKIPPED", self.text)
        self.assertIn("FAK_SCOREBOARD_TOKEN secret not set", self.text)

    def test_wired_into_ci(self) -> None:
        ci = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python tools/release_artifacts_workflow_test.py", ci)


class InstallShContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(INSTALL_SH.exists(), f"missing {INSTALL_SH}")
        self.text = INSTALL_SH.read_text(encoding="utf-8")

    def test_naming_matches_workflow(self) -> None:
        # Installer reconstructs the asset name the workflow publishes:
        # fak_<version>_<os>_<arch>.tar.gz
        self.assertIn('NAME="fak_${VERSION}_${GOOS}_${GOARCH}"', self.text)
        self.assertIn('ARCHIVE="${NAME}.tar.gz"', self.text)

    def test_verifies_checksum(self) -> None:
        self.assertIn("SHA256SUMS", self.text)
        self.assertIn("checksum mismatch", self.text)
        # Refuse to install something it could not verify.
        self.assertIn("refusing to install unverified", self.text)

    def test_checksum_match_tolerates_binary_marker(self) -> None:
        # Regression: sha256sum writes "<hash> *<name>"; a plain `grep " <name>$"`
        # misses the `*` marker. The matcher must strip a leading `*` from the field.
        self.assertNotIn('grep " ${ARCHIVE}', self.text)
        self.assertIn('sub(/^[*]/,"",n)', self.text)

    def test_honors_overrides(self) -> None:
        for knob in ("FAK_VERSION", "FAK_INSTALL_DIR", "FAK_REPO"):
            self.assertIn(knob, self.text)

    def test_refuses_unsupported_target(self) -> None:
        self.assertIn("unsupported OS", self.text)
        self.assertIn("unsupported arch", self.text)


class DockerfileContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(DOCKERFILE.exists(), f"missing {DOCKERFILE}")
        self.text = DOCKERFILE.read_text(encoding="utf-8")

    def test_two_stage_static_distroless(self) -> None:
        self.assertIn("FROM golang:1.26 AS build", self.text)
        self.assertIn("CGO_ENABLED=0", self.text)
        self.assertIn("gcr.io/distroless/static", self.text)
        # The version stamp is DRY'd into scripts/build.sh (#3709); the Dockerfile
        # routes through it instead of carrying the ldflag inline.
        self.assertIn("scripts/build.sh", self.text)
        self.assertIn(LDFLAG, BUILD_SH.read_text(encoding="utf-8"))

    def test_serves_on_all_interfaces(self) -> None:
        # Containers must bind 0.0.0.0, not loopback.
        self.assertIn("0.0.0.0:8080", self.text)
        self.assertIn("EXPOSE 8080", self.text)


if __name__ == "__main__":
    unittest.main()
