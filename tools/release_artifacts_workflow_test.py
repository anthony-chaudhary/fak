#!/usr/bin/env python3
"""Contract tests for the release-artifacts workflow + the install surfaces.

The release-artifacts workflow attaches cross-compiled `fak` binaries to the GitHub
release on a `v*` tag push (issue #133). These tests pin the contract the installer
and adopters rely on WITHOUT cross-compiling: the four targets, the static/no-cgo
build, the version stamp, and idempotent uploads. They also smoke-check that the
installer and Dockerfile stay consistent with the assets the workflow publishes.
"""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "release-artifacts.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
INSTALL_SH = ROOT / "install.sh"
DOCKERFILE = ROOT / "Dockerfile"

# The four targets the issue's "Done when" enumerates.
TARGETS = [
    ("linux", "amd64"),
    ("darwin", "amd64"),
    ("darwin", "arm64"),
    ("windows", "amd64"),
]
LDFLAG = "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion="


class ReleaseArtifactsWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(WORKFLOW.exists(), f"missing {WORKFLOW}")
        self.text = WORKFLOW.read_text(encoding="utf-8")

    def test_triggers_on_tag_and_dispatch(self) -> None:
        self.assertIn("push:", self.text)
        self.assertIn('tags: ["v*"]', self.text)
        self.assertIn("workflow_dispatch:", self.text)

    def test_builds_all_four_targets(self) -> None:
        for goos, goarch in TARGETS:
            self.assertIn(f"goos: {goos}", self.text, f"{goos} target missing")
            self.assertIn(f"goarch: {goarch}", self.text, f"{goarch} arch missing")

    def test_static_no_cgo_build(self) -> None:
        # Static, reproducible, no cgo — the property that lets the binary run
        # anywhere and the distroless image stay tiny.
        self.assertIn('CGO_ENABLED: "0"', self.text)
        self.assertIn("-trimpath", self.text)

    def test_stamps_version_ldflag(self) -> None:
        # A shipped binary with no VERSION file alongside resolves its version from
        # this ldflag (internal/appversion.Current precedence).
        self.assertIn(LDFLAG, self.text)

    def test_uploads_idempotently_with_checksums(self) -> None:
        self.assertIn("gh release upload", self.text)
        self.assertIn("--clobber", self.text)
        self.assertIn("sha256sum", self.text)
        self.assertIn("SHA256SUMS", self.text)

    def test_uses_module_go_version(self) -> None:
        self.assertIn("go-version-file: go.mod", self.text)

    def test_write_permission_scoped_to_upload_jobs(self) -> None:
        # The top-level token is read-only; only the jobs that touch the release
        # escalate to contents: write.
        self.assertIn("permissions:\n  contents: read", self.text)
        self.assertIn("contents: write", self.text)

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
        self.assertIn(LDFLAG, self.text)

    def test_serves_on_all_interfaces(self) -> None:
        # Containers must bind 0.0.0.0, not loopback.
        self.assertIn("0.0.0.0:8080", self.text)
        self.assertIn("EXPOSE 8080", self.text)


if __name__ == "__main__":
    unittest.main()
