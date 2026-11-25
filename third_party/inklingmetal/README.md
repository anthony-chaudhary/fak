# Inkling Metal operators (vendored)

This directory is a vendored third-party runtime. FAK did not author it. The C++ target, draft, Metal command buffer, and shader operators under `runtime/` are copied from [incoai/splash](https://github.com/incoai/splash) at commit `2afc7337aacc044722fa95e4cb3da75c281a19a8` under Apache-2.0. The adjacent `LICENSE` and `THIRD_PARTY_NOTICES` are the upstream files, unchanged.

FAK's own code is the in-process adapter in `internal/inklingmetal`: it loads model tensors, drives each token and candidate block, selects tokens, and commits accepted request state.

## Provenance

`PINNED-SOURCE.json` is the machine-checkable record. It names the upstream repository, commit, and license, and lists every file under `runtime/` with its size, SHA-256, and one of three statuses:

| Status | Meaning |
| --- | --- |
| `identical` | Byte-identical to the upstream object at the pinned commit. |
| `modified` | Changed locally. The entry also records `upstream_sha256` and a one-line `local_change`. |
| `fak-added` | Written by FAK. No upstream object exists. |

At the pinned commit, 272 files are identical, 1 is modified (`runtime/metal/MetalBackend.mm`), and 2 are FAK-added (`runtime/metal/FakEmbeddedShaders.hpp` and the generated `runtime/metal/FakShaderIndex.inc`). `upstream_runtime` records that the import is partial: 273 of the 377 upstream `runtime/` files were taken, and `not_imported_by_directory` counts the remaining 104.

## Verify

```sh
go test ./internal/inklingmetal -run Provenance -count=1
```

The test needs no cgo, GPU, or network. It fails when a file under `runtime/` is added, removed, or changed without a matching manifest entry, and when `LICENSE` or `THIRD_PARTY_NOTICES` drift.

A sparse checkout verifies only the files on disk; run the test in a full checkout to cover every vendored file.

To also compare every recorded upstream digest and the not-imported counts against the real upstream objects, point the test at a local clone of incoai/splash that contains the pinned commit:

```sh
FAK_INKLINGMETAL_UPSTREAM=/path/to/splash go test ./internal/inklingmetal -run Provenance -count=1
```

## Refresh the manifest

After any change under `runtime/`, regenerate the file list from the tree and the upstream clone:

```sh
FAK_INKLINGMETAL_UPSTREAM=/path/to/splash FAK_INKLINGMETAL_REFRESH=1 \
  go test ./internal/inklingmetal -run Provenance -count=1
```

The refresh derives sizes, digests, statuses, and counts. Three things stay hand-written and are preserved:

- `upstream_commit`: edit it first when moving to a new upstream revision, then refresh. Update the Splash revision in the root `THIRD_PARTY_NOTICES.md` in the same change.
- `local_change`: every `modified` file needs a one-line description. The test fails while one is blank.
- `fak-added`: the refresh keeps this status only for files that already carry it. A file that is new to the tree and absent upstream, or a previously `identical` or `modified` file whose upstream object is gone at the new pin, fails the refresh. Add the `fak-added` entry by hand for a file FAK wrote; resolve a vanished upstream file before refreshing.
- `notices`: the list of upstream license files kept beside the manifest.

## Shaders

Regenerate the shader translation-unit index with `go run ./internal/inklingmetal/cmd/genshaders` after modifying a shader, then refresh the manifest. The Metal device compiles embedded MSL source from these translation units.
