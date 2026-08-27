# Local-app helper packaging

`Package.swift` deliberately builds an app-scoped helper and Swift client without embedding a general LAN service. The host app creates a random per-install capability, launches the helper on loopback, and supplies signed-host identity metadata. `internal/localapphelper` verifies that identity/capability binding before task admission.

A release must be packaged on macOS with a Developer ID Application identity:

```bash
./scripts/package-helper.sh \
  --identity "Developer ID Application: Example (TEAMID)" \
  --team-id TEAMID \
  --bundle-id com.example.JobApply.FakLocalAppHelper \
  --output ./dist \
  --notary-profile fak-notary
```

The script fails closed when signing, strict verification, notarization, stapling, or Gatekeeper assessment fails. It emits `dist/receipt.json` binding signing status, Team ID, bundle ID, archive name, and SHA-256. Omitting `--notary-profile` produces a typed `SIGNED` development artifact, never a claim of notarization.

The clean-Mac issue witness must retain the receipt, `codesign -dv --verbose=4`, `spctl --assess`, install-to-ready/task receipts, and the exact fak-native model artifact receipt. A green package receipt alone does not certify the product path.
