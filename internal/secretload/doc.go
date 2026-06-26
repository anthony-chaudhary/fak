// Package secretload is fak's first-class secret/config loader — the structural seam for
// pillars [C] (.env / config loading) and [D] (the vault backend) of the secret-handling
// epic (#880, foundation issue #887).
//
// Before this package, credentials entered the kernel through ~87 scattered os.Getenv
// callers and hand-rolled file reads (resolveAnthropicOAuthToken walked three sources by
// hand; the gateway key, account registry, and store tokens each did their own thing).
// There was no priority-ordered loader and — the load-bearing gap — no SecretSource seam,
// so a vault backend (HashiCorp Vault KV2 / AWS Secrets Manager / Azure Key Vault) would
// have been a rewrite rather than an addition.
//
// The package is built around one interface, SecretSource, and one resolver, Loader:
//
//   - SecretSource is a single priority-ordered origin of secret material (Lookup). The
//     two backends shipped here are OSEnv (os.LookupEnv-backed) and EncryptedFile
//     (AES-256-GCM at rest, key stretched from a passphrase with PBKDF2-HMAC-SHA256). A
//     .env backend (DotEnv) is the third, opt-in source (#889). An external vault is an
//     explicit follow-on added BEHIND this same interface — purely additive, never a
//     rewrite. Building the loader as a []SecretSource from day one is the single most
//     important structural decision in the epic; it lives here.
//
//   - Loader queries its sources in priority order (first hit wins), tracks a
//     required-secret checklist (Require), and reports missing/invalid required secrets
//     LOUDLY at startup (StartupReport) instead of failing late inside a request.
//
//   - Redact masks any known secret value (and any canon.SecretPatterns shape) in a
//     string so a credential can never ride a log line. It delegates shape detection to
//     internal/canon — the one detector both normgate and recall share — and never forks
//     it.
//
// Mechanism, not policy: the package resolves and masks secrets; WHERE a passphrase comes
// from, WHICH sources are wired, and WHO may read a value stay the caller's decision.
// Tier-1 foundation, stdlib-only (no external crypto dependency), off the hot path. This
// issue stands up the package, the interface, the two backends, and the checklist/redact
// helpers; migrating the existing callers through it is the separate refactor #888.
package secretload
