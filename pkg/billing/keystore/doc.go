// Package keystore is fak's durable tenant-key store for multi-tenant token
// metering: a versioned, file-backed registry that binds an opaque
// credential to a tenant and tracks its lifecycle (active / revoked).
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/billing/keystore)
// on purpose, beside the neutral usage record in pkg/billing and the ledger
// mechanics in pkg/billing/ledger. Placement follows Plan C ("off-the-shelf
// public spine"), billing-modularization lane, Phase 3 item 11 (companion
// private repository,
// docs/tickets/billing-modularization/PLAN-C-off-the-shelf-public-spine.md):
// the durable tenant-key store that the operator key verbs ride.
//
// This package is pure MECHANISM. It contains NO pricing and NO policy, and
// no default may encode a billing choice:
//
//   - Keys only. The package mints, validates, revokes, and lists keys.
//     BALANCES ARE NOT STORED HERE: a tenant's balance rides
//     pkg/billing/ledger (Deposit / GetAccount) at the operator-CLI layer,
//     which joins a validated key to a tenant account. This package answers
//     only "does this credential belong to this tenant, and is it still
//     active?".
//   - Tenants are opaque. TenantID is carried verbatim and never
//     interpreted; the package neither mints nor registers tenants.
//   - No admission or entitlement decisions: whether a request may proceed
//     is the caller's verdict, built on Validate's typed outcome.
//
// Hash at rest. Mint generates a high-entropy key — sk-fak- followed by 32
// bytes of crypto/rand as lowercase hex — and returns the FULL plaintext
// exactly once, at mint time. The store persists only the SHA-256 digest of
// the key and a short display prefix; the plaintext is unrecoverable by
// design and appears nowhere in the store file.
//
// Durability. Open(path) loads the store from a single JSON file. Every
// mutation rewrites the whole file ATOMICALLY through pkg/fsatomic (sibling
// temporary file, fsync, rename), so a crash or a failed write never leaves
// a torn file in place. Mutations are durable-first: in-memory state commits
// only after the atomic rewrite succeeds. Open on a MISSING FILE returns an
// EMPTY store and creates no file; the file comes into existence on the
// first mutation.
//
// The file format is closed: a version stamp (KeystoreVersionV1) and records
// that decode strictly to the KeyRecord schema — unknown fields are refused,
// mirroring pkg/billing/pricebook. A store file that is not well-formed is
// refused with ErrCorruptStore and left untouched.
//
// Concurrency. All exported methods are safe for concurrent use; concurrent
// Mint / Validate / Revoke are race-clean under a single mutex.
//
// The package depends only on the standard library and pkg/fsatomic, and
// holds no global state: a Store is constructed by its owner via Open.
package keystore
