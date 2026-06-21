package agent

// The agentic loop drives the REAL kernel, so it needs the runtime backends
// registered before it runs: the blob store (the Ref Resolver every tool call
// resolves through), the vDSO fast path (dedup), and the context-MMU (the
// write-time result admitter that quarantines poisoned results). They register
// via their own init() on blank import. grammar / preflight / adjudicator are
// already direct imports (tools.go configures them), so they need no entry here.
//
// This makes the agent package self-contained — usable from a test binary that
// does not blank-import internal/registrations — without the loop having to know
// which concrete drivers back each seam.
import (
	_ "github.com/anthony-chaudhary/fak/internal/blob"   // the Ref Resolver (CAS backend)
	_ "github.com/anthony-chaudhary/fak/internal/ctxmmu" // write-time result admission (quarantine)
	_ "github.com/anthony-chaudhary/fak/internal/vdso"   // 3-tier local fast path (dedup)
)
