package corelockgate

import (
	"bytes"
	"fmt"
	"io"
	"os"
)

// binarygrammar.go — "is the fak executable in front of this worker OLD ENOUGH to be
// missing the `changed:` verb?" (#6005)
//
// THE FAILURE. `changed:<path>` arrived with changedwitness.go (ffc676b46b). A fak
// binary built before it has no such kind, so the claim falls through to the SHARED
// witness resolver, whose grammar has never contained `changed:` — and an unrecognized
// kind ABSTAINS. Abstain is not clearance, so an exactly-correct claim keeps the lock
// closed and the refusal reads `CORE_SELF_MODIFY ... resolved abstain`. Worse, the
// remedy that binary prints is CoreLockRemedyCommit as it existed BEFORE the verb: one
// sentence, naming no cure that can work for a file the change ADDS. A maintainer is
// therefore told to supply a witness, given no verb that can satisfy it, and shown a
// refusal indistinguishable from a badly-spelled claim. Hit live while landing #5933.
//
// WHY THE QUESTION HAS TO BE ASKED OF THE FILE. From inside the refusal, "stale tool"
// and "bad claim" look identical, and no code added to the gate can fix that — the
// binary doing the refusing is precisely the one that does not contain the new code.
// The only vantage point that is never stale is SOURCE: the probe below asks the
// question of an executable ON DISK, from a checkout, so the answer comes from a
// revision that by construction knows what the current grammar is. That is also why
// the enforcement point this ships with is a test (binarygrammar_test.go) rather than a
// banner inside some launcher binary: a compiled launcher can be stale itself, which
// would reproduce this very defect one level up.
//
// WHY A CAPABILITY PROBE AND NOT AN mtime OR VCS-STAMP CHECK. internal/repoguard
// deliberately DROPPED its mtime staleness walk (shim_test.go) because a timestamp is a
// proxy that fires on cases `make build` already prevents; internal/versionskew answers
// the richer question "which commit" but needs a git repo in the process cwd and
// collapses to UNKNOWN without one. The question here is narrower and needs neither:
// every binary that has the verb embeds the verb's own remedy sentence verbatim, so the
// executable IS the evidence. No git, no clock, no build metadata.
//
// THREE-VALUED, AND IT NEVER INVENTS A SKEW. The same posture as the stallscan skew
// guard (cmd/fak/stallscan_skew.go): STALE is claimed only when the file positively
// shows itself to be a fak binary of an older grammar — the anchor sentence present
// since the gate was born but not the verb. A file with neither marker is UNKNOWN and
// says nothing at all, so a stripped, packed, compressed or simply unrelated executable
// is never accused.

const (
	// GrammarChangedMarker is the byte sequence a fak binary embeds if and only if its
	// core-lock grammar knows `changed:`. It is a verbatim slice of
	// CoreLockRemedyCommit's second sentence, which landed in the SAME commit as the
	// verb — so it is exactly the thing a pre-verb binary provably cannot print.
	// TestGrammarMarkersAreSlicesOfTheRemedy pins the substring relation, so a reworded
	// remedy cannot silently classify every binary in the fleet as stale.
	GrammarChangedMarker = "name it with " + ChangedWitnessKind + ":<path>"

	// GrammarAnchorMarker is the byte sequence EVERY fak binary carrying the hard-self
	// core-lock gate embeds: the remedy's first sentence, unchanged since the gate was
	// born (d0f14083d9, when it still lived in internal/safecommit). It is what
	// separates "an old fak" from "not a fak" — without it the probe declines to judge
	// rather than calling a stranger's executable stale.
	GrammarAnchorMarker = "rerun fak commit with --core-lock-maintenance-witness <claim>"
)

// BinaryGrammar is the closed set of answers the probe can give about one executable.
// It is a string type so a caller can put the token straight into a JSON record without
// a mapping table, the way stallBuildSkew carries its verdict.
type BinaryGrammar string

const (
	// GrammarCurrent: the file carries the `changed:` verb. A correct change-relative
	// claim resolves CONFIRMED through it.
	GrammarCurrent BinaryGrammar = "CURRENT"
	// GrammarStale: the file is a fak binary with the core-lock gate but WITHOUT the
	// verb. A correct `changed:` claim abstains through it, and its own refusal cannot
	// name the cure.
	GrammarStale BinaryGrammar = "STALE"
	// GrammarUnknown: neither marker is present, or the file could not be read. The
	// honest residual — the probe establishes nothing and the caller should say nothing.
	GrammarUnknown BinaryGrammar = "UNKNOWN"
)

// grammarScanChunk is the read size of one probe window. A fak binary is ~70MB, so the
// scan is streamed rather than slurped; 1MiB keeps the syscall count low without the
// probe itself becoming a memory event on a fleet node running many of them.
const grammarScanChunk = 1 << 20

// ScanBinaryGrammar classifies an executable's core-lock grammar from its bytes. It
// reads r to EOF (or until the verb is found, whichever is first) and never seeks, so
// the caller may hand it any stream.
//
// A read error returns (GrammarUnknown, err): a probe that could not finish has
// established nothing, and reporting a partial scan as STALE would be exactly the false
// accusation the three-valued design exists to avoid.
func ScanBinaryGrammar(r io.Reader) (BinaryGrammar, error) {
	changed := []byte(GrammarChangedMarker)
	anchor := []byte(GrammarAnchorMarker)
	// Carry the longest-marker-minus-one bytes across the chunk boundary, so a marker
	// straddling two reads is still matched. Without it the probe's answer would depend
	// on where the OS happened to split the file.
	overlap := len(changed)
	if len(anchor) > overlap {
		overlap = len(anchor)
	}
	overlap--

	buf := make([]byte, grammarScanChunk+overlap)
	held := 0 // bytes carried over from the previous window, at buf[:held]
	sawAnchor := false
	for {
		n, err := r.Read(buf[held:])
		if n > 0 {
			window := buf[:held+n]
			if bytes.Contains(window, changed) {
				return GrammarCurrent, nil
			}
			if !sawAnchor && bytes.Contains(window, anchor) {
				sawAnchor = true
			}
			if len(window) > overlap {
				held = overlap
				copy(buf[:held], window[len(window)-overlap:])
			} else {
				held = len(window)
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return GrammarUnknown, err
		}
	}
	if sawAnchor {
		return GrammarStale, nil
	}
	return GrammarUnknown, nil
}

// BinaryGrammarAt classifies the executable at path. An unopenable path is
// (GrammarUnknown, err) for the same reason a short read is: nothing was established.
func BinaryGrammarAt(path string) (BinaryGrammar, error) {
	f, err := os.Open(path)
	if err != nil {
		return GrammarUnknown, err
	}
	defer f.Close()
	return ScanBinaryGrammar(f)
}

// StaleBinaryNote is the operator-facing sentence for a STALE verdict, and "" for every
// other verdict — so a caller can write `if note := StaleBinaryNote(p, g); note != ""`
// and stay silent on CURRENT and UNKNOWN without a second condition.
//
// It names STALENESS first, because the whole cost of this defect is that the failure
// presents as a bad claim: the maintainer re-spells the witness instead of rebuilding
// the tool. It then names the cure the stale binary's own remedy text cannot.
func StaleBinaryNote(path string, g BinaryGrammar) string {
	if g != GrammarStale {
		return ""
	}
	return fmt.Sprintf(
		"STALE fak binary: %s was built before the %s:<path> maintenance witness existed, so a CORRECT %s: claim resolves ABSTAIN through it (an unrecognized kind abstains, and an abstain is not clearance) and its CORE_SELF_MODIFY refusal cannot name the one verb that works for a file the change ADDS. "+
			"The refusal will look like a bad claim rather than a stale tool. Rebuild it (`go build -o %s ./cmd/fak`) before landing a change under a hard-self core-lock path.",
		path, ChangedWitnessKind, ChangedWitnessKind, path)
}
