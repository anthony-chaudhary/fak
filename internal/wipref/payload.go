package wipref

import (
	"fmt"
	"sort"
	"strings"
)

// payload.go — the PER-PAYLOAD-FILE census of a checkpoint ref (#5940). It is pure:
// the cmd shell (cmd/fak/wip_census.go) runs the git plumbing and hands the raw bytes
// here; this file turns them into an A/M/- state per file and into the remedy that
// state permits. No git, no I/O.
//
// It exists because the per-REF classification in census.go answers a question that has
// no honest per-ref answer. A checkpoint that is 90% landed and 10% novel reads
// all-or-nothing, and the two ways it can read wrong BOTH read as good news:
//
// TRAP A — an unreadable ref reads as EMPTY. A checkpoint minted with no reachable
// parent (a parentless commit, or one whose parent the repo no longer has) makes
// `git diff <obj>^ <obj>` exit 128 with EMPTY STDOUT. A caller that reads only stdout
// takes that as "this ref holds nothing" and points at "safe to collect". So the
// payload read reports Read=false with a reason, and Classify maps that to UNKNOWN
// (kept). EMPTINESS IS NEVER INFERRED FROM AN EMPTY STDOUT — see PayloadReading.Read.
//
// TRAP B — "the blob differs from HEAD" is NOT "unlanded". The obvious per-file test
// compares the checkpoint's blob to HEAD's and calls any difference unlanded work.
// That is how you recommend `git checkout <ref> -- <path>` for a file HEAD already
// carries in a LARGER, LATER form: the checkout REVERTS the newer copy. Present-but-
// different is its own outcome — DIVERGED — because the only correct action on it is a
// three-way diff. PayloadRemedy refuses the checkout for exactly that state.
//
// The three states come from a TWO-DOT `git diff --name-status HEAD <obj> -- <payload>`.
// Two-dot compares the two trees directly and needs no merge base, so ONE code path
// serves a parentless ref and a descendant ref alike.

// PayloadState is where ONE payload file of a checkpoint stands relative to HEAD. The
// values are the git status letters the census reads, so the table a human sees and the
// value the code branches on are the same token.
type PayloadState string

const (
	// PayloadAbsent ("A"): HEAD does not carry this path AT ALL. This is the only
	// state that is true at-risk salvage — nothing on HEAD can be destroyed by
	// materializing it, and nothing on HEAD preserves it if the ref is collected.
	PayloadAbsent PayloadState = "A"
	// PayloadDiverged ("M"): HEAD HAS the path, with different content. It may be
	// older than the checkpoint, NEWER than the checkpoint, or a parallel rewrite;
	// this package deliberately does not guess. Landing it wholesale is the
	// data-loss path, so PayloadRemedy refuses a checkout here.
	PayloadDiverged PayloadState = "M"
	// PayloadLanded ("-"): HEAD's copy is byte-identical to the checkpoint's. Landed.
	PayloadLanded PayloadState = "-"
)

// ClassifyPayloadStatus maps one `git diff --name-status` letter to a PayloadState.
//
// The second return is false for a letter that describes no parked work at all. That is
// only 'D': HEAD HAS a path the checkpoint does not, which is the checkpoint being
// behind, not HEAD missing something. Folding D into either real state would inflate
// exactly the number this census exists to get right.
//
// An UNRECOGNISED letter also returns false rather than defaulting to PayloadLanded —
// "we did not recognise this" must never read as "already landed".
func ClassifyPayloadStatus(code byte) (PayloadState, bool) {
	switch code {
	case 'A':
		return PayloadAbsent, true
	// T is a type change (file <-> symlink) and R/C carry a similarity score: all
	// three mean HEAD has the path with DIFFERENT content, which is the diverged
	// case, not the absent one.
	case 'M', 'T', 'R', 'C':
		return PayloadDiverged, true
	}
	return PayloadLanded, false
}

// ParseNameStatus splits the raw output of a two-dot
// `git diff --name-status HEAD <obj> -- <payload>` into the paths HEAD lacks and the
// paths HEAD has but differs on. Paths absent from the output are byte-identical to
// HEAD and are therefore landed — git prints nothing for them.
//
// The status letter is read from the FIRST field ONLY. A rename shows as `R100` with
// TWO path columns, so taking fields[1] as the path would silently record the OLD name;
// renames resolve to their destination (the last field).
func ParseNameStatus(out string) (absent, diverged []string) {
	if strings.IndexByte(out, 0) >= 0 {
		fields := strings.Split(out, "\x00")
		for i := 0; i < len(fields); {
			if fields[i] == "" {
				i++
				continue
			}
			status := fields[i]
			i++
			if i >= len(fields) || fields[i] == "" {
				break
			}
			path := fields[i]
			i++
			switch status[0] {
			case 'A':
				absent = append(absent, path)
			case 'M':
				diverged = append(diverged, path)
			}
		}
		return absent, diverged
	}
	for _, line := range strings.Split(out, "\n") {
		if line = strings.TrimSuffix(line, "\r"); line == "" {
			continue
		}
		status, path, ok := strings.Cut(line, "\t")
		if !ok || status == "" || path == "" {
			continue
		}
		switch status[0] {
		case 'A':
			absent = append(absent, path)
		case 'M':
			diverged = append(diverged, path)
		}
	}
	return absent, diverged
}

// PayloadReading is the OUTCOME of the shell's attempt to read one ref's payload — not
// the payload itself. The distinction is the whole of trap A: a caller must be able to
// tell "this ref contributes no files" from "this ref could not be read", and those two
// look IDENTICAL if the only channel is a []string.
//
// Read is therefore an explicit positive: it is true only when EVERY plumbing call in
// the payload path exited 0. The zero value has Read=false, so a shell that forgets to
// set it reports unreadable — the fail-safe, kept side — and never a false empty.
type PayloadReading struct {
	// Read is true only when every plumbing call in the payload path exited 0.
	Read bool
	// Unreadable is why Read is false (the failing command and git's stderr). Empty
	// when Read is true.
	Unreadable string
	// Paths are the payload files: the files the checkpoint CONTRIBUTES. For a
	// descendant that is its diff against its parent; for a PARENTLESS ref there is
	// no parent to diff, so the shell reads the whole tree (`git ls-tree`) instead.
	Paths []string
	// NameStatus is the raw two-dot `git diff --name-status HEAD <obj> -- <paths>`
	// output, concatenated across pathspec chunks. Empty with Read=true and a
	// non-empty Paths means every payload file is byte-identical to HEAD.
	NameStatus string
}

// Unread builds the trap-A reading: a payload the shell could NOT measure, carrying the
// reason. It exists so the shell has no syntactically shorter way to report a failure
// than to report it honestly — returning a bare PayloadReading{} would be the same
// thing, and returning PayloadReading{Read: true} for a failed command is the bug.
func Unread(format string, args ...any) PayloadReading {
	return PayloadReading{Unreadable: fmt.Sprintf(format, args...)}
}

// PayloadCensus is one checkpoint ref's per-file A/M/- verdict.
type PayloadCensus struct {
	// Read mirrors PayloadReading.Read: false means UNMEASURED, never "empty".
	Read bool `json:"read"`
	// Unreadable is why Read is false.
	Unreadable string `json:"unreadable,omitempty"`
	// Files is the payload size. Meaningful ONLY when Read is true — with Read false
	// it is 0 because nothing was measured, which is precisely the reading that must
	// not be mistaken for "this ref holds nothing".
	Files int `json:"files"`
	// AbsentPaths are the payload files HEAD does not carry at all ("A") — the true
	// at-risk salvage, and the only files a checkout may safely materialize.
	AbsentPaths []string `json:"absent_paths,omitempty"`
	// DivergedPaths are the payload files HEAD carries with DIFFERENT content ("M").
	// They need a three-way diff and NEVER a checkout.
	DivergedPaths []string `json:"diverged_paths,omitempty"`
	// Landed is how many payload files are byte-identical to HEAD ("-").
	Landed int `json:"landed"`
}

// AtRisk reports whether this ref holds a file HEAD does not have at all. A DIVERGED-only
// ref is deliberately NOT at risk by this definition: HEAD has every one of its paths,
// and which side is newer is a question this package refuses to guess.
func (p PayloadCensus) AtRisk() bool { return len(p.AbsentPaths) > 0 }

// BuildPayloadCensus folds a reading into the per-file census. An UNREADABLE reading
// yields Read=false and NO counts at all — the counts are not zeroed as a measurement,
// they are absent because no measurement happened.
//
// Landed is computed by SUBTRACTION (payload minus absent minus diverged), because git
// prints nothing for a byte-identical path: the landed files are exactly the ones the
// name-status output does not mention. It is clamped at zero so a name-status naming a
// path outside the requested payload (which git does not do, but a caller that passed a
// mismatched pathspec could provoke) cannot produce a negative count.
func BuildPayloadCensus(r PayloadReading) PayloadCensus {
	if !r.Read {
		reason := r.Unreadable
		if reason == "" {
			reason = "payload not read (no reason recorded) — reported unmeasured, never empty"
		}
		return PayloadCensus{Read: false, Unreadable: reason}
	}
	absent, diverged := ParseNameStatus(r.NameStatus)
	p := PayloadCensus{
		Read:          true,
		Files:         len(r.Paths),
		AbsentPaths:   absent,
		DivergedPaths: diverged,
	}
	sort.Strings(p.AbsentPaths)
	sort.Strings(p.DivergedPaths)
	if n := p.Files - len(absent) - len(diverged); n > 0 {
		p.Landed = n
	}
	return p
}

// Facts projects the per-file census onto the CensusFacts fields Classify reads, so the
// shell cannot wire the payload into the classifier through any path that loses the
// read/unread distinction.
func (p PayloadCensus) Facts() (read bool, files, absent, diverged int) {
	if !p.Read {
		return false, 0, 0, 0
	}
	return true, p.Files, len(p.AbsentPaths), len(p.DivergedPaths)
}

// StateOf returns the A/M/- state of one payload path. A path the census does not name
// as absent or diverged is byte-identical to HEAD, hence landed.
func (p PayloadCensus) StateOf(path string) PayloadState {
	for _, a := range p.AbsentPaths {
		if a == path {
			return PayloadAbsent
		}
	}
	for _, d := range p.DivergedPaths {
		if d == path {
			return PayloadDiverged
		}
	}
	return PayloadLanded
}

// RefuseDivergedCheckout is the closed refusal token any collect path emits when it is
// asked to materialize a DIVERGED payload file.
const RefuseDivergedCheckout = "DIVERGED_NEEDS_THREE_WAY"

// RefusePayloadCheckout reports whether a per-file CHECKOUT of a payload file in state s
// (writing the ref's blob over HEAD's copy) must be REFUSED.
//
// Only PayloadDiverged refuses, and it refuses unconditionally: HEAD has the path with
// different content, the tool does not know which side is newer, and the checkout is
// silent and lossy when HEAD's is. PayloadAbsent does not refuse — HEAD has nothing
// there to destroy. PayloadLanded does not refuse because there is nothing to do.
func RefusePayloadCheckout(s PayloadState) bool { return s == PayloadDiverged }

// PayloadRemedy is the ONE safe command for a payload file in state s, and whether that
// command is a REFUSAL of the checkout the naive classifier would have recommended.
//
// The refusal is expressed by what the returned command IS, not by a warning beside it:
// for a DIVERGED file the command is the three-way review diff, so a caller that pipes
// PayloadRemedy's output to a shell cannot revert HEAD's newer copy even if it ignores
// the boolean entirely. That is the point — #605's landing recipe was unsafe precisely
// because the warning and the command lived in different places.
//
//	A  ->  git checkout <ref> -- <path>          (safe: HEAD has nothing there)
//	M  ->  git diff HEAD:<path> <ref>:<path>     (REFUSED as a checkout; review first)
//	-  ->  ""                                     (landed: nothing to do)
func PayloadRemedy(s PayloadState, ref, path string) (command string, refused bool) {
	switch s {
	case PayloadAbsent:
		return fmt.Sprintf("git checkout %s -- %s", ref, path), false
	case PayloadDiverged:
		return fmt.Sprintf("git diff HEAD:%s %s:%s", path, ref, path), true
	default:
		return "", false
	}
}

// PayloadRemedyNote is the human sentence behind PayloadRemedy's choice — the reason a
// DIVERGED file gets a review command and not a checkout.
func PayloadRemedyNote(s PayloadState) string {
	switch s {
	case PayloadAbsent:
		return "absent from HEAD — the only true at-risk salvage; a checkout can destroy nothing"
	case PayloadDiverged:
		return RefuseDivergedCheckout + ": HEAD carries this path with different content and may hold the NEWER copy — review the three-way diff; a checkout would silently revert it"
	default:
		return "byte-identical to HEAD — already landed, nothing to do"
	}
}

// PayloadRemedies renders the remedy line for every file a census names, absent files
// first (they are the ones that can be lost) then diverged, each already sorted. A
// landed file gets no line: there is nothing to do about it.
func PayloadRemedies(p PayloadCensus, ref string) []string {
	if !p.Read {
		return []string{fmt.Sprintf("%s: PAYLOAD UNREADABLE — %s (NOT an empty payload; kept)", ref, p.Unreadable)}
	}
	out := make([]string, 0, len(p.AbsentPaths)+len(p.DivergedPaths))
	for _, path := range p.AbsentPaths {
		cmd, _ := PayloadRemedy(PayloadAbsent, ref, path)
		out = append(out, fmt.Sprintf("A %s\t%s", path, cmd))
	}
	for _, path := range p.DivergedPaths {
		cmd, _ := PayloadRemedy(PayloadDiverged, ref, path)
		out = append(out, fmt.Sprintf("M %s\t%s", path, cmd))
	}
	return out
}
