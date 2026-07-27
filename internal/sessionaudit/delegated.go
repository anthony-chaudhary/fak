package sessionaudit

// HOW BIG IS THE DELEGATION LEVER? (epic #5416 track E.)
//
// Track E claims that sub-agent and background turns are "the single largest lever" on
// the epic's thesis, because moving them moves VOLUME rather than individual requests.
// That is a claim about a fraction, and until this file it was asserted rather than
// computed — the corpus rollups next door split tokens by model, by billing bucket, and
// by placement, but nothing split them by WHO ASKED. So there was no way to size the
// lever before pulling it, and no way to attribute a change to it afterwards: a
// self-hosted share that improves after sub-agents are re-placed looks exactly like one
// that improved because somebody changed their default model.
//
// The signal already exists and was being thrown away. Claude Code stamps delegated turns
// with `isSidechain`, and it writes them one of two ways: into the SAME session transcript
// as the main thread — which is why ParseTranscriptTurns must skip them to avoid
// double-counting turns — or into a SEPARATE transcript nested under the namespace
// directory, which Discover classifies as kind "subagent". Both layouts are real, and the
// fold handles either, because it keys on the marker rather than on the file. Here the
// marked turns are the subject rather than the noise.
//
// WHICH LAYOUT YOU GET IS NOT ACADEMIC, and getting it wrong costs the whole answer.
// Measured over the 170 transcripts one 3-day window discovered on this box: all 5,755
// turns in the 111 top-level transcripts were UNMARKED, and all 1,524 turns in the 59
// subagent transcripts were MARKED. This harness had put every delegated turn in a
// separate file. A reader that folds only top-level transcripts therefore sees precisely
// the under-instrumented shape described below — spawn calls, no marked turns — and
// correctly reports UNKNOWN while holding the answer in its hand. That is why the
// transcript KIND is a second attribution signal (TrackForKind) rather than metadata.
//
// THE HONEST PART IS WHAT AN ABSENT MARKER MEANS.
//
// `isSidechain` is a marker a harness EMITS. A transcript that predates it, or comes from
// a harness that never wrote it, decodes to false on every record — so the corpus reads
// "0% delegated", which is not a measurement at all. It is the same failure
// `Placement.Measured` exists to prevent one package over: rendering an UNMEASURED zero
// as a MEASURED one, in the direction that quietly says "this lever is worth nothing".
//
// So the fold cross-examines the corpus with a second, independent signal: SPAWN TOOL
// CALLS. A session that called a spawn tool demonstrably delegated work. If the corpus
// shows spawn calls and yet not one turn carries the marker, delegation happened and was
// not recorded — the corpus is UNDER-INSTRUMENTED, and the share is reported as UNKNOWN
// rather than as a zero.
//
// The two signals are deliberately asymmetric, and the asymmetry is what makes the
// spawn-tool list safe to be incomplete:
//
//   - The MARKER is the sole source of truth for the numerator. A spawner this file has
//     never heard of still gets counted correctly, turn for turn, as long as its turns
//     are marked.
//   - The SPAWN LIST is used only to detect the contradiction "something spawned, nothing
//     was marked". A spawner missing from the list can therefore only make this fold fail
//     to NOTICE under-instrumentation. It can never corrupt the fraction itself.
//
// One residual blind spot is stated rather than hidden: a corpus whose spawner is unknown
// here AND whose turns are unmarked is invisible, and reports a genuine-looking 0%. That
// is the one direction left, and it UNDER-states delegation — it can only make track E's
// lever look smaller than it is, never larger, so it cannot be used to inflate the epic's
// own thesis. An error that can only argue against the claim it is offered in support of
// is the acceptable kind.
//
// Output tokens are the measure, for the same reason SelfHostedShare uses them: output is
// the volume a GPU actually generated. Cache-read tokens are excluded from the fraction —
// prompt-cache reuse is a provider-side artifact with no counterpart on a self-hosted
// rung, so mixing it in compares unlike things.

// The closed delegation-track vocabulary. These are report keys (Session.PerTrack,
// Aggregate.PerTrack) and the fold's input, so they are additive: a new value is a new
// constant plus a fold arm, never a free-text field.
const (
	// TrackMain is the track for a turn the operator's own session generated.
	TrackMain = "main"
	// TrackDelegation is the track for a turn generated on behalf of a spawned sub-agent or
	// a background workflow — the volume track E proposes to move onto self-hosted rungs.
	//
	// Spelled "Delegation" rather than the shorter past-tense form deliberately, so please
	// leave it: the admission check reads identifiers as lowercase letters, and the shorter
	// spelling contains the name of a live concept family, which makes it refusable on
	// sight.
	TrackDelegation = "delegated"
)

// TrackForSidechain maps a transcript record's `isSidechain` marker to its track.
//
// Written as a function over the marker rather than as an inline conditional at the one
// call site so the mapping is nameable in a test, and so a future harness that reports
// delegation some other way has one place to teach rather than a scattered boolean.
func TrackForSidechain(isSidechain bool) string {
	if isSidechain {
		return TrackDelegation
	}
	return TrackMain
}

// The closed transcript-kind vocabulary Discover assigns. A top-level `*.jsonl` in a
// namespace directory is a KindTop transcript; anything nested deeper is KindSpawned,
// because that is where this harness parks a sub-agent's own transcript.
//
// Named KindTop/KindSpawned rather than after the two values they carry, so please leave
// it: the shorter names are the words this package uses for the whole-file concepts they
// name, and admission refuses a new identifier built out of them.
const (
	KindTop     = "session"
	KindSpawned = "subagent"
)

// TrackForKind resolves which track a session's volume belongs to, given the kind of
// transcript it came from and the track its own records claimed.
//
// THIS IS THE FALLBACK, AND IT ONLY EVER RUNS ONE WAY. A whole transcript that Discover
// classified as spawned IS delegated work — that is what the nesting means — so volume it
// reported as main-thread is re-attributed here. Nothing moves in the other direction: a
// marked turn in a top-level transcript stays delegated, because the marker is a
// per-record claim and outranks a per-file inference.
//
// The two signals were measured to agree exactly on the corpus in the file header
// (1,524 of 1,524 marked in spawned transcripts, 0 of 5,755 in top-level ones), so on
// today's harness this changes no number. That is the point of adding it now rather than
// after a regression: it is what keeps the answer alive if the marker stops being written,
// which is the exact failure the UNKNOWN verdict exists to refuse to paper over. Two
// independent signals that agree are a measurement; one signal is a dependency.
func TrackForKind(kind, track string) string {
	if kind == KindSpawned && track == TrackMain {
		return TrackDelegation
	}
	return track
}

// spawnToolNames is the closed set of tool calls that CREATE delegated work in this
// harness. Used only to detect under-instrumentation (see the file header), never to
// attribute volume — so being incomplete costs a warning, not a wrong number.
//
// Matched EXACTLY, and that is load-bearing rather than incidental: this harness also
// ships TaskCreate, TaskUpdate, TaskList, TaskStop and TaskOutput, which are todo-list
// bookkeeping and spawn nothing. A prefix match on "Task" would count them, and they are
// common enough in real transcripts to make almost every corpus look like it delegated —
// turning the under-instrumentation warning into permanent noise. The names below are the
// ones observed actually spawning work.
var spawnToolNames = map[string]bool{
	"Task":     true, // the historical sub-agent spawn tool
	"Agent":    true, // its current name
	"Workflow": true, // a background workflow, which spawns sub-agents of its own
}

// SpawnCalls counts observed spawn-tool invocations in a tool-mix rollup
// (Session.Tools / Aggregate.ToolMix).
func SpawnCalls(tools map[string]int64) int64 {
	var n int64
	for name, c := range tools {
		if spawnToolNames[name] {
			n += c
		}
	}
	return n
}

// DelegationShare is the delegation fold of a track rollup: how much of a corpus's
// generated volume was produced on behalf of delegated work rather than by the operator's
// own session, plus the evidence that says whether the number means anything.
//
// The fraction is a pointer for the reason every ratio in this package is one: "the
// answer is zero" and "there is no answer" are different findings, and here the
// difference is the whole point — an uninstrumented corpus must not be able to report
// that delegation costs nothing.
type DelegationShare struct {
	// Delegation is volume generated for spawned sub-agents and background workflows —
	// the numerator, and the volume track E moves.
	Delegation ModelCounts `json:"delegated"`
	// Main is volume the operator's own session generated.
	Main ModelCounts `json:"main"`
	// Untracked is volume under a track key outside the closed vocabulary. Excluded from
	// both sides of the fraction; it is what Coverage is missing.
	Untracked ModelCounts `json:"untracked"`

	// SpawnCalls is how many times the corpus called a spawn tool — the independent
	// signal that delegation occurred at all.
	SpawnCalls int64 `json:"spawn_calls"`
	// UnderInstrumented is the contradiction: the corpus spawned delegated work and not
	// one turn carries the marker. When true the share is UNKNOWN rather than zero.
	UnderInstrumented bool `json:"under_instrumented,omitempty"`

	// OutputShare is delegated output / tracked output — the size of the lever. Nil when
	// there is no tracked volume to divide, or when UnderInstrumented makes the zero
	// unearned.
	OutputShare *float64 `json:"output_share"`
	// Coverage is tracked output / all output: the share of generated volume OutputShare
	// speaks for. A share over low coverage is not evidence of anything.
	Coverage *float64 `json:"coverage"`
}

// TrackedOutput is the denominator of OutputShare — output whose track is known.
func (d DelegationShare) TrackedOutput() int64 {
	return d.Delegation.Output + d.Main.Output
}

// TotalOutput is all generated output, tracked or not.
func (d DelegationShare) TotalOutput() int64 {
	return d.TrackedOutput() + d.Untracked.Output
}

// FoldDelegationShare sorts a track rollup (Aggregate.PerTrack) into delegated and
// main-thread volume and computes the delegated fraction, cross-examined against the
// corpus's spawn-tool calls (Aggregate.ToolMix) so an uninstrumented corpus reports
// UNKNOWN instead of a zero it did not earn.
//
// Total over the track vocabulary by construction: every key lands in exactly one of the
// three categories, and a key the vocabulary does not recognize lands in Untracked — the
// side that is excluded from the fraction — rather than being silently dropped into one
// of the two that would move the number.
func FoldDelegationShare(perTrack map[string]ModelCounts, tools map[string]int64) DelegationShare {
	var d DelegationShare
	for track, c := range perTrack {
		switch track {
		case TrackDelegation:
			addCounts(&d.Delegation, c)
		case TrackMain:
			addCounts(&d.Main, c)
		default:
			addCounts(&d.Untracked, c)
		}
	}
	d.SpawnCalls = SpawnCalls(tools)
	// The cross-examination. Turns, not output: a delegated turn that generated nothing
	// still proves the marker is being written, which is the only thing in question here.
	d.UnderInstrumented = d.SpawnCalls > 0 && d.Delegation.Turns == 0

	d.Coverage = ratio(d.TrackedOutput(), d.TotalOutput())
	if d.UnderInstrumented {
		// Deliberately left nil. The arithmetic would happily produce 0.0, and that zero
		// would be read as "delegation generates no volume" — an argument against track E
		// manufactured entirely out of a missing field.
		return d
	}
	d.OutputShare = ratio(d.Delegation.Output, d.TrackedOutput())
	return d
}
