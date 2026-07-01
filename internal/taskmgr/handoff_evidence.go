package taskmgr

import (
	"fmt"
	"path/filepath"
	"strings"
)

// TestRefKind is the EvidenceRef.Kind for a targeted command that witnessed the
// task, usually a focused go test invocation.
const TestRefKind = "test"

// HandoffEvidenceInputs are raw producer-side signals collected while a task is
// running. DraftHandoffFromTask folds them into typed EvidenceRefs before an
// operator edits or syncs the handoff.
type HandoffEvidenceInputs struct {
	ChangedPaths       []string `json:"changed_paths,omitempty"`
	TestCommands       []string `json:"test_commands,omitempty"`
	GeneratedArtifacts []string `json:"generated_artifacts,omitempty"`
}

// HandoffDraftOptions controls the handoff draft generated from a live task
// snapshot.
type HandoffDraftOptions struct {
	CurrentState        string
	Summary             string
	CompletedBy         string
	Labels              map[string]string
	CompletionEvidence  []EvidenceRef
	CompletionTimestamp int64
	Evidence            HandoffEvidenceInputs
}

// DraftHandoffFromTask creates the editable task-handoff record from the current
// task snapshot and producer-side evidence signals. It is intentionally usable for
// a running task: ReviewHandoff may later refuse that draft as incomplete, but the
// changed-path/test/artifact refs are already present at the origin.
func DraftHandoffFromTask(task TaskSnapshot, opt HandoffDraftOptions) Handoff {
	refs := make([]EvidenceRef, 0, len(task.EvidenceRefs)+len(opt.CompletionEvidence))
	refs = append(refs, task.EvidenceRefs...)
	if task.Witness != nil {
		refs = append(refs, task.Witness.EvidenceRefs...)
	}
	for _, step := range task.Steps {
		refs = append(refs, step.EvidenceRefs...)
		if step.Witness != nil {
			refs = append(refs, step.Witness.EvidenceRefs...)
		}
	}
	refs = append(refs, opt.CompletionEvidence...)
	refs = append(refs, DeriveHandoffEvidenceRefs(opt.Evidence)...)

	return Handoff{
		Schema: SchemaHandoff,
		Task: HandoffTask{
			TaskID:  task.TaskID,
			Title:   task.Title,
			State:   task.State,
			Witness: cloneWitness(task.Witness),
		},
		CurrentState:        firstNonEmpty(opt.CurrentState, fmt.Sprintf("Task %s is %s.", task.TaskID, task.State)),
		Summary:             strings.TrimSpace(opt.Summary),
		CompletedBy:         strings.TrimSpace(opt.CompletedBy),
		Labels:              cloneLabels(opt.Labels),
		CompletionEvidence:  dedupeHandoffEvidenceRefs(refs),
		CompletionTimestamp: opt.CompletionTimestamp,
	}
}

// DeriveHandoffEvidenceRefs turns raw producer signals into bounded, typed refs.
func DeriveHandoffEvidenceRefs(in HandoffEvidenceInputs) []EvidenceRef {
	var refs []EvidenceRef
	for _, path := range compactStrings(in.ChangedPaths) {
		if ref := normalizedPathRef(path, "changed path"); ref.Ref != "" {
			refs = append(refs, ref)
		}
	}
	for _, cmd := range compactStrings(in.TestCommands) {
		if cmd = strings.Join(strings.Fields(cmd), " "); cmd != "" {
			refs = append(refs, EvidenceRef{Kind: TestRefKind, Ref: cmd, Note: "targeted test command"})
		}
	}
	for _, path := range compactStrings(in.GeneratedArtifacts) {
		if ref := normalizedPathRef(path, "generated artifact"); ref.Ref != "" {
			refs = append(refs, ref)
		}
	}
	return dedupeHandoffEvidenceRefs(refs)
}

func normalizedPathRef(path, note string) EvidenceRef {
	path = strings.TrimSpace(path)
	if path == "" {
		return EvidenceRef{}
	}
	path = filepath.ToSlash(filepath.Clean(path))
	if path == "." {
		return EvidenceRef{}
	}
	return EvidenceRef{Kind: PathRefKind, Ref: path, Note: note}
}

func dedupeHandoffEvidenceRefs(refs []EvidenceRef) []EvidenceRef {
	seen := map[string]bool{}
	out := make([]EvidenceRef, 0, len(refs))
	for _, ref := range refs {
		ref = handoffSafeEvidenceRef(ref)
		if ref.Kind == "" && ref.Ref == "" && ref.Note == "" {
			continue
		}
		key := ref.Kind + "\x00" + ref.Ref + "\x00" + ref.Note
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, ref)
	}
	return out
}

func handoffSafeEvidenceRef(ref EvidenceRef) EvidenceRef {
	ref.Kind = strings.TrimSpace(ref.Kind)
	ref.Ref = strings.TrimSpace(ref.Ref)
	ref.Note = strings.TrimSpace(ref.Note)
	if ref.Kind == OutputRefKind && ref.Ref != "" {
		note := fmt.Sprintf("output evidence omitted from handoff; %d bytes", len(ref.Ref))
		if ref.Note != "" {
			note = ref.Note + "; " + note
		}
		ref.Ref = ""
		ref.Note = note
	}
	return ref
}
