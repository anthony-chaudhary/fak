package wipref

import (
	"fmt"
	"sort"
	"strings"
)

const RemoteWIPPrefix = "refs/fak/wip/"

type RemoteDrainState string

const (
	RemoteDrainSafe    RemoteDrainState = "SAFE_TO_DELETE"
	RemoteDrainKeep    RemoteDrainState = "KEEP_UNLANDED"
	RemoteDrainPeer    RemoteDrainState = "KEEP_PEER"
	RemoteDrainUnknown RemoteDrainState = "KEEP_UNKNOWN"
)

type RemoteDrainCandidate struct {
	Session        string           `json:"session"`
	Ref            string           `json:"ref"`
	SHA            string           `json:"sha"`
	State          RemoteDrainState `json:"state"`
	Owned          bool             `json:"owned"`
	DeltaContained bool             `json:"delta_contained"`
	Reason         string           `json:"reason"`
	DeleteRefspec  string           `json:"delete_refspec,omitempty"`
}

type RemoteRef struct{ Ref, SHA string }

// PlanRemoteDrain is pure policy: age is intentionally absent. A checkpoint is
// deletable only after remote branch bytes independently witness its entire delta.
func PlanRemoteDrain(refs []RemoteRef, owned map[string]bool, contained func(RemoteRef) (bool, error), allowPeer bool) []RemoteDrainCandidate {
	out := make([]RemoteDrainCandidate, 0, len(refs))
	for _, ref := range refs {
		session := strings.TrimPrefix(ref.Ref, RemoteWIPPrefix)
		row := RemoteDrainCandidate{Session: session, Ref: ref.Ref, SHA: ref.SHA, Owned: owned[session]}
		if session == ref.Ref || session == "" || strings.Contains(session, "/") {
			row.State, row.Reason = RemoteDrainUnknown, "malformed checkpoint ref; keep"
		} else if !row.Owned && !allowPeer {
			row.State, row.Reason = RemoteDrainPeer, "peer checkpoint requires --allow-peer"
		} else if ok, err := contained(ref); err != nil {
			row.State, row.Reason = RemoteDrainUnknown, "containment witness failed: "+err.Error()
		} else if !ok {
			row.State, row.Reason = RemoteDrainKeep, "checkpoint delta is not contained in the remote default branch"
		} else {
			row.State, row.DeltaContained, row.Reason, row.DeleteRefspec = RemoteDrainSafe, true, "checkpoint delta is contained in the remote default branch", ":"+ref.Ref
		}
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Ref < out[j].Ref })
	return out
}

func ParseRemoteRefs(text string) ([]RemoteRef, error) {
	var out []RemoteRef
	for _, line := range strings.Split(strings.TrimSpace(text), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Fields(line)
		if len(f) != 2 || !strings.HasPrefix(f[1], RemoteWIPPrefix) {
			return nil, fmt.Errorf("invalid remote checkpoint row %q", line)
		}
		out = append(out, RemoteRef{SHA: f[0], Ref: f[1]})
	}
	return out, nil
}
