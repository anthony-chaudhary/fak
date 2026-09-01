package workerworktree

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	RemoteSnapshotSchema   = "fak-worker-lifecycle-snapshot/1"
	RemoteSnapshotRefRoot  = "refs/fak/worker-lifecycle/v1"
	RemoteSnapshotMaxRows  = 256
	RemoteSnapshotMaxBytes = 256 << 10
	RemoteSnapshotTTL      = 24 * time.Hour
)

type SnapshotAssociation struct {
	State   string `json:"state"`
	Lane    string `json:"lane"`
	LeaseID string `json:"lease_id,omitempty"`
}

type SnapshotLiveness struct {
	Owner string `json:"owner"`
	Lease string `json:"lease"`
}
type SnapshotCleanliness struct {
	State string `json:"state"`
}

type SnapshotRow struct {
	HeadSHA     string              `json:"head_sha"`
	BaseSHA     string              `json:"base_sha"`
	Association SnapshotAssociation `json:"association"`
	Liveness    SnapshotLiveness    `json:"liveness"`
	Cleanliness SnapshotCleanliness `json:"cleanliness"`
	Lifecycle   string              `json:"lifecycle"`
}

type RemoteSnapshot struct {
	Schema     string        `json:"schema"`
	Host       string        `json:"host"`
	ObservedAt time.Time     `json:"observed_at"`
	ExpiresAt  time.Time     `json:"expires_at"`
	Rows       []SnapshotRow `json:"rows"`
}

type SnapshotPublishResult struct {
	OK           bool   `json:"ok"`
	Applied      bool   `json:"applied"`
	Remote       string `json:"remote"`
	Ref          string `json:"ref"`
	Host         string `json:"host"`
	Rows         int    `json:"rows"`
	Bytes        int    `json:"bytes"`
	PreviousOID  string `json:"previous_oid,omitempty"`
	PublishedOID string `json:"published_oid,omitempty"`
	ReadBackOID  string `json:"read_back_oid,omitempty"`
	Reason       string `json:"reason,omitempty"`
}

type SnapshotFreshness string

const (
	SnapshotFresh   SnapshotFreshness = "FRESH"
	SnapshotStale   SnapshotFreshness = "STALE"
	SnapshotUnknown SnapshotFreshness = "UNKNOWN"
)

type RemoteSnapshotGroup struct {
	Host          string            `json:"host"`
	Provenance    string            `json:"provenance"`
	Freshness     SnapshotFreshness `json:"freshness"`
	ObservedAt    time.Time         `json:"observed_at,omitempty"`
	ExpiresAt     time.Time         `json:"expires_at,omitempty"`
	Schema        string            `json:"schema,omitempty"`
	Ref           string            `json:"ref"`
	Rows          []SnapshotRow     `json:"rows"`
	Reason        string            `json:"reason,omitempty"`
	Authoritative bool              `json:"authoritative"`
}

func SnapshotHostID(hostname string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(strings.ToLower(hostname))))
	return "host-" + hex.EncodeToString(sum[:6])
}

func SnapshotRef(host string) string { return RemoteSnapshotRefRoot + "/" + host }

func NewRemoteSnapshot(hostname string, observed time.Time, rows []SnapshotRow) (RemoteSnapshot, error) {
	if len(rows) > RemoteSnapshotMaxRows {
		return RemoteSnapshot{}, fmt.Errorf("snapshot rows %d exceed bound %d", len(rows), RemoteSnapshotMaxRows)
	}
	cpy := append([]SnapshotRow(nil), rows...)
	for i := range cpy {
		cpy[i] = boundSnapshotRow(cpy[i])
	}
	sort.Slice(cpy, func(i, j int) bool {
		return cpy[i].Association.Lane+"\x00"+cpy[i].HeadSHA < cpy[j].Association.Lane+"\x00"+cpy[j].HeadSHA
	})
	observed = observed.UTC().Truncate(time.Second)
	s := RemoteSnapshot{Schema: RemoteSnapshotSchema, Host: SnapshotHostID(hostname), ObservedAt: observed, ExpiresAt: observed.Add(RemoteSnapshotTTL), Rows: cpy}
	b, err := json.Marshal(s)
	if err != nil {
		return RemoteSnapshot{}, err
	}
	if len(b) > RemoteSnapshotMaxBytes {
		return RemoteSnapshot{}, fmt.Errorf("snapshot bytes %d exceed bound %d", len(b), RemoteSnapshotMaxBytes)
	}
	return s, nil
}

func boundSnapshotRow(r SnapshotRow) SnapshotRow {
	r.HeadSHA = boundText(r.HeadSHA, 64)
	r.BaseSHA = boundText(r.BaseSHA, 64)
	r.Association.State = boundText(r.Association.State, 24)
	r.Association.Lane = boundText(r.Association.Lane, 96)
	r.Association.LeaseID = boundText(r.Association.LeaseID, 128)
	r.Liveness.Owner = boundText(r.Liveness.Owner, 24)
	r.Liveness.Lease = boundText(r.Liveness.Lease, 24)
	r.Cleanliness.State = boundText(r.Cleanliness.State, 24)
	r.Lifecycle = boundText(r.Lifecycle, 24)
	return r
}
func boundText(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) > n {
		return s[:n]
	}
	return s
}

func PublishRemoteSnapshot(root, remote string, snapshot RemoteSnapshot, apply bool, git GitRunner) SnapshotPublishResult {
	out := SnapshotPublishResult{Remote: remote, Host: snapshot.Host, Ref: SnapshotRef(snapshot.Host), Rows: len(snapshot.Rows), Applied: false}
	if strings.TrimSpace(remote) == "" {
		out.Reason = "remote is required"
		return out
	}
	body, err := json.Marshal(snapshot)
	if err != nil {
		out.Reason = err.Error()
		return out
	}
	out.Bytes = len(body)
	if len(body) > RemoteSnapshotMaxBytes {
		out.Reason = "snapshot exceeds byte bound"
		return out
	}
	old, err := remoteRefOID(root, remote, out.Ref, git)
	if err != nil {
		out.Reason = err.Error()
		return out
	}
	out.PreviousOID = old
	if !apply {
		out.OK = true
		return out
	}
	tmp, err := os.CreateTemp("", "fak-worker-lifecycle-*.json")
	if err != nil {
		out.Reason = "stage snapshot: " + err.Error()
		return out
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.Write(body); err != nil {
		tmp.Close()
		out.Reason = "stage snapshot: " + err.Error()
		return out
	}
	if err = tmp.Close(); err != nil {
		out.Reason = "stage snapshot: " + err.Error()
		return out
	}
	rc, oid := run(git, root, []string{"hash-object", "-w", "--", tmpName})
	oid = strings.TrimSpace(oid)
	if rc != 0 || oid == "" {
		out.Reason = "write snapshot object: " + oid
		return out
	}
	lease := "--force-with-lease=" + out.Ref + ":" + old
	rc, msg := run(git, root, []string{"push", lease, remote, oid + ":" + out.Ref})
	if rc != 0 {
		out.Reason = "compare-and-swap publish refused: " + strings.TrimSpace(msg)
		return out
	}
	out.PublishedOID = oid
	got, err := remoteRefOID(root, remote, out.Ref, git)
	if err != nil {
		out.Reason = "read back snapshot: " + err.Error()
		return out
	}
	out.ReadBackOID = got
	if got != oid {
		out.Reason = "remote read-back mismatch"
		return out
	}
	out.OK = true
	out.Applied = true
	return out
}

func remoteRefOID(root, remote, ref string, git GitRunner) (string, error) {
	rc, out := run(git, root, []string{"ls-remote", "--refs", remote, ref})
	if rc != 0 {
		return "", fmt.Errorf("ls-remote: %s", strings.TrimSpace(out))
	}
	fields := strings.Fields(out)
	if len(fields) == 0 {
		return "", nil
	}
	if len(fields) < 2 || fields[1] != ref {
		return "", errors.New("unexpected ls-remote response")
	}
	return fields[0], nil
}

func FetchRemoteSnapshots(root, remote string, git GitRunner) error {
	if strings.TrimSpace(remote) == "" {
		return errors.New("remote is required")
	}
	spec := "+" + RemoteSnapshotRefRoot + "/*:refs/fak/remotes/" + remote + "/worker-lifecycle/v1/*"
	rc, out := run(git, root, []string{"fetch", "--prune", remote, spec})
	if rc != 0 {
		return fmt.Errorf("fetch remote snapshots: %s", strings.TrimSpace(out))
	}
	return nil
}

func ListRemoteSnapshots(root, remote string, now time.Time, git GitRunner) ([]RemoteSnapshotGroup, error) {
	prefix := "refs/fak/remotes/" + remote + "/worker-lifecycle/v1"
	rc, out := run(git, root, []string{"for-each-ref", "--format=%(refname) %(objectname)", prefix})
	if rc != 0 {
		return nil, fmt.Errorf("list remote snapshots: %s", strings.TrimSpace(out))
	}
	groups := []RemoteSnapshotGroup{}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Fields(line)
		if len(f) != 2 {
			continue
		}
		ref, oid := f[0], f[1]
		g := RemoteSnapshotGroup{Host: strings.TrimPrefix(ref, prefix+"/"), Provenance: "REMOTE_SNAPSHOT", Freshness: SnapshotUnknown, Ref: ref, Rows: []SnapshotRow{}, Authoritative: false}
		rc, raw := run(git, root, []string{"cat-file", "blob", oid})
		if rc != 0 {
			g.Reason = "snapshot object unreadable"
			groups = append(groups, g)
			continue
		}
		if len(raw) > RemoteSnapshotMaxBytes {
			g.Reason = "snapshot exceeds byte bound"
			groups = append(groups, g)
			continue
		}
		var s RemoteSnapshot
		if json.Unmarshal([]byte(raw), &s) != nil {
			g.Reason = "snapshot malformed"
			groups = append(groups, g)
			continue
		}
		g.Schema = s.Schema
		g.ObservedAt = s.ObservedAt
		g.ExpiresAt = s.ExpiresAt
		if s.Schema != RemoteSnapshotSchema {
			g.Reason = "unsupported snapshot schema"
			groups = append(groups, g)
			continue
		}
		if s.Host != g.Host || len(s.Rows) > RemoteSnapshotMaxRows {
			g.Reason = "snapshot identity or row bound invalid"
			groups = append(groups, g)
			continue
		}
		g.Rows = s.Rows
		if now.UTC().After(s.ExpiresAt) {
			g.Freshness = SnapshotStale
			g.Reason = "snapshot expired"
		} else {
			g.Freshness = SnapshotFresh
		}
		groups = append(groups, g)
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i].Host < groups[j].Host })
	return groups, nil
}
