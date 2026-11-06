package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/refid"
	"github.com/anthony-chaudhary/fak/internal/sessionref"
)

// Readiness is a read-only diagnostic, never lease admission. In particular a
// stable Git ref observation does not establish a fresh authority snapshot.
// #12100's READY branch remains dependent on the authority contract in #11835.
type leaserefReadinessReport struct {
	Schema   string                     `json:"schema"`
	OK       bool                       `json:"ok"`
	Ready    bool                       `json:"ready"`
	Verdict  string                     `json:"verdict"`
	Snapshot leaserefReadinessSnapshot  `json:"snapshot"`
	Counts   map[string]int             `json:"counts"`
	Blockers []leaserefReadinessBlocker `json:"blockers"`
}

type leaserefReadinessSnapshot struct {
	Source        string `json:"source"`
	Freshness     string `json:"freshness"`
	Authoritative bool   `json:"authoritative"`
	Complete      bool   `json:"complete"`
	Stable        bool   `json:"stable"`
	ObservedUnix  int64  `json:"observed_unix"`
	RefDigest     string `json:"ref_digest,omitempty"`
}

type leaserefReadinessBlocker struct {
	Code       string `json:"code"`
	ID         string `json:"id,omitempty"`
	Ref        string `json:"ref,omitempty"`
	Evidence   string `json:"evidence"`
	NextAction string `json:"next_action"`
}

func runLeaserefReadiness(stdout, stderr io.Writer, dir string) int {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	report := collectLeaserefReadiness(ctx, dir, time.Now().UTC())
	if code := emitLeaserefJSON(stdout, stderr, report, "audit --require-ready"); code != 0 {
		return code
	}
	if !report.Ready {
		return 1
	}
	return 0
}

func collectLeaserefReadiness(ctx context.Context, dir string, now time.Time) leaserefReadinessReport {
	return collectLeaserefReadinessWithReader(ctx, dir, now, leaserefReadinessGit)
}

func collectLeaserefReadinessWithReader(ctx context.Context, dir string, now time.Time, read func(context.Context, string, string, ...string) (string, error)) leaserefReadinessReport {
	r := leaserefReadinessReport{
		Schema: "fak.leaseref-readiness.v1", Verdict: "BLOCKED",
		Snapshot: leaserefReadinessSnapshot{Source: "local_git_refs_oid_pinned", Freshness: "unverified", ObservedUnix: now.Unix()},
		Counts:   map[string]int{"leases": 0, "live_leases": 0, "expired_leases": 0, "age_stale_leases": 0, "sessions": 0, "expired_descriptors": 0, "incompatible_records": 0},
		Blockers: []leaserefReadinessBlocker{},
	}
	add := func(code, id, ref, evidence, action string) {
		r.Blockers = append(r.Blockers, leaserefReadinessBlocker{code, id, ref, evidence, action})
	}
	add("AUTHORITATIVE_SNAPSHOT_UNAVAILABLE", "", "",
		"Local Git refs provide visibility, not verified authoritative freshness or write fencing.",
		"Keep LAN readiness blocked; obtain the verified authority snapshot required by #11835. Fetching refs or setting a coordinator URL is not sufficient.")
	fail := func(err error) leaserefReadinessReport {
		r.Snapshot.Complete = false
		add("SNAPSHOT_UNAVAILABLE", "", "", err.Error(), "Restore readable local Git objects, then rerun fak leaseref audit --require-ready.")
		return r
	}
	refs, err := read(ctx, dir, "", "for-each-ref", "--format=%(refname) %(objectname)", sessionref.RefPrefix)
	if err != nil {
		return fail(err)
	}
	type pinnedRef struct{ name, oid string }
	var selected []pinnedRef
	var input strings.Builder
	for _, line := range strings.Split(strings.TrimSpace(refs), "\n") {
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			return fail(fmt.Errorf("invalid ref enumeration entry"))
		}
		if !leaseref.IsLeaseRef(fields[0]) && !leaseref.IsSessionRef(fields[0]) {
			continue
		}
		selected = append(selected, pinnedRef{fields[0], fields[1]})
		input.WriteString(fields[1] + "\n")
	}
	r.Snapshot.RefDigest = fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(refs)))
	stream := ""
	if len(selected) > 0 {
		stream, err = read(ctx, dir, input.String(), "cat-file", "--batch")
		if err != nil {
			return fail(err)
		}
	}
	reader := bufio.NewReader(strings.NewReader(stream))
	sessions := map[string]leaseref.SessionDescriptor{}
	var leases []leaseref.Record
	r.Snapshot.Complete = true
	for _, ref := range selected {
		header, err := reader.ReadString('\n')
		if err != nil {
			r.Snapshot.Complete = false
			return fail(fmt.Errorf("incomplete object stream for %s", ref.name))
		}
		fields := strings.Fields(header)
		if len(fields) != 3 || fields[0] != ref.oid {
			r.Snapshot.Complete = false
			return fail(fmt.Errorf("missing or mismatched object for %s", ref.name))
		}
		size, err := strconv.ParseInt(fields[2], 10, 64)
		// The output is already bounded by the actual stream length; never allocate
		// from an untrusted header or silently accept a truncated batch.
		if err != nil || size < 0 || size >= int64(len(stream)) {
			r.Snapshot.Complete = false
			return fail(fmt.Errorf("invalid object length for %s", ref.name))
		}
		data := make([]byte, int(size)+1)
		if _, err = io.ReadFull(reader, data); err != nil || data[len(data)-1] != '\n' {
			r.Snapshot.Complete = false
			return fail(fmt.Errorf("truncated object for %s", ref.name))
		}
		data = data[:len(data)-1]
		id := strings.TrimPrefix(ref.name, sessionref.RefPrefix)
		code := "UNREADABLE_LEASE_RECORD"
		valid := false
		if leaseref.IsSessionRef(ref.name) {
			id = strings.TrimPrefix(id, sessionref.SessionPrefix)
			code = "UNREADABLE_SESSION_RECORD"
			var d leaseref.SessionDescriptor
			if fields[1] == "blob" && json.Unmarshal(data, &d) == nil && sessionref.ValidSessionID(d.ID) && d.ID == id {
				valid = true
				r.Counts["sessions"]++
				if d.TTLSecs <= 0 || d.UpdatedAt <= 0 || d.UpdatedAt > now.Unix() || d.TTLSecs > (1<<63-1)-d.UpdatedAt {
					add("UNKNOWN_SESSION_EVIDENCE", id, ref.name, "Session heartbeat is unbounded, undated, future-dated or invalid.", "Repair the owning session's bounded heartbeat publisher; do not infer owner death.")
				} else {
					sessions[id] = d
					if d.Expired(now) {
						r.Counts["expired_descriptors"]++
						add("ACTION_DRAIN", id, ref.name, "Expired session descriptor remains in this observation.", "Verify current authority and use the sanctioned descriptor drain in #9885; local deletion alone does not drain origin.")
					}
				}
			}
		} else {
			var lease leaseref.Record
			if fields[1] == "blob" && json.Unmarshal(data, &lease) == nil && refid.Valid(lease.ID) && lease.ID == id {
				valid = true
				leases = append(leases, lease)
			} else if strings.HasPrefix(id, "epoch-") {
				if _, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64); err == nil {
					code = "INCOMPATIBLE_EPOCH_RECORD"
				}
			}
		}
		if !valid {
			r.Snapshot.Complete = false
			r.Counts["incompatible_records"]++
			add(code, id, ref.name, "Observed object cannot be read as the record type required by its namespace.", "Preserve this ref and reconcile the publisher/reader schema. Do not delete epoch high-water evidence or exclude valid epoch-* leases.")
		}
	}
	if _, err := reader.ReadByte(); err != io.EOF {
		r.Snapshot.Complete = false
		return fail(fmt.Errorf("unexpected trailing object stream"))
	}
	// This detects changes during observation, but is not an atomicity or remote
	// freshness claim. Every decoded blob was pinned to the first observed OID.
	after, err := read(ctx, dir, "", "for-each-ref", "--format=%(refname) %(objectname)", sessionref.RefPrefix)
	if err != nil {
		return fail(err)
	}
	r.Snapshot.Stable = after == refs
	if !r.Snapshot.Stable {
		add("SNAPSHOT_CHANGED", "", "", "Local refs changed during the pinned observation.", "Rerun fak leaseref audit --require-ready; do not use the changing view for admission.")
	}
	r.OK = true
	for _, lease := range leases {
		r.Counts["leases"]++
		active := max(lease.AcquiredAt, lease.RenewedAt)
		if active <= 0 || active > now.Unix() || lease.TTLSeconds < 0 || lease.TTLSeconds > (1<<63-1)-active {
			add("UNKNOWN_LEASE_EVIDENCE", lease.ID, lease.Ref(), "Lease time or TTL evidence is invalid or unavailable.", "Have the holder repair or renew its fenced lease; do not infer expiry from invalid timestamps.")
			continue
		}
		expired := lease.Expired(now)
		if expired {
			r.Counts["expired_leases"]++
		} else {
			r.Counts["live_leases"]++
		}
		stale := lease.TTLSeconds == 0 && now.Unix()-active >= leaserefNoTTLStaleAgeS
		if stale {
			r.Counts["age_stale_leases"]++
		}
		class, _, evidence := leaseref.ClassifyLiveness(lease, sessions, "", now)
		switch {
		case class == leaseref.LivenessPeerUnknown:
			add("UNKNOWN_LIVENESS", lease.ID, lease.Ref(), evidence, "Restore the owning session binding and bounded descriptor publisher; absence is not proof of death.")
		case (expired || stale) && class == leaseref.LivenessPeerLive:
			add("BLOCKED_CONTRADICTION", lease.ID, lease.Ref(), "Lease is expired or NO_TTL_AGE_EXCEEDED while its owner is peer-live.", "Preserve the heartbeating holder. Have it perform a generation-fenced renewal or release/reacquire with finite TTL; never auto-reap.")
		case expired && class == leaseref.LivenessPeerDead:
			add("ACTION_REAP", lease.ID, lease.Ref(), evidence, "Recheck the fresh authority and generation, then use the sanctioned fenced reap path. Do not run a blanket reap while any live or unknown holder conflicts.")
		case lease.TTLSeconds == 0:
			add("UNBOUNDED_LEASE", lease.ID, lease.Ref(), "Lease has no finite TTL.", "Have the holder renew with finite TTL or perform its generation-fenced release/reacquire; preserve the current holder until then.")
		case class == leaseref.LivenessPeerDead:
			add("OWNER_DEAD", lease.ID, lease.Ref(), evidence, "Recheck authoritative liveness and wait for the finite lease to expire before sanctioned fenced recovery.")
		}
	}
	sort.Slice(r.Blockers, func(i, j int) bool {
		a, b := r.Blockers[i], r.Blockers[j]
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		return a.Ref < b.Ref
	})
	return r
}

func leaserefReadinessGit(ctx context.Context, dir, stdin string, args ...string) (string, error) {
	args = append([]string{"--no-replace-objects"}, args...)
	cmd := exec.CommandContext(ctx, "git", args...)
	configureDispatchHelperCommand(cmd)
	cmd.WaitDelay = time.Second
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(stdin)
	out := leaserefReadinessOutput{limit: 32 << 20}
	errOut := leaserefReadinessOutput{limit: 16 << 10}
	cmd.Stdout, cmd.Stderr = &out, &errOut
	err := cmd.Run()
	if out.exceeded || errOut.exceeded {
		return "", fmt.Errorf("Git readiness output exceeded the bounded snapshot budget")
	}
	if err != nil {
		return "", fmt.Errorf("git readiness read: %w: %s", err, strings.TrimSpace(errOut.String()))
	}
	return out.String(), nil
}

// Bound capture before parsing: corrupt or unexpectedly large blobs must not
// allocate unbounded memory. Excess output is discarded until the command exits
// or the caller's deadline kills it, then the entire observation is refused.
type leaserefReadinessOutput struct {
	buffer   bytes.Buffer
	limit    int
	exceeded bool
}

func (b *leaserefReadinessOutput) Len() int       { return b.buffer.Len() }
func (b *leaserefReadinessOutput) String() string { return b.buffer.String() }

func (b *leaserefReadinessOutput) Write(p []byte) (int, error) {
	n := len(p)
	if n > b.limit-b.Len() {
		b.exceeded = true
		p = p[:b.limit-b.Len()]
	}
	_, _ = b.buffer.Write(p)
	return n, nil
}
