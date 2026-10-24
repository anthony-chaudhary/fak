package leaseref

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
)

// PublishOptions binds one immutable, locally owned blob to one expected remote
// value. Remote is a configured remote name, never an arbitrary ref or URL.
type PublishOptions struct {
	Remote, Ref, PinnedOID, ExpectedRemoteOID string
	ExpectAbsent                              bool
	Holder, SessionID                         string
	Generation                                int64
	Timeout                                   time.Duration
	// TrustedDirectRefs must cite documented-enforcement:<source> or
	// authorized-server-control:<source> for direct refs throughout refs/fak/locks/.
	// The caller must establish this prerequisite independently. Neither this
	// assertion nor ls-remote can prove it, including on GitHub.
	TrustedDirectRefs string
}

// PublishReceipt is a point-in-time visibility witness, conditional on the stated
// server assumption. It is not tree arbitration or a persistent ownership grant.
// Transport fields describe the single push, excluding read-only preflight.
// Any attempted but unconfirmed push is indeterminate; callers must inspect it,
// not automatically retry. Raw endpoints, assurance text and Git output are omitted.
type PublishReceipt struct {
	Schema               string `json:"schema"`
	Ref                  string `json:"ref,omitempty"`
	PinnedOID            string `json:"pinned_oid,omitempty"`
	ExpectedRemoteOID    string `json:"expected_remote_oid,omitempty"`
	ExpectedAbsent       bool   `json:"expected_absent"`
	ObservedRemoteOID    string `json:"observed_remote_oid,omitempty"`
	LocalReadbackOID     string `json:"local_readback_oid,omitempty"`
	RemoteReadbackOID    string `json:"remote_readback_oid,omitempty"`
	DestinationSHA256    string `json:"destination_sha256,omitempty"`
	AssuranceMethod      string `json:"assurance_method,omitempty"`
	AssuranceSHA256      string `json:"assurance_sha256,omitempty"`
	ServerAssumption     string `json:"server_assumption,omitempty"`
	TransportAttempted   bool   `json:"transport_attempted"`
	TransportCompleted   bool   `json:"transport_completed"`
	PushSucceeded        bool   `json:"push_succeeded"`
	ConfirmedPublication bool   `json:"confirmed_publication"`
	Indeterminate        bool   `json:"indeterminate"`
	Reason               string `json:"reason,omitempty"`
}

const publishServerAssumption = "caller-established trusted server preserves direct refs throughout refs/fak/locks/; not client-verified"
const publishBlobLimit = 64 << 10

// PublishOwnedRef never fetches, retries, writes a local ref, or publishes a
// movable source. Its exact-target guarantee requires publishServerAssumption:
// Git cannot advertise dangling symrefs or prevent a server-side symref race.
func (s *Store) PublishOwnedRef(parent context.Context, o PublishOptions) (r PublishReceipt) {
	r.Schema = "fak.leaseref.publish.v1"
	if o.Timeout == 0 {
		o.Timeout = 30 * time.Second
	}
	if parent == nil || s == nil || s.run == nil || o.Timeout <= 0 || o.Timeout > 2*time.Minute {
		r.Reason = "INVALID_INPUT"
		return
	}
	ctx, cancel := context.WithTimeout(parent, o.Timeout)
	defer cancel()
	fail := func(reason string) PublishReceipt {
		if ended := ctx.Err(); ended != nil {
			reason = context.Canceled.Error()
			if ended == context.DeadlineExceeded {
				reason = context.DeadlineExceeded.Error()
			}
		}
		r.Reason, r.Indeterminate = reason, r.TransportAttempted
		return r
	}
	if !validPublishRef(o.Ref) || !validPublishOID(o.PinnedOID) ||
		o.ExpectAbsent == (o.ExpectedRemoteOID != "") ||
		(!o.ExpectAbsent && (!validPublishOID(o.ExpectedRemoteOID) || len(o.ExpectedRemoteOID) != len(o.PinnedOID))) ||
		!validID(o.Remote) || !publishIdentity(o.Holder) || !publishIdentity(o.SessionID) {
		return fail("INVALID_INPUT")
	}
	r.Ref, r.PinnedOID, r.ExpectedRemoteOID, r.ExpectedAbsent = o.Ref, o.PinnedOID, o.ExpectedRemoteOID, o.ExpectAbsent
	method, source, ok := strings.Cut(o.TrustedDirectRefs, ":")
	if !ok || (method != "documented-enforcement" && method != "authorized-server-control") || !publishText(source, 1024) {
		return fail("DIRECT_REFERENCE_ASSURANCE_REQUIRED")
	}
	r.AssuranceMethod, r.AssuranceSHA256, r.ServerAssumption = method, publishDigest(source), publishServerAssumption
	// The existing runner inherits its environment. Refuse redirects rather than
	// accidentally operate on a caller's hook repository or another namespace.
	for _, name := range []string{"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_INDEX_FILE", "GIT_CONFIG"} {
		if os.Getenv(name) != "" {
			return fail("REPOSITORY_REDIRECTION_REFUSED")
		}
	}
	format, code, err := s.publishGit(ctx, "rev-parse", "--show-object-format")
	if err == nil && code == 129 && ctx.Err() == nil {
		// Git before 2.45 rejects --no-lazy-fetch with this usage status. Omit it
		// only where no promisor configuration or pack can make a read fetch.
		unguarded := context.WithValue(ctx, publishWithoutLazyFetch{}, true)
		if !s.publishCannotLazyFetch(unguarded) {
			return fail("LAZY_FETCH_GUARD_UNAVAILABLE")
		}
		ctx = unguarded
		format, code, err = s.publishGit(ctx, "rev-parse", "--show-object-format")
	}
	if err != nil || code != 0 || (strings.TrimSpace(format) != "sha1" && strings.TrimSpace(format) != "sha256") ||
		(strings.TrimSpace(format) == "sha1") != (len(o.PinnedOID) == 40) {
		return fail("OBJECT_FORMAT_UNCONFIRMED")
	}
	current, ok := s.publishLocalOID(ctx, o.Ref)
	if !ok || current != o.PinnedOID {
		return fail("LOCAL_REF_CHANGED")
	}
	owned, ok := s.publishRecord(ctx, o.Ref, o.PinnedOID)
	if !ok || !owned.live(time.Now()) || owned.holder != o.Holder || owned.session != o.SessionID ||
		owned.generation != o.Generation || (!owned.intent && o.Generation <= 0) {
		return fail("OWNERSHIP_REFUSED")
	}
	if !o.ExpectAbsent {
		old, known := s.publishRecord(ctx, o.Ref, o.ExpectedRemoteOID)
		if !known {
			return fail("PREDECESSOR_UNKNOWN")
		}
		if !owned.succeeds(old, time.Now()) {
			return fail("PREDECESSOR_OWNERSHIP_REFUSED")
		}
	}
	destination, ok := s.publishDestination(ctx, o.Remote)
	if !ok {
		return fail("DESTINATION_REFUSED")
	}
	r.DestinationSHA256 = publishDigest(destination)
	r.ObservedRemoteOID, ok = s.publishRemoteOID(ctx, destination, o.Ref, len(o.PinnedOID))
	if !ok {
		return fail("REMOTE_READBACK_UNCONFIRMED")
	}
	if r.ObservedRemoteOID != o.ExpectedRemoteOID {
		return fail("REMOTE_REF_CHANGED")
	}
	if again, valid := s.publishDestination(ctx, o.Remote); !valid || again != destination {
		return fail("DESTINATION_CHANGED")
	}
	// The last subprocess before push checks ownership, after all potentially
	// slow preflight and configuration reads have finished.
	current, ok = s.publishLocalOID(ctx, o.Ref)
	if !ok || current != o.PinnedOID || !owned.live(time.Now()) {
		return fail("LOCAL_REF_CHANGED")
	}
	if ctx.Err() != nil {
		return fail(context.Canceled.Error())
	}
	r.TransportAttempted = true
	// push.negotiate otherwise spawns fetch --negotiate-only, even for blob tips.
	_, code, err = s.publishGit(ctx, "-c", "push.negotiate=false", "push", "--no-follow-tags", "--recurse-submodules=no",
		"--force-with-lease="+o.Ref+":"+o.ExpectedRemoteOID, "--", destination, o.PinnedOID+":"+o.Ref)
	r.TransportCompleted = err == nil && code >= 0 && ctx.Err() == nil
	if err != nil || code != 0 || ctx.Err() != nil {
		return fail("PUSH_UNCONFIRMED")
	}
	r.PushSucceeded = true
	r.RemoteReadbackOID, ok = s.publishRemoteOID(ctx, destination, o.Ref, len(o.PinnedOID))
	if !ok || r.RemoteReadbackOID != o.PinnedOID {
		return fail("REMOTE_READBACK_UNCONFIRMED")
	}
	r.LocalReadbackOID, ok = s.publishLocalOID(ctx, o.Ref)
	if !ok || r.LocalReadbackOID != o.PinnedOID || !owned.live(time.Now()) || ctx.Err() != nil {
		return fail("LOCAL_READBACK_UNCONFIRMED")
	}
	r.ConfirmedPublication = true
	return r
}

type publishWithoutLazyFetch struct{}

func (s *Store) publishGit(ctx context.Context, args ...string) (string, int, error) {
	if err := ctx.Err(); err != nil {
		return "", -1, err
	}
	// Immutable object identity must survive replace refs; a missing predecessor
	// must not cause an implicit partial-clone fetch.
	globals := []string{"--no-replace-objects", "--no-lazy-fetch"}
	if ctx.Value(publishWithoutLazyFetch{}) != nil {
		globals = globals[:1]
	}
	return s.run(ctx, s.dir, append(globals, args...)...)
}

// publishCannotLazyFetch reports whether the repository has no promisor remote,
// partial-clone extension or promisor pack. Any such key counts, even one set false.
func (s *Store) publishCannotLazyFetch(ctx context.Context) bool {
	_, code, err := s.publishGit(ctx, "config", "--name-only", "--get-regexp", `^(extensions\.partialclone|remote\..*\.(promisor|partialclonefilter))$`)
	if err != nil || code != 1 || ctx.Err() != nil {
		return false
	}
	out, code, err := s.publishGit(ctx, "rev-parse", "--git-path", "objects/pack")
	pack := strings.TrimSuffix(out, "\n")
	if err != nil || code != 0 || pack == "" || ctx.Err() != nil {
		return false
	}
	if !filepath.IsAbs(pack) {
		pack = filepath.Join(s.dir, pack)
	}
	entries, err := os.ReadDir(pack)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return false
	}
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".promisor") {
			return false
		}
	}
	return true
}

func validPublishRef(ref string) bool {
	id := strings.TrimPrefix(ref, refPrefix)
	return strings.HasPrefix(ref, refPrefix) && validID(id) && !strings.Contains(id, "..") &&
		!strings.HasSuffix(id, ".") && !strings.HasSuffix(id, ".lock") && (isLeaseRef(ref) || isIntentRef(ref))
}

func validPublishOID(oid string) bool {
	if len(oid) != 40 && len(oid) != 64 {
		return false
	}
	nonzero := false
	for _, b := range []byte(oid) {
		if !(b >= '0' && b <= '9' || b >= 'a' && b <= 'f') {
			return false
		}
		nonzero = nonzero || b != '0'
	}
	return nonzero
}

func publishText(s string, limit int) bool {
	return s != "" && len(s) <= limit && strings.TrimSpace(s) == s && !strings.ContainsAny(s, "\x00\r\n\t")
}

func publishIdentity(s string) bool { return publishText(s, 256) }
func publishDigest(s string) string { return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(s))) }

func (s *Store) publishLocalOID(ctx context.Context, ref string) (string, bool) {
	out, code, err := s.publishGit(ctx, "for-each-ref", "--format=%(refname) %(objectname) %(symref)", "--", ref)
	fields := strings.Fields(out)
	if err != nil || code != 0 || ctx.Err() != nil || len(fields) != 2 || fields[0] != ref || !validPublishOID(fields[1]) {
		return "", false
	}
	return fields[1], true
}

func (s *Store) publishRemoteOID(ctx context.Context, destination, ref string, oidSize int) (string, bool) {
	out, code, err := s.publishGit(ctx, "ls-remote", "--symref", "--refs", "--", destination, ref)
	if err != nil || code != 0 || ctx.Err() != nil {
		return "", false
	}
	fields := strings.Fields(out)
	if len(fields) == 0 {
		return "", true
	}
	// An advertised symref adds a ref: line; unknown/extra lines also fail closed.
	if len(fields) != 2 || fields[1] != ref || !validPublishOID(fields[0]) || len(fields[0]) != oidSize {
		return "", false
	}
	return fields[0], true
}

func (s *Store) publishDestination(ctx context.Context, remote string) (string, bool) {
	out, code, err := s.publishGit(ctx, "config", "--bool", "--get-all", "remote."+remote+".mirror")
	if err != nil || ctx.Err() != nil || code != 0 && code != 1 {
		return "", false
	}
	for _, value := range strings.Fields(out) {
		if value != "false" {
			return "", false
		}
	}
	// An already-resolved push URL can be rewritten again on a later invocation,
	// differently for ls-remote and push. Refuse that ambiguity without changing config.
	out, code, err = s.publishGit(ctx, "config", "--get-regexp", `^url\..*\.(insteadof|pushinsteadof)$`)
	if err != nil || code != 1 || strings.TrimSpace(out) != "" || ctx.Err() != nil {
		return "", false
	}
	out, code, err = s.publishGit(ctx, "remote", "get-url", "--push", "--all", remote)
	if err != nil || code != 0 || ctx.Err() != nil {
		return "", false
	}
	destination := strings.TrimSuffix(out, "\n")
	if !publishText(destination, 8192) || strings.HasPrefix(destination, "-") || strings.Contains(destination, "::") {
		return "", false
	}
	if strings.Contains(destination, "://") {
		u, err := url.Parse(destination)
		if err != nil || (u.Scheme != "https" && u.Scheme != "http" && u.Scheme != "ssh" && u.Scheme != "git" && u.Scheme != "file") {
			return "", false
		}
		if u.Scheme != "file" && u.Host == "" {
			return "", false
		}
	} else if !strings.Contains(destination, ":") {
		// A URL consisting of a relative filename must never be reinterpreted as
		// another configured remote name by the push command.
		if !filepath.IsAbs(destination) {
			destination = filepath.Join(s.dir, destination)
		}
		var err error
		destination, err = filepath.Abs(destination)
		if err != nil {
			return "", false
		}
	}
	// Git looks up configured remotes before treating even a URL-shaped token
	// as a literal destination. Such an alias can redirect or enable mirror mode.
	out, code, err = s.publishGit(ctx, "remote")
	if err != nil || code != 0 || ctx.Err() != nil {
		return "", false
	}
	for _, name := range strings.Split(strings.TrimSuffix(out, "\n"), "\n") {
		if name == destination {
			return "", false
		}
	}
	return destination, true
}

type publishOwnership struct {
	holder, session                    string
	generation, acquired, renewed, ttl int64
	intent                             bool
	trees                              []string
}

func (r publishOwnership) active() int64 { return max(r.acquired, r.renewed) }
func (r publishOwnership) expired(now time.Time) bool {
	return r.ttl > 0 && now.Unix() >= r.active()+r.ttl
}
func (r publishOwnership) live(now time.Time) bool {
	return r.active() <= now.Unix() && !r.expired(now)
}

func (r publishOwnership) succeeds(old publishOwnership, now time.Time) bool {
	if r.intent != old.intent {
		return false
	}
	if r.holder == old.holder && r.session == old.session && r.generation == old.generation &&
		r.acquired == old.acquired && r.active() >= old.active() && slices.Equal(r.trees, old.trees) {
		return true
	}
	// Intent records have no generation. Even an expired foreign intent cannot
	// prove the required generation advance, so only continuity above is admitted.
	return !r.intent && old.expired(now) && r.generation > old.generation && r.acquired >= old.active()+old.ttl
}

func (s *Store) publishRecord(ctx context.Context, ref, oid string) (publishOwnership, bool) {
	var r publishOwnership
	kind, code, err := s.publishGit(ctx, "cat-file", "-t", oid)
	if err != nil || code != 0 || strings.TrimSpace(kind) != "blob" {
		return r, false
	}
	size, code, err := s.publishGit(ctx, "cat-file", "-s", oid)
	n, parseErr := strconv.ParseInt(strings.TrimSpace(size), 10, 64)
	if err != nil || code != 0 || parseErr != nil || n <= 0 || n > publishBlobLimit {
		return r, false
	}
	body, code, err := s.publishGit(ctx, "cat-file", "blob", oid)
	if err != nil || code != 0 || int64(len(body)) != n || ctx.Err() != nil {
		return r, false
	}
	if isIntentRef(ref) {
		var rec IntentRecord
		if !publishDecode(body, &rec) || strings.TrimSpace(rec.Target) == "" || rec.Key != IntentKey(rec.Target) || rec.Ref() != ref || rec.TTLSeconds <= 0 {
			return r, false
		}
		r = publishOwnership{holder: rec.Holder, session: rec.SessionID, acquired: rec.AcquiredAt, renewed: rec.RenewedAt, ttl: rec.TTLSeconds, intent: true}
	} else {
		var rec Record
		if !publishDecode(body, &rec) || rec.Ref() != ref || !validID(rec.ID) || rec.Generation < 0 {
			return r, false
		}
		r = publishOwnership{holder: rec.Holder, session: rec.SessionID, generation: rec.Generation, acquired: rec.AcquiredAt, renewed: rec.RenewedAt, ttl: rec.TTLSeconds, trees: rec.TreeGlobs}
	}
	if !publishIdentity(r.holder) || !publishIdentity(r.session) || r.acquired <= 0 || r.renewed < 0 ||
		(r.renewed != 0 && r.renewed < r.acquired) || r.ttl < 0 || r.ttl > math.MaxInt64-r.active() {
		return publishOwnership{}, false
	}
	return r, true
}

func publishDecode(body string, dst any) bool {
	dec := json.NewDecoder(strings.NewReader(body))
	start, err := dec.Token()
	if err != nil || start != json.Delim('{') {
		return false
	}
	seen := map[string]bool{}
	for dec.More() {
		key, err := dec.Token()
		name, ok := key.(string)
		if err != nil || !ok || seen[name] {
			return false
		}
		// encoding/json accepts case-folded aliases, including Unicode long-s.
		// Native records use lowercase ASCII keys; require that canonical spelling.
		for _, c := range name {
			if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_') {
				return false
			}
		}
		seen[name] = true
		var value json.RawMessage
		if dec.Decode(&value) != nil {
			return false
		}
	}
	if _, err := dec.Token(); err != nil {
		return false
	}
	if _, err := dec.Token(); err != io.EOF {
		return false
	}
	dec = json.NewDecoder(strings.NewReader(body))
	dec.DisallowUnknownFields()
	return dec.Decode(dst) == nil
}
