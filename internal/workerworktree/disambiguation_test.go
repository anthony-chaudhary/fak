package workerworktree

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
)

func stubDisambiguationReader(fn func(repo, tree string) DisambiguationWitness) boundedReader {
	return func(_ context.Context, repo, tree string, _ func(string)) DisambiguationWitness {
		return fn(repo, tree)
	}
}

func archiveStreamFromBytes(body []byte, waitErr error) disambiguationArchiveStream {
	return disambiguationArchiveStream{
		Reader: io.NopCloser(bytes.NewReader(body)),
		Wait:   func() error { return waitErr },
	}
}

type gatedArchiveReader struct {
	prefix  *bytes.Reader
	suffix  *bytes.Reader
	reached chan struct{}
	release chan struct{}
	once    sync.Once
}

func (r *gatedArchiveReader) Read(p []byte) (int, error) {
	if r.prefix.Len() > 0 {
		return r.prefix.Read(p)
	}
	r.once.Do(func() { close(r.reached) })
	<-r.release
	return r.suffix.Read(p)
}

func (*gatedArchiveReader) Close() error { return nil }

type contextBlockingArchiveReader struct {
	ctx     context.Context
	started chan struct{}
	once    sync.Once
}

func (r *contextBlockingArchiveReader) Read([]byte) (int, error) {
	r.once.Do(func() { close(r.started) })
	<-r.ctx.Done()
	return 0, r.ctx.Err()
}

func (*contextBlockingArchiveReader) Close() error { return nil }

type manualDeadlineContext struct {
	done chan struct{}
}

func newManualDeadlineContext() *manualDeadlineContext {
	return &manualDeadlineContext{done: make(chan struct{})}
}

func (c *manualDeadlineContext) Deadline() (time.Time, bool) { return time.Unix(0, 0), true }
func (c *manualDeadlineContext) Done() <-chan struct{}       { return c.done }
func (c *manualDeadlineContext) Err() error {
	select {
	case <-c.done:
		return context.DeadlineExceeded
	default:
		return nil
	}
}
func (c *manualDeadlineContext) Value(any) any { return nil }
func (c *manualDeadlineContext) expire()       { close(c.done) }

func writeDisambiguationFixture(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", rel, err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func TestVerifyAppliedDisambiguationRejectsStaleBaseCollision(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()
	calls := 0
	readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
		calls++
		w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, FamilyCoverage: map[string]float64{"loop": 100}}
		if calls == 3 {
			w.SemanticValid = false
			w.CriticalClean = false
			w.Detail = "duplicate canonical row collision"
		}
		return w
	})
	got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
	if ok || got.PostApply.SemanticValid || got.PostApply.Detail == "" {
		t.Fatalf("collision accepted: %+v", got)
	}
}
func TestVerifyAppliedDisambiguationRejectsConcurrentCorpusGap(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()
	calls := 0
	readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
		calls++
		w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, CoverageDebt: 0, FamilyCoverage: map[string]float64{"loop": 100}}
		if calls == 3 {
			w.Coverage = 99
			w.CoverageDebt = 1
			w.FamilyCoverage["loop"] = 90
			w.Detail = "loop: newtoken unpositioned"
		}
		return w
	})
	got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
	if ok || got.PostApply.CoverageDebt != 1 {
		t.Fatalf("gap accepted: %+v", got)
	}
}
func TestVerifyAppliedDisambiguationRecordsThreeWitnesses(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()
	readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
		return DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, FamilyCoverage: map[string]float64{"loop": 100}}
	})
	got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
	if !ok || got.Before.Tree != "HEAD" || got.Worktree.Tree != "HEAD" || got.PostApply.Tree != "candidate" {
		t.Fatalf("witnesses=%+v ok=%v", got, ok)
	}
	if got.Timeout.DefaultTimeoutMS != 120_000 || got.Timeout.RequestedTimeoutMS != nil ||
		got.Timeout.EffectiveTimeoutMS != 120_000 || got.Timeout.RecoveryMode != disambiguationRecoveryDefault {
		t.Fatalf("default timeout receipt = %+v", got.Timeout)
	}
}

func TestVerifyAppliedDisambiguationExplicitTimeoutKeepsSameThreeWitnesses(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()

	type call struct{ repo, tree string }
	var calls []call
	readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
		calls = append(calls, call{repo: repo, tree: tree})
		return DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, FamilyCoverage: map[string]float64{"loop": 100}}
	})
	t.Setenv(DisambiguationTimeoutEnv, "300000")
	got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
	if !ok {
		t.Fatalf("explicit bounded timeout changed oracle verdict: %+v", got)
	}
	wantCalls := []call{{"trunk", "HEAD"}, {"worker", "HEAD"}, {"trunk", "candidate"}}
	if !reflect.DeepEqual(calls, wantCalls) {
		t.Fatalf("witness inputs = %+v, want %+v", calls, wantCalls)
	}
	if got.Timeout.DefaultTimeoutMS != 120_000 || got.Timeout.RequestedTimeoutMS == nil ||
		*got.Timeout.RequestedTimeoutMS != 300_000 || got.Timeout.EffectiveTimeoutMS != 300_000 ||
		got.Timeout.RecoveryMode != disambiguationRecoveryExplicit {
		t.Fatalf("explicit timeout receipt = %+v", got.Timeout)
	}
}

func TestResolveDisambiguationTimeoutBoundsExplicitRecovery(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		present   bool
		want      time.Duration
		wantMode  string
		wantError bool
	}{
		{name: "unset default", want: 2 * time.Minute, wantMode: disambiguationRecoveryDefault},
		{name: "blank present", raw: "  ", present: true, wantMode: disambiguationRecoveryInvalid, wantError: true},
		{name: "minimum", raw: "1", present: true, want: time.Millisecond, wantMode: disambiguationRecoveryExplicit},
		{name: "maximum", raw: "900000", present: true, want: 15 * time.Minute, wantMode: disambiguationRecoveryExplicit},
		{name: "zero", raw: "0", present: true, wantMode: disambiguationRecoveryInvalid, wantError: true},
		{name: "negative", raw: "-1", present: true, wantMode: disambiguationRecoveryInvalid, wantError: true},
		{name: "above maximum", raw: "900001", present: true, wantMode: disambiguationRecoveryInvalid, wantError: true},
		{name: "not an integer", raw: "later", present: true, wantMode: disambiguationRecoveryInvalid, wantError: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt, got, err := resolveDisambiguationTimeout(func(string) (string, bool) { return tt.raw, tt.present })
			if (err != nil) != tt.wantError || got != tt.want || receipt.DefaultTimeoutMS != 120_000 || receipt.RecoveryMode != tt.wantMode {
				t.Fatalf("receipt=%+v timeout=%s err=%v, want timeout=%s mode=%s error=%v", receipt, got, err, tt.want, tt.wantMode, tt.wantError)
			}
			if tt.wantError && receipt.EffectiveTimeoutMS != 0 {
				t.Fatalf("invalid request gained an effective timeout: %+v", receipt)
			}
		})
	}
}

// TestVerifyAppliedDisambiguationFreshnessIsNonRegression pins the #5359 fix: freshness gates
// land as a NON-REGRESSION, not an absolute bar. When HEAD is already concept-stale (a peer
// left a generated artifact un-regenerated), a diff that regresses nothing must still land even
// though its post tree is also stale; only turning a fresh HEAD stale is refused. The stub is
// called in order before/worktree/post, so call 1 sets before.Fresh and call 3 sets post.Fresh
// while every other witness field stays clean and identical (no coverage regression), isolating
// freshness as the only variable.
func TestVerifyAppliedDisambiguationFreshnessIsNonRegression(t *testing.T) {
	cases := []struct {
		name        string
		beforeFresh bool
		postFresh   bool
		wantOK      bool
	}{
		{"stale HEAD, still stale post -> admitted", false, false, true},
		{"stale HEAD, fresh post -> admitted", false, true, true},
		{"fresh HEAD, fresh post -> admitted", true, true, true},
		{"fresh HEAD, stale post -> refused (regression)", true, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			old := readDisambiguation
			defer func() { readDisambiguation = old }()
			calls := 0
			readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
				calls++
				w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, CoverageDebt: 0, FamilyCoverage: map[string]float64{"loop": 100}}
				if calls == 1 {
					w.Fresh = tc.beforeFresh
				}
				if calls == 3 {
					w.Fresh = tc.postFresh
				}
				return w
			})
			_, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
			if ok != tc.wantOK {
				t.Fatalf("freshness non-regression: got ok=%v want %v (before.Fresh=%v post.Fresh=%v)", ok, tc.wantOK, tc.beforeFresh, tc.postFresh)
			}
		})
	}
}

// TestVerifyAppliedDisambiguationClarityIsNonRegression pins #8957: a land may preserve
// or reduce pre-existing clarity debt, but it may neither make a clean HEAD dirty nor add
// debt to an already-dirty HEAD. All other invariant fields stay valid and unchanged so the
// table isolates the clarity decision.
func TestVerifyAppliedDisambiguationClarityIsNonRegression(t *testing.T) {
	cases := []struct {
		name                string
		beforeCriticalClean bool
		beforeClarityDebt   int
		postCriticalClean   bool
		postClarityDebt     int
		wantOK              bool
	}{
		{"clean HEAD stays clean -> admitted", true, 0, true, 0, true},
		{"clean HEAD becomes dirty -> refused", true, 0, false, 1, false},
		{"dirty HEAD gains debt -> refused", false, 2, false, 3, false},
		{"dirty HEAD keeps equal debt -> admitted", false, 2, false, 2, true},
		{"dirty HEAD reduces debt but remains dirty -> admitted", false, 2, false, 1, true},
		{"dirty HEAD becomes clean -> admitted", false, 2, true, 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			old := readDisambiguation
			defer func() { readDisambiguation = old }()
			calls := 0
			readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
				calls++
				w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, CoverageDebt: 0, FamilyCoverage: map[string]float64{"loop": 100}}
				if calls == 1 {
					w.CriticalClean = tc.beforeCriticalClean
					w.ClarityDebt = tc.beforeClarityDebt
				}
				if calls == 3 {
					w.CriticalClean = tc.postCriticalClean
					w.ClarityDebt = tc.postClarityDebt
				}
				return w
			})
			got, ok := verifyAppliedDisambiguation("trunk", "worker", "candidate")
			if ok != tc.wantOK {
				t.Fatalf("clarity non-regression: got ok=%v want %v (before clean=%v debt=%d; post clean=%v debt=%d; witness=%+v)", ok, tc.wantOK, tc.beforeCriticalClean, tc.beforeClarityDebt, tc.postCriticalClean, tc.postClarityDebt, got)
			}
		})
	}
}

func TestCheckDisambiguationInvariantContextPreservesFreshnessAndCoverage(t *testing.T) {
	oldScorecard := runAnalyzer
	defer func() { runAnalyzer = oldScorecard }()

	root := t.TempDir()
	writeDisambiguationFixture(t, root, conceptcatalog.GeneratedReadme, "readme\r\n")
	writeDisambiguationFixture(t, root, conceptcatalog.GeneratedIndex, "index\n")
	writeDisambiguationFixture(t, root, "tools/concept_disambiguation_scorecard.data/_meta.json", `{"families":[]}`)

	runAnalyzer = func(ctx context.Context, gotRoot, generated string) ([]byte, error) {
		if gotRoot != root {
			t.Fatalf("scorecard root = %q, want %q", gotRoot, root)
		}
		writeDisambiguationFixture(t, generated, "README.md", "readme\n")
		writeDisambiguationFixture(t, generated, "INDEX.md", "index\n")
		return []byte(`{
			"ok": true,
			"reason": "",
			"corpus": {
				"coverage_debt": 1,
				"clarity_defects": 0,
				"coverage": {
					"coverage_pct": 87.5,
					"per_family": [
						{"family":"loop","discovered":4,"covered":3}
					]
				}
			}
		}`), nil
	}
	var phases []string
	inv, err := checkInvariantBounded(context.Background(), root, func(phase string) {
		phases = append(phases, phase)
	})
	if err != nil {
		t.Fatalf("context-aware invariant: %v", err)
	}
	if !inv.Freshness.Fresh || !inv.SemanticValid || !inv.CriticalClean ||
		inv.ClarityDebt != 0 || inv.Coverage != 87.5 || inv.CoverageDebt != 1 ||
		inv.FamilyCoverage["loop"] != 75 {
		t.Fatalf("invariant fields regressed: %+v", inv)
	}
	wantPhases := []string{"scorecard-command", "generated-freshness", "catalog-validation", "scorecard-decode"}
	if !reflect.DeepEqual(phases, wantPhases) {
		t.Fatalf("subphases = %v, want %v", phases, wantPhases)
	}
}

func TestLandIsolatedDisambiguationTimeoutIsTypedCancellableAndPreCAS(t *testing.T) {
	oldArchive := runDisambiguationArchive
	oldScorecard := runAnalyzer
	oldContext := newDeadline
	defer func() {
		runDisambiguationArchive = oldArchive
		runAnalyzer = oldScorecard
		newDeadline = oldContext
	}()

	manual := newManualDeadlineContext()
	t.Setenv(DisambiguationTimeoutEnv, "37000")
	var requestedDeadline time.Duration
	newDeadline = func(timeout time.Duration) (context.Context, context.CancelFunc) {
		requestedDeadline = timeout
		return manual, func() {}
	}
	runDisambiguationArchive = func(context.Context, string, string) (disambiguationArchiveStream, error) {
		return archiveStreamFromBytes(make([]byte, 1024), nil), nil // empty but valid tar stream
	}
	scorecardStarted := make(chan struct{})
	scorecardCanceled := make(chan error, 1)
	runAnalyzer = func(ctx context.Context, root, generated string) ([]byte, error) {
		close(scorecardStarted)
		<-ctx.Done()
		scorecardCanceled <- ctx.Err()
		return nil, ctx.Err()
	}

	g := isolatedHappyFake()
	msg := writeMsg(t, "fix(workerland): bound disambiguation (fak workerworktree)")
	type outcome struct {
		result  Result
		handled bool
	}
	outcomeCh := make(chan outcome, 1)
	go func() {
		res, handled := landIsolated(
			"/repo",
			"/worker",
			"diff --git a/tools/concept_disambiguation_scorecard.data/rows-loop-family.json b/tools/concept_disambiguation_scorecard.data/rows-loop-family.json\n@@\n-old\n+new\n",
			msg,
			[]string{"tools/concept_disambiguation_scorecard.data/rows-loop-family.json"},
			g.run,
			g.runEnv,
		)
		outcomeCh <- outcome{result: res, handled: handled}
	}()

	select {
	case <-scorecardStarted:
	case <-time.After(time.Second):
		t.Fatal("scorecard witness command did not start")
	}
	manual.expire()

	var got outcome
	select {
	case got = <-outcomeCh:
	case <-time.After(time.Second):
		t.Fatal("land did not return after the disambiguation deadline")
	}
	select {
	case err := <-scorecardCanceled:
		if err != context.DeadlineExceeded {
			t.Fatalf("scorecard cancellation = %v, want deadline exceeded", err)
		}
	case <-time.After(time.Second):
		t.Fatal("scorecard command seam did not observe cancellation")
	}

	if !got.handled || got.result.OK || got.result.Committed || got.result.Applied {
		t.Fatalf("timeout must be a handled pre-CAS refusal: handled=%v result=%+v", got.handled, got.result)
	}
	if got.result.Disambiguation == nil || got.result.Disambiguation.Before.Diagnostic == nil {
		t.Fatalf("typed timeout diagnostic missing: %+v", got.result.Disambiguation)
	}
	diagnostic := got.result.Disambiguation.Before.Diagnostic
	if diagnostic.Code != DisambiguationTimeoutCode || diagnostic.Witness != "before" || diagnostic.Subphase != "scorecard-command" || diagnostic.TimeoutMS != 37_000 {
		t.Fatalf("timeout diagnostic = %+v", diagnostic)
	}
	if !strings.Contains(got.result.Detail, DisambiguationTimeoutCode) || !strings.Contains(got.result.Detail, `"subphase":"scorecard-command"`) {
		t.Fatalf("compact refusal detail does not carry typed timeout/subphase: %s", got.result.Detail)
	}
	deadline := got.result.Disambiguation.Timeout
	if requestedDeadline != 37*time.Second || deadline.DefaultTimeoutMS != 120_000 ||
		deadline.RequestedTimeoutMS == nil || *deadline.RequestedTimeoutMS != 37_000 ||
		deadline.EffectiveTimeoutMS != 37_000 || deadline.RecoveryMode != disambiguationRecoveryExplicit {
		t.Fatalf("timeout authority was not carried into the receipt: requested=%s receipt=%+v", requestedDeadline, deadline)
	}
	if len(g.envCallsWithPrefix("commit-tree")) != 0 ||
		len(g.callsWithPrefix("update-ref")) != 0 ||
		len(g.callsWithPrefix("checkout")) != 0 ||
		len(g.callsWithPrefix("reset")) != 0 {
		t.Fatalf("timeout crossed the pre-CAS boundary: calls=%v envCalls=%v", g.calls, g.envCalls)
	}
	if g.lastEnv["GIT_INDEX_FILE"] == "" {
		t.Fatalf("pre-timeout index construction must remain isolated: env=%v", g.lastEnv)
	}
}

func TestLandIsolatedInvalidDisambiguationTimeoutRefusesTypedAndPreCAS(t *testing.T) {
	old := readDisambiguation
	defer func() { readDisambiguation = old }()

	for _, raw := range []string{"", "0", "-1", "900001", "malformed-secret-sentinel"} {
		t.Run(raw, func(t *testing.T) {
			t.Setenv(DisambiguationTimeoutEnv, raw)
			reads := 0
			readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
				reads++
				return DisambiguationWitness{Tree: tree}
			})
			g := isolatedHappyFake()
			msg := writeMsg(t, "fix(workerland): reject invalid timeout (fak workerworktree)")
			got, handled := landIsolated(
				"/repo", "/worker",
				"diff --git a/internal/workerworktree/disambiguation.go b/internal/workerworktree/disambiguation.go\n@@\n-old\n+new\n",
				msg, []string{"internal/workerworktree/disambiguation.go"}, g.run, g.runEnv,
			)
			if !handled || got.OK || got.Committed || got.Applied || reads != 0 {
				t.Fatalf("invalid request must refuse before oracle/CAS: handled=%v reads=%d result=%+v", handled, reads, got)
			}
			if got.Disambiguation == nil || got.Disambiguation.Before.Diagnostic == nil {
				t.Fatalf("typed invalid-timeout diagnostic missing: %+v", got.Disambiguation)
			}
			diagnostic := got.Disambiguation.Before.Diagnostic
			if diagnostic.Code != DisambiguationTimeoutCode || diagnostic.Witness != "configuration" || diagnostic.Subphase != "timeout-config" {
				t.Fatalf("invalid-timeout diagnostic = %+v", diagnostic)
			}
			deadline := got.Disambiguation.Timeout
			if deadline.DefaultTimeoutMS != 120_000 || deadline.EffectiveTimeoutMS != 0 ||
				deadline.RecoveryMode != disambiguationRecoveryInvalid {
				t.Fatalf("invalid timeout receipt = %+v", deadline)
			}
			if !strings.Contains(got.Detail, DisambiguationTimeoutCode) || !strings.Contains(got.Detail, DisambiguationTimeoutEnv) {
				t.Fatalf("typed invalid refusal detail = %q", got.Detail)
			}
			if raw == "malformed-secret-sentinel" && strings.Contains(got.Detail, raw) {
				t.Fatalf("malformed environment value leaked into refusal detail: %q", got.Detail)
			}
			if len(g.envCallsWithPrefix("commit-tree")) != 0 || len(g.callsWithPrefix("update-ref")) != 0 ||
				len(g.callsWithPrefix("checkout")) != 0 || len(g.callsWithPrefix("reset")) != 0 {
				t.Fatalf("invalid timeout crossed pre-CAS boundary: calls=%v envCalls=%v", g.calls, g.envCalls)
			}
			if g.lastEnv["GIT_INDEX_FILE"] == "" {
				t.Fatalf("invalid timeout must leave index construction isolated: env=%v", g.lastEnv)
			}
		})
	}
}

func TestDisambiguationArchiveStreamsExtractionBeforeProducerCompletion(t *testing.T) {
	oldArchive := runDisambiguationArchive
	defer func() { runDisambiguationArchive = oldArchive }()

	fixture := t.TempDir()
	writeDisambiguationFixture(t, fixture, "first.txt", "first body\n")
	writeDisambiguationFixture(t, fixture, "second.txt", "second body\n")
	archive := tarDisambiguationFixture(t, fixture)
	secondHeader := bytes.Index(archive, []byte("second.txt"))
	if secondHeader <= 0 {
		t.Fatal("second tar header not found")
	}
	split := secondHeader - secondHeader%512
	reader := &gatedArchiveReader{
		prefix:  bytes.NewReader(archive[:split]),
		suffix:  bytes.NewReader(archive[split:]),
		reached: make(chan struct{}),
		release: make(chan struct{}),
	}
	waited := false
	runDisambiguationArchive = func(context.Context, string, string) (disambiguationArchiveStream, error) {
		return disambiguationArchiveStream{
			Reader: reader,
			Wait: func() error {
				waited = true
				return nil
			},
		}, nil
	}

	dst := t.TempDir()
	type result struct {
		key string
		err error
	}
	done := make(chan result, 1)
	go func() {
		key, err := materializeDisambiguationArchive(context.Background(), "/repo", "HEAD", dst, func(string) {})
		done <- result{key: key, err: err}
	}()

	select {
	case <-reader.reached:
	case <-time.After(time.Second):
		t.Fatal("archive extractor did not reach the gated second header")
	}
	first, err := os.ReadFile(filepath.Join(dst, "first.txt"))
	if err != nil || string(first) != "first body\n" {
		t.Fatalf("first file was not materialized while producer remained open: body=%q err=%v", first, err)
	}
	close(reader.release)

	select {
	case got := <-done:
		if got.err != nil {
			t.Fatal(got.err)
		}
		if got.key != disambiguationCacheKey(archive) {
			t.Fatalf("streamed cache key = %q, want %q", got.key, disambiguationCacheKey(archive))
		}
	case <-time.After(time.Second):
		t.Fatal("archive materialization did not finish after producer release")
	}
	if !waited {
		t.Fatal("archive producer was not joined")
	}
	second, err := os.ReadFile(filepath.Join(dst, "second.txt"))
	if err != nil || string(second) != "second body\n" {
		t.Fatalf("second file differs after streamed extraction: body=%q err=%v", second, err)
	}
}

func TestDisambiguationArchiveCancellationJoinsProducerWithoutPartialWitness(t *testing.T) {
	oldArchive := runDisambiguationArchive
	oldAnalyzer := runAnalyzer
	defer func() {
		runDisambiguationArchive = oldArchive
		runAnalyzer = oldAnalyzer
	}()

	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	joined := make(chan struct{})
	runDisambiguationArchive = func(got context.Context, _, _ string) (disambiguationArchiveStream, error) {
		if got != ctx {
			t.Fatalf("archive context was not propagated")
		}
		return disambiguationArchiveStream{
			Reader: &contextBlockingArchiveReader{ctx: got, started: started},
			Wait: func() error {
				close(joined)
				return got.Err()
			},
		}, nil
	}
	analyzerCalls := 0
	runAnalyzer = func(context.Context, string, string) ([]byte, error) {
		analyzerCalls++
		return nil, nil
	}

	type outcome struct {
		w      DisambiguationWitness
		phases []string
	}
	result := make(chan outcome, 1)
	go func() {
		var phases []string
		w := readDisambiguationWitness(ctx, "/repo", "HEAD", func(phase string) {
			phases = append(phases, phase)
		})
		result <- outcome{w: w, phases: phases}
	}()
	<-started
	cancel()

	var got outcome
	select {
	case got = <-result:
	case <-time.After(time.Second):
		t.Fatal("canceled archive stream did not return")
	}
	select {
	case <-joined:
	default:
		t.Fatal("archive producer was not joined before witness return")
	}
	if analyzerCalls != 0 {
		t.Fatalf("canceled archive published work to analyzer: calls=%d", analyzerCalls)
	}
	if got.w.CacheIdentity != "" || got.w.CacheState != "" || got.w.Fresh || got.w.SemanticValid ||
		got.w.CriticalClean || got.w.Coverage != 0 || got.w.CoverageDebt != 0 || got.w.FamilyCoverage != nil {
		t.Fatalf("canceled archive published a partial witness: %+v", got.w)
	}
	if !strings.Contains(got.w.Detail, context.Canceled.Error()) || got.phases[len(got.phases)-1] != "extract-archive" {
		t.Fatalf("cancellation detail/phases = %q/%v", got.w.Detail, got.phases)
	}
}

func TestDisambiguationWitnessPersistentContentCache(t *testing.T) {
	oldArchive := runDisambiguationArchive
	oldAnalyzer := runAnalyzer
	oldRoot := disambiguationCacheRoot
	oldConfig := disambiguationAnalyzerConfig
	oldVersion := disambiguationAnalyzerVersion
	defer func() {
		runDisambiguationArchive = oldArchive
		runAnalyzer = oldAnalyzer
		disambiguationCacheRoot = oldRoot
		disambiguationAnalyzerConfig = oldConfig
		disambiguationAnalyzerVersion = oldVersion
	}()

	fixture := t.TempDir()
	writeDisambiguationFixture(t, fixture, conceptcatalog.GeneratedReadme, "readme\n")
	writeDisambiguationFixture(t, fixture, conceptcatalog.GeneratedIndex, "index\n")
	writeDisambiguationFixture(t, fixture, "tools/concept_disambiguation_scorecard.data/_meta.json", `{"families":[]}`)
	writeDisambiguationFixture(t, fixture, "docs/extraction-equivalence.txt", "streamed corpus bytes\n")
	archive := tarDisambiguationFixture(t, fixture)
	changedArchive := append([]byte(nil), archive...)
	changedArchive = append(changedArchive, 0)
	cacheRoot := t.TempDir()
	disambiguationCacheRoot = func(string) (string, error) { return cacheRoot, nil }

	var mu sync.Mutex
	analyzerCalls := 0
	runAnalyzer = func(_ context.Context, root, generated string) ([]byte, error) {
		mu.Lock()
		analyzerCalls++
		mu.Unlock()
		got, err := os.ReadFile(filepath.Join(root, "docs", "extraction-equivalence.txt"))
		if err != nil || string(got) != "streamed corpus bytes\n" {
			return nil, errors.New("streamed extraction differs from archive")
		}
		writeDisambiguationFixture(t, generated, "README.md", "readme\n")
		writeDisambiguationFixture(t, generated, "INDEX.md", "index\n")
		return []byte(`{"ok":true,"corpus":{"coverage_debt":1,"clarity_defects":0,"coverage":{"coverage_pct":87.5,"per_family":[{"family":"loop","discovered":4,"covered":3}]}}}`), nil
	}
	currentArchive := archive
	runDisambiguationArchive = func(context.Context, string, string) (disambiguationArchiveStream, error) {
		return archiveStreamFromBytes(append([]byte(nil), currentArchive...), nil), nil
	}
	read := func(repo, tree string) DisambiguationWitness {
		return readDisambiguationWitness(context.Background(), repo, tree, func(string) {})
	}
	semanticJSON := func(w DisambiguationWitness) []byte {
		w.Tree, w.CacheIdentity, w.CacheState, w.CacheReason = "", "", "", ""
		b, err := json.Marshal(w)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}

	oracle := read("/repo-a", "tree-a")
	if oracle.CacheState != "miss" || oracle.CacheIdentity != disambiguationCacheKey(archive) || analyzerCalls != 1 {
		t.Fatalf("oracle state/calls = %q/%d, want miss/1 (%+v)", oracle.CacheState, analyzerCalls, oracle)
	}
	hit := read("/disjoint-exact-path-worker", "same-content")
	if hit.CacheState != "hit" || hit.CacheIdentity != oracle.CacheIdentity || analyzerCalls != 1 {
		t.Fatalf("same tree did not reuse cache: oracle=%+v hit=%+v calls=%d", oracle, hit, analyzerCalls)
	}
	if !bytes.Equal(semanticJSON(hit), semanticJSON(oracle)) {
		t.Fatalf("cached semantic decision differs from oracle:\ncache=%s\noracle=%s", semanticJSON(hit), semanticJSON(oracle))
	}

	currentArchive = changedArchive
	changed := read("/repo-a", "tree-b")
	if changed.CacheState != "miss" || changed.CacheIdentity == oracle.CacheIdentity || analyzerCalls != 2 {
		t.Fatalf("changed relevant content did not miss: %+v calls=%d", changed, analyzerCalls)
	}
	currentArchive = archive
	disambiguationAnalyzerConfig = oldConfig + "+changed"
	config := read("/repo-a", "tree-a")
	if config.CacheState != "miss" || analyzerCalls != 3 {
		t.Fatalf("changed analyzer config did not miss: %+v calls=%d", config, analyzerCalls)
	}
	disambiguationAnalyzerConfig = oldConfig
	disambiguationAnalyzerVersion = oldVersion + "+changed"
	tool := read("/repo-a", "tree-a")
	if tool.CacheState != "miss" || analyzerCalls != 4 {
		t.Fatalf("changed tool version did not miss: %+v calls=%d", tool, analyzerCalls)
	}
	disambiguationAnalyzerVersion = oldVersion

	cachePath := filepath.Join(cacheRoot, oracle.CacheIdentity+".json")
	if err := os.WriteFile(cachePath, []byte("not-json"), 0600); err != nil {
		t.Fatal(err)
	}
	corrupt := read("/repo-a", "tree-a")
	if corrupt.CacheState != "miss" || corrupt.CacheReason != "corrupt" || analyzerCalls != 5 {
		t.Fatalf("corrupt cache was not a miss: %+v calls=%d", corrupt, analyzerCalls)
	}
	if !bytes.Equal(semanticJSON(corrupt), semanticJSON(oracle)) {
		t.Fatalf("corrupt-cache recompute differs from oracle")
	}

	if err := os.Remove(cachePath); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	analyzerCalls = 0
	mu.Unlock()
	const callers = 12
	results := make(chan DisambiguationWitness, callers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			results <- read("/repo-concurrent", "same-tree")
		}()
	}
	close(start)
	wg.Wait()
	close(results)
	mu.Lock()
	calls := analyzerCalls
	mu.Unlock()
	if calls != 1 {
		t.Fatalf("concurrent callers executed analyzer %d times, want 1", calls)
	}
	misses, hits := 0, 0
	for got := range results {
		if !bytes.Equal(semanticJSON(got), semanticJSON(oracle)) {
			t.Fatalf("concurrent cached result differs from oracle: %+v", got)
		}
		switch got.CacheState {
		case "miss":
			misses++
		case "hit":
			hits++
		}
	}
	if misses != 1 || hits != callers-1 {
		t.Fatalf("concurrent cache states miss/hit = %d/%d, want 1/%d", misses, hits, callers-1)
	}
}

func tarDisambiguationFixture(t *testing.T, root string) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		h := &tar.Header{Name: filepath.ToSlash(rel), Mode: 0644, Size: int64(len(body))}
		if err := tw.WriteHeader(h); err != nil {
			return err
		}
		_, err = tw.Write(body)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}
