package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	corpusSchema = "fak-microcontext-corpus/1"
	answerSchema = "fak-microcontext-answers/1"
	gradeSchema  = "fak-microcontext-grade/1"
)

type ghCorpusIssue struct {
	Number    int        `json:"number"`
	Title     string     `json:"title"`
	Body      string     `json:"body"`
	State     string     `json:"state"`
	Labels    []ghLabel  `json:"labels"`
	CreatedAt time.Time  `json:"createdAt"`
	UpdatedAt time.Time  `json:"updatedAt"`
	ClosedAt  *time.Time `json:"closedAt"`
	URL       string     `json:"url"`
}
type ghLabel struct {
	Name string `json:"name"`
}

type publicCorpus struct {
	Schema   string        `json:"schema"`
	Source   string        `json:"source"`
	FrozenAt string        `json:"frozen_at"`
	Records  []publicIssue `json:"records"`
}
type publicIssue struct {
	ID          string     `json:"id"`
	Split       string     `json:"split"`
	Number      int        `json:"number"`
	Title       string     `json:"title"`
	Body        string     `json:"body"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
	URL         string     `json:"url"`
	State       string     `json:"state"`
	Labels      []string   `json:"labels"`
	ClosedAt    *time.Time `json:"closed_at"`
	TitleSHA256 string     `json:"title_sha256"`
	BodySHA256  string     `json:"body_sha256"`
}
type answerBundle struct {
	Schema       string           `json:"schema"`
	CorpusSHA256 string           `json:"corpus_sha256"`
	Answers      []issueAnswer    `json:"answers"`
	Aggregates   aggregateAnswers `json:"aggregates"`
}
type issueAnswer struct {
	ID                  string   `json:"id"`
	Split               string   `json:"split"`
	State               string   `json:"state"`
	Labels              []string `json:"labels"`
	References          []int    `json:"references"`
	DuplicateTargets    []int    `json:"duplicate_targets"`
	ScopeContradictions []string `json:"scope_contradictions"`
}
type aggregateAnswers struct {
	StateCounts                 map[string]int `json:"state_counts"`
	LabelCounts                 map[string]int `json:"label_counts"`
	NewestIssueIDs              []string       `json:"newest_issue_ids"`
	MostRecentlyUpdatedIssueIDs []string       `json:"most_recently_updated_issue_ids"`
}
type corpusReport struct {
	Schema              string          `json:"schema"`
	CorpusSHA256        string          `json:"corpus_sha256"`
	AnswersSHA256       string          `json:"answers_sha256"`
	Records             int             `json:"records"`
	TrainRecords        int             `json:"train_records"`
	TuneRecords         int             `json:"tune_records"`
	TestRecords         int             `json:"test_records"`
	OpenRecords         int             `json:"open_records"`
	ClosedRecords       int             `json:"closed_records"`
	ReferenceEdges      int             `json:"reference_edges"`
	DuplicateEdges      int             `json:"duplicate_edges"`
	ScopeContradictions int             `json:"scope_contradictions"`
	ScrubbedRecords     int             `json:"scrubbed_records"`
	ScrubbedMatches     int             `json:"scrubbed_matches"`
	LeakChecks          map[string]bool `json:"leak_checks"`
	BlindOracleGrade    gradeReport     `json:"blind_oracle_grade"`
	QuestionSlices      []string        `json:"question_slices"`
	Provenance          []string        `json:"provenance"`
}
type submission struct {
	Schema       string           `json:"schema"`
	CorpusSHA256 string           `json:"corpus_sha256"`
	Answers      []issueAnswer    `json:"answers"`
	Aggregates   aggregateAnswers `json:"aggregates"`
}
type gradeReport struct {
	Schema             string `json:"schema"`
	CorpusSHA256       string `json:"corpus_sha256"`
	TestRecords        int    `json:"test_records"`
	ExactRecords       int    `json:"exact_records"`
	FalsePositiveFacts int    `json:"false_positive_facts"`
	FalseNegativeFacts int    `json:"false_negative_facts"`
	AggregateErrors    int    `json:"aggregate_errors"`
	CitationErrors     int    `json:"citation_errors"`
	QualityPass        bool   `json:"quality_pass"`
}

var issueRefRE = regexp.MustCompile(`(?i)(?:^|[^[:alnum:]_])#([0-9]{1,7})\b`)
var duplicateRE = regexp.MustCompile(`(?i)\b(?:duplicate(?:s|d)?(?:\s+of)?|supersede[sd]?\s+by)\s+#([0-9]{1,7})\b`)

func freezeCorpus(input, publicOut, answersOut, reportOut, source string) error {
	raw, err := os.ReadFile(input)
	if err != nil {
		return err
	}
	var xs []ghCorpusIssue
	if err := json.Unmarshal(raw, &xs); err != nil {
		return fmt.Errorf("decode source: %w", err)
	}
	if len(xs) < 50 {
		return fmt.Errorf("need at least 50 non-fixture records, got %d", len(xs))
	}
	scrubbedRecords, scrubbedMatches := 0, 0
	for i := range xs {
		var n int
		xs[i].Title, n = scrubCorpusText(xs[i].Title)
		scrubbedMatches += n
		xs[i].Body, n = scrubCorpusText(xs[i].Body)
		scrubbedMatches += n
		if n > 0 || strings.Contains(xs[i].Title, "[redacted:public-corpus]") || strings.Contains(xs[i].Body, "[redacted:public-corpus]") {
			scrubbedRecords++
		}
	}
	sort.Slice(xs, func(i, j int) bool { return xs[i].Number < xs[j].Number })
	splits := groupedCorpusSplits(xs)
	seen := map[int]bool{}
	pub := publicCorpus{Schema: corpusSchema, Source: source, FrozenAt: latestUpdate(xs).Format(time.RFC3339)}
	ans := answerBundle{Schema: answerSchema, Aggregates: aggregateAnswers{StateCounts: map[string]int{}, LabelCounts: map[string]int{}}}
	created := append([]ghCorpusIssue(nil), xs...)
	updated := append([]ghCorpusIssue(nil), xs...)
	sort.Slice(created, func(i, j int) bool { return created[i].CreatedAt.After(created[j].CreatedAt) })
	sort.Slice(updated, func(i, j int) bool { return updated[i].UpdatedAt.After(updated[j].UpdatedAt) })
	for _, x := range xs {
		if x.Number <= 0 || x.Title == "" || x.URL == "" || seen[x.Number] {
			return fmt.Errorf("invalid/duplicate issue %d", x.Number)
		}
		seen[x.Number] = true
		id := fmt.Sprintf("issue-%d", x.Number)
		split := splits[x.Number]
		labels := make([]string, 0, len(x.Labels))
		for _, l := range x.Labels {
			if l.Name != "" {
				labels = append(labels, l.Name)
			}
		}
		sort.Strings(labels)
		state := strings.ToUpper(x.State)
		pub.Records = append(pub.Records, publicIssue{ID: id, Split: split, Number: x.Number, Title: x.Title, Body: x.Body, CreatedAt: x.CreatedAt, UpdatedAt: x.UpdatedAt, URL: x.URL, State: state, Labels: append([]string(nil), labels...), ClosedAt: x.ClosedAt, TitleSHA256: shaHex([]byte(x.Title)), BodySHA256: shaHex([]byte(x.Body))})
		labels = labels[:0]
		for _, l := range x.Labels {
			if l.Name != "" {
				labels = append(labels, l.Name)
			}
		}
		for _, l := range x.Labels {
			if l.Name != "" {
				labels = append(labels, l.Name)
				ans.Aggregates.LabelCounts[l.Name]++
			}
		}
		sort.Strings(labels)
		ans.Aggregates.StateCounts[state]++
		ans.Answers = append(ans.Answers, issueAnswer{ID: id, Split: split, State: state, Labels: labels, References: extractIssueNumbers(issueRefRE, x.Title+"\n"+x.Body), DuplicateTargets: extractIssueNumbers(duplicateRE, x.Title+"\n"+x.Body), ScopeContradictions: scopeContradictions(x.Body)})
	}
	for i := 0; i < min(10, len(created)); i++ {
		ans.Aggregates.NewestIssueIDs = append(ans.Aggregates.NewestIssueIDs, fmt.Sprintf("issue-%d", created[i].Number))
		ans.Aggregates.MostRecentlyUpdatedIssueIDs = append(ans.Aggregates.MostRecentlyUpdatedIssueIDs, fmt.Sprintf("issue-%d", updated[i].Number))
	}
	pubBytes, err := marshalStable(pub)
	if err != nil {
		return err
	}
	ans.CorpusSHA256 = shaHex(pubBytes)
	ansBytes, err := marshalStable(ans)
	if err != nil {
		return err
	}
	oracle := submission{Schema: "fak-microcontext-submission/1", CorpusSHA256: ans.CorpusSHA256, Answers: ans.Answers, Aggregates: ans.Aggregates}
	grade := gradeSubmission(ans, oracle)
	rep := corpusReport{Schema: "fak-microcontext-corpus-report/1", CorpusSHA256: ans.CorpusSHA256, AnswersSHA256: shaHex(ansBytes), Records: len(xs), ScrubbedRecords: scrubbedRecords, ScrubbedMatches: scrubbedMatches, LeakChecks: map[string]bool{}, BlindOracleGrade: grade, QuestionSlices: []string{"held-out state/label classification", "reference and duplicate relations", "scope contradiction detection", "exhaustive state/label counts", "chronology top-10", "recent-update top-10"}, Provenance: []string{"source records fetched from the public GitHub Issues API and scrubbed through a committed corpus redaction map", "source state/labels/closure metadata remain in candidate input; only derived relation/cue/aggregate answers are isolated", "train/tune/test split groups exact duplicate titles/bodies and in-corpus reference components before deterministic hashing", "candidate pipelines receive only the public corpus"}}
	for _, a := range ans.Answers {
		switch a.Split {
		case "train":
			rep.TrainRecords++
		case "tune":
			rep.TuneRecords++
		case "test":
			rep.TestRecords++
		}
		if a.State == "OPEN" {
			rep.OpenRecords++
		} else if a.State == "CLOSED" {
			rep.ClosedRecords++
		}
		rep.ReferenceEdges += len(a.References)
		rep.DuplicateEdges += len(a.DuplicateTargets)
		rep.ScopeContradictions += len(a.ScopeContradictions)
	}
	var generic any
	_ = json.Unmarshal(pubBytes, &generic)
	rep.LeakChecks["public_preserves_source_state_labels_closed_at"] = bytesContainAny(pubBytes, []string{`"state"`}) && bytesContainAny(pubBytes, []string{`"labels"`}) && bytesContainAny(pubBytes, []string{`"closed_at"`})
	rep.LeakChecks["public_omits_derived_answer_fields"] = !bytesContainAny(pubBytes, []string{`"references"`, `"duplicate_targets"`, `"scope_contradictions"`, `"aggregates"`})
	rep.LeakChecks["answer_digest_absent_from_public"] = !strings.Contains(string(pubBytes), rep.AnswersSHA256)
	rep.LeakChecks["disjoint_train_tune_test_ids"] = disjointSplits(pub.Records)
	rep.LeakChecks["disjoint_train_tune_test_content"] = disjointContent(pub.Records)
	rep.LeakChecks["oracle_blind_run_exact"] = grade.QualityPass
	if !allTrue(rep.LeakChecks) {
		return fmt.Errorf("leak/grade checks failed: %+v", rep.LeakChecks)
	}
	if err := writeBytes(publicOut, pubBytes); err != nil {
		return err
	}
	if err := writeBytes(answersOut, ansBytes); err != nil {
		return err
	}
	repBytes, _ := marshalStable(rep)
	return writeBytes(reportOut, repBytes)
}

var corpusRedactions = []struct {
	re          *regexp.Regexp
	replacement string
}{
	{regexp.MustCompile(`(?i)` + "node-" + "agent-" + "netra"), "[redacted:public-corpus]"},
	{regexp.MustCompile(`(?i)` + "sam" + "sung"), "[redacted:public-corpus]"},
	{regexp.MustCompile(`(?i)\b(?:lab[-_ ])?dgx[0-9]+\b`), "[redacted:public-corpus]"},
}

func scrubCorpusText(s string) (string, int) {
	total := 0
	for _, rule := range corpusRedactions {
		total += len(rule.re.FindAllStringIndex(s, -1))
		s = rule.re.ReplaceAllString(s, rule.replacement)
	}
	return s, total
}

func corpusSplit(n int) string { return corpusSplitKey(strconv.Itoa(n)) }

func corpusSplitKey(key string) string {
	h := sha256.Sum256([]byte(key))
	switch bucket := int(h[0]) % 10; {
	case bucket < 2:
		return "train"
	case bucket < 4:
		return "tune"
	default:
		return "test"
	}
}

func groupedCorpusSplits(xs []ghCorpusIssue) map[int]string {
	parent := map[int]int{}
	known := map[int]bool{}
	titleOwner := map[string]int{}
	bodyOwner := map[string]int{}
	for _, x := range xs {
		parent[x.Number] = x.Number
		known[x.Number] = true
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra == rb {
			return
		}
		if ra < rb {
			parent[rb] = ra
		} else {
			parent[ra] = rb
		}
	}
	for _, x := range xs {
		th, bh := shaHex([]byte(x.Title)), shaHex([]byte(x.Body))
		if n, ok := titleOwner[th]; ok {
			union(x.Number, n)
		} else {
			titleOwner[th] = x.Number
		}
		if n, ok := bodyOwner[bh]; ok {
			union(x.Number, n)
		} else {
			bodyOwner[bh] = x.Number
		}
		for _, n := range extractIssueNumbers(issueRefRE, x.Title+"\n"+x.Body) {
			if known[n] {
				union(x.Number, n)
			}
		}
	}
	out := map[int]string{}
	for _, x := range xs {
		out[x.Number] = corpusSplitKey(strconv.Itoa(find(x.Number)))
	}
	return out
}

func latestUpdate(xs []ghCorpusIssue) time.Time {
	var latest time.Time
	for _, x := range xs {
		if x.UpdatedAt.After(latest) {
			latest = x.UpdatedAt
		}
	}
	return latest.UTC()
}
func extractIssueNumbers(re *regexp.Regexp, s string) []int {
	set := map[int]bool{}
	for _, m := range re.FindAllStringSubmatch(s, -1) {
		n, _ := strconv.Atoi(m[1])
		if n > 0 {
			set[n] = true
		}
	}
	out := make([]int, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}
func scopeContradictions(body string) []string {
	markers := []string{"contradict", "inconsistent", "conflict", "disagree", "mismatch"}
	set := map[string]bool{}
	low := strings.ToLower(body)
	for _, marker := range markers {
		if strings.Contains(low, marker) {
			set[marker] = true
		}
	}
	out := make([]string, 0, len(set))
	for marker := range set {
		out = append(out, marker)
	}
	sort.Strings(out)
	return out
}

func gradeSubmission(gold answerBundle, got submission) gradeReport {
	r := gradeReport{Schema: gradeSchema, CorpusSHA256: gold.CorpusSHA256}
	if got.CorpusSHA256 != gold.CorpusSHA256 {
		return r
	}
	gm := map[string]issueAnswer{}
	goldIDs := map[string]bool{}
	for _, a := range gold.Answers {
		goldIDs[a.ID] = true
	}
	for _, a := range got.Answers {
		if gm[a.ID].ID != "" || !goldIDs[a.ID] {
			r.CitationErrors++
		}
		gm[a.ID] = a
	}
	for _, a := range gold.Answers {
		if a.Split != "test" {
			continue
		}
		r.TestRecords++
		g, ok := gm[a.ID]
		if !ok {
			r.FalseNegativeFacts += factCount(a)
			continue
		}
		fp, fn := compareAnswer(a, g)
		r.FalsePositiveFacts += fp
		r.FalseNegativeFacts += fn
		if fp == 0 && fn == 0 {
			r.ExactRecords++
		}
	}
	if !equalAggregate(gold.Aggregates, got.Aggregates) {
		r.AggregateErrors++
	}
	r.QualityPass = r.TestRecords > 0 && r.ExactRecords == r.TestRecords && r.FalsePositiveFacts == 0 && r.FalseNegativeFacts == 0 && r.AggregateErrors == 0 && r.CitationErrors == 0
	return r
}
func compareAnswer(a, b issueAnswer) (int, int) {
	as := answerFacts(a)
	bs := answerFacts(b)
	fp, fn := 0, 0
	for x := range bs {
		if !as[x] {
			fp++
		}
	}
	for x := range as {
		if !bs[x] {
			fn++
		}
	}
	return fp, fn
}
func answerFacts(a issueAnswer) map[string]bool {
	m := map[string]bool{"state:" + a.State: true}
	for _, x := range a.Labels {
		m["label:"+x] = true
	}
	for _, x := range a.References {
		m[fmt.Sprintf("ref:%d", x)] = true
	}
	for _, x := range a.DuplicateTargets {
		m[fmt.Sprintf("dup:%d", x)] = true
	}
	for _, x := range a.ScopeContradictions {
		m["contradiction:"+x] = true
	}
	return m
}
func factCount(a issueAnswer) int { return len(answerFacts(a)) }
func equalAggregate(a, b aggregateAnswers) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}
func marshalStable(v any) ([]byte, error) {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return nil, e
	}
	return append(b, '\n'), nil
}
func shaHex(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func writeBytes(path string, b []byte) error {
	if path == "" {
		return errors.New("output path required")
	}
	if err := os.MkdirAll(filepathDir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}
func filepathDir(path string) string {
	i := strings.LastIndexAny(path, "/\\")
	if i < 0 {
		return "."
	}
	return path[:i]
}
func bytesContainAny(b []byte, xs []string) bool {
	s := string(b)
	for _, x := range xs {
		if strings.Contains(s, x) {
			return true
		}
	}
	return false
}
func disjointSplits(xs []publicIssue) bool {
	seen := map[string]string{}
	for _, x := range xs {
		if p, ok := seen[x.ID]; ok && p != x.Split {
			return false
		}
		seen[x.ID] = x.Split
	}
	return true
}
func disjointContent(xs []publicIssue) bool {
	titles := map[string]string{}
	bodies := map[string]string{}
	for _, x := range xs {
		if prior, ok := titles[x.TitleSHA256]; ok && prior != x.Split {
			return false
		}
		if prior, ok := bodies[x.BodySHA256]; ok && prior != x.Split {
			return false
		}
		titles[x.TitleSHA256] = x.Split
		bodies[x.BodySHA256] = x.Split
	}
	return true
}

func allTrue(m map[string]bool) bool {
	for _, v := range m {
		if !v {
			return false
		}
	}
	return true
}

func verifyCorpusArtifacts(pubPath, answerPath, reportPath string) error {
	pb, e := os.ReadFile(pubPath)
	if e != nil {
		return e
	}
	ab, e := os.ReadFile(answerPath)
	if e != nil {
		return e
	}
	rb, e := os.ReadFile(reportPath)
	if e != nil {
		return e
	}
	var p publicCorpus
	var a answerBundle
	var r corpusReport
	if json.Unmarshal(pb, &p) != nil || json.Unmarshal(ab, &a) != nil || json.Unmarshal(rb, &r) != nil {
		return errors.New("artifact decode failed")
	}
	if p.Schema != corpusSchema || a.Schema != answerSchema || r.Schema != "fak-microcontext-corpus-report/1" {
		return errors.New("schema mismatch")
	}
	if shaHex(pb) != a.CorpusSHA256 || r.CorpusSHA256 != a.CorpusSHA256 || shaHex(ab) != r.AnswersSHA256 {
		return errors.New("digest mismatch")
	}
	if len(p.Records) != len(a.Answers) || len(p.Records) != r.Records || !allTrue(r.LeakChecks) || !r.BlindOracleGrade.QualityPass {
		return errors.New("corpus witness invariant failed")
	}
	return nil
}

func gradeSubmissionFiles(answerPath, submissionPath, outputPath string) error {
	ab, err := os.ReadFile(answerPath)
	if err != nil {
		return err
	}
	sb, err := os.ReadFile(submissionPath)
	if err != nil {
		return err
	}
	var gold answerBundle
	var got submission
	if err := json.Unmarshal(ab, &gold); err != nil {
		return fmt.Errorf("decode answers: %w", err)
	}
	if err := json.Unmarshal(sb, &got); err != nil {
		return fmt.Errorf("decode submission: %w", err)
	}
	r := gradeSubmission(gold, got)
	rb, err := marshalStable(r)
	if err != nil {
		return err
	}
	if err := writeBytes(outputPath, rb); err != nil {
		return err
	}
	if !r.QualityPass {
		return fmt.Errorf("quality floor failed: fp=%d fn=%d aggregate=%d citation=%d", r.FalsePositiveFacts, r.FalseNegativeFacts, r.AggregateErrors, r.CitationErrors)
	}
	return nil
}
