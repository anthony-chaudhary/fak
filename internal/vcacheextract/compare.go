package vcacheextract

import (
	"bytes"
	"os"
	"path/filepath"
	"time"
)

// ComparisonArm is one same-workload Codex token-sanitization arm. Unavailable
// integration and external arms deliberately retain zero measurements: an
// in-process adapter is not a witness for the real telemetry pipeline.
type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	CounterCorrect  bool
	Latency         time.Duration
	InputRows       int
	EligibleRows    int
	OutputRows      int
	MissedRows      int
	ExtraRows       int
	ForbiddenFields int
	ForbiddenBytes  int
	ParseFailures   int
	InputBytes      int64
	OutputBytes     int64
	CPUSeconds      float64
	PeakRSSBytes    int64
	NetworkBytes    int64
	CostUSD         float64
	Note            string
}

type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

var comparisonSecrets = [][]byte{[]byte("PROMPT_SECRET"), []byte("TOOL_SECRET"), []byte("RESPONSE_SECRET")}

func comparisonFixture() []byte {
	return []byte("{\"type\":\"response_item\",\"payload\":{\"content\":\"RESPONSE_SECRET\"}}\n" +
		"{\"type\":\"event_msg\",\"payload\":{\"type\":\"token_count\",\"info\":{\"last_token_usage\":{\"input_tokens\":100,\"cached_input_tokens\":80},\"prompt_text\":\"PROMPT_SECRET\"}}}\n" +
		"{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":250,\"cached_input_tokens\":200,\"output_tokens\":17},\"response\":\"RESPONSE_SECRET\"}\n" +
		"{\"type\":\"tool_call\",\"arguments\":\"TOOL_SECRET\"}\n")
}

func forbiddenCounts(body []byte) (fields, secrets int) {
	for _, field := range [][]byte{[]byte("prompt_text"), []byte("arguments"), []byte("content"), []byte("response"), []byte("output_tokens")} {
		fields += bytes.Count(body, field)
	}
	for _, secret := range comparisonSecrets {
		secrets += bytes.Count(body, secret)
	}
	return fields, secrets
}

func exactComparisonCounters(rows []map[string]any) bool {
	if len(rows) != 2 {
		return false
	}
	t0, c0, ok0 := sanitizedUsagePair(rows[0])
	t1, c1, ok1 := sanitizedUsagePair(rows[1])
	return ok0 && ok1 && t0 == 100 && c0 == 80 && t1 == 250 && c1 == 200
}

// CompareLocal executes only the native sanitizer and a tuned raw pass-through.
// Real fak integrations and external processors remain unavailable until their
// actual product boundaries run this exact fixture and an independent read-back.
func CompareLocal() ComparisonResult {
	fixture := comparisonFixture()
	dir, err := os.MkdirTemp("", "fak-vcache-extract-compare-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	input := filepath.Join(dir, "session.jsonl")
	output := filepath.Join(dir, "sanitized.jsonl")
	if err = os.WriteFile(input, fixture, 0o600); err != nil {
		panic(err)
	}

	start := time.Now()
	rows, extractErr := ExtractRows(input)
	writeErr := error(nil)
	if extractErr == nil {
		writeErr = WriteRows(output, rows, nil)
	}
	nativeLatency := time.Since(start)
	nativeBody, readErr := os.ReadFile(output)
	fields, secrets := forbiddenCounts(nativeBody)
	parseFailures := 0
	if extractErr != nil || writeErr != nil || readErr != nil {
		parseFailures = 1
	}
	nativeCorrect := parseFailures == 0 && exactComparisonCounters(rows) && fields == 0 && secrets == 0

	start = time.Now()
	rawBody := append([]byte(nil), fixture...)
	rawLatency := time.Since(start)
	rawFields, rawSecrets := forbiddenCounts(rawBody)

	return ComparisonResult{
		Workload: "sanitize four ordered mixed Codex JSONL records, preserve two exact token-counter rows, and emit no prompt/tool/response content",
		Arms: []ComparisonArm{
			{Name: "fak native Codex token sanitizer", Kind: "native", Available: true, Correct: nativeCorrect, CounterCorrect: exactComparisonCounters(rows), Latency: nativeLatency, InputRows: 4, EligibleRows: 2, OutputRows: len(rows), ForbiddenFields: fields, ForbiddenBytes: secrets, ParseFailures: parseFailures, InputBytes: int64(len(fixture)), OutputBytes: int64(len(nativeBody)), Note: "streaming JSONL decode plus allowlisted token-counter reconstruction"},
			{Name: "raw JSONL pass-through", Kind: "baseline", Available: true, Correct: false, CounterCorrect: true, Latency: rawLatency, InputRows: 4, EligibleRows: 2, OutputRows: 4, ExtraRows: 2, ForbiddenFields: rawFields, ForbiddenBytes: rawSecrets, InputBytes: int64(len(fixture)), OutputBytes: int64(len(rawBody)), Note: "tuned no-feature baseline preserves counters but leaks content and emits ineligible rows"},
			{Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires the real collector/exporter boundary and content-leak read-back"},
			{Name: "fak + Prometheus", Kind: "integration", Note: "requires the real metric export/scrape boundary and content-leak read-back"},
			{Name: "jq streaming projection", Kind: "external", Note: "requires a pinned jq binary and real streaming filter"},
			{Name: "Vector VRL remap", Kind: "external", Note: "requires a pinned Vector pipeline and sink read-back"},
			{Name: "Fluent Bit filter pipeline", Kind: "external", Note: "requires a pinned Fluent Bit pipeline and sink read-back"},
		},
	}
}
