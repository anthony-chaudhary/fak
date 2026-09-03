package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

const (
	vcacheFixtureEventSchema = "fak.vcache.fixture-event/v1"
	vcacheInspectSchema      = "fak.vcache.inspect/v1"
	vcacheFixturePutSchema   = "fak.vcache.fixture-put/v1"
	vcacheFixtureEventsFile  = "vcache-fixtures.jsonl"
	vcacheFixtureMaxPayload  = 1 << 20
	vcacheFixtureMaxArgs     = 64 << 10
	vcacheFixtureMaxLedger   = 16 << 20
	vcacheFixtureMaxLine     = 2 << 20
)

type vcacheFixtureMetadata struct {
	Plane            string `json:"plane"`
	Producer         string `json:"producer"`
	Tool             string `json:"tool"`
	ArgsDigest       string `json:"args_digest"`
	AdmittedAtEpoch  string `json:"admitted_at_epoch"`
	Witness          string `json:"witness,omitempty"`
	MediaType        string `json:"media_type"`
	PayloadBytes     int64  `json:"payload_bytes"`
	LengthUnit       string `json:"length_unit"`
	ResidencyTier    string `json:"residency_tier"`
	InvalidationMode string `json:"invalidation_mode"`
	Eligibility      string `json:"eligibility"`
	FixtureMode      string `json:"fixture_mode"`
}

type vcacheFixtureEvent struct {
	Schema   string                `json:"schema"`
	Event    string                `json:"event"`
	Digest   string                `json:"digest"`
	Metadata vcacheFixtureMetadata `json:"metadata"`
	Payload  []byte                `json:"payload"`
}

type vcacheFixturePutReport struct {
	Schema   string                `json:"schema"`
	Stored   bool                  `json:"stored"`
	Digest   string                `json:"digest"`
	Metadata vcacheFixtureMetadata `json:"metadata"`
}

type vcacheInspectReport struct {
	Schema        string                `json:"schema"`
	Digest        string                `json:"digest"`
	Metadata      vcacheFixtureMetadata `json:"metadata"`
	PayloadHidden bool                  `json:"payload_hidden"`
	Payload       *string               `json:"payload,omitempty"`
	PayloadBase64 []byte                `json:"payload_base64,omitempty"`
}

func runVCacheFixturePut(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache put", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "absolute caller-owned fixture directory")
	payloadText := fs.String("payload", "", "literal fixture payload")
	payloadFile := fs.String("payload-file", "", "absolute regular file containing fixture payload bytes")
	fixtureMode := fs.String("fixture-mode", "", "required fixture gate: offline or test")
	tool := fs.String("tool", "fixture_read", "read-only fixture tool name")
	argsText := fs.String("args", "{}", "JSON arguments submitted through the vDSO event path")
	witness := fs.String("witness", "", "optional external world-state witness")
	asJSON := fs.Bool("json", false, "emit JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	*payloadFile = pathutil.ExpandTilde(*payloadFile)
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak vcache put: unexpected positional arguments")
		return 2
	}
	if *fixtureMode != "offline" && *fixtureMode != "test" {
		fmt.Fprintln(stderr, "fak vcache put: refused: --fixture-mode must be offline or test")
		return 2
	}
	textSet, fileSet := flagWasSet(fs, "payload"), flagWasSet(fs, "payload-file")
	if textSet == fileSet {
		fmt.Fprintln(stderr, "fak vcache put: exactly one of --payload or --payload-file is required")
		return 2
	}
	if err := validateVCacheFixtureTool(*tool); err != nil {
		fmt.Fprintf(stderr, "fak vcache put: %v\n", err)
		return 2
	}
	if len(*argsText) > vcacheFixtureMaxArgs || !json.Valid([]byte(*argsText)) {
		fmt.Fprintln(stderr, "fak vcache put: --args must be valid JSON no larger than 65536 bytes")
		return 2
	}
	if strings.ContainsAny(*witness, "\r\n") || len(*witness) > 1024 {
		fmt.Fprintln(stderr, "fak vcache put: --witness must be a single line no larger than 1024 bytes")
		return 2
	}

	root, err := validateVCacheFixtureDir(*dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache put: %v\n", err)
		return 2
	}
	var payload []byte
	if textSet {
		payload = []byte(*payloadText)
	} else {
		payload, err = readVCacheFixturePayload(*payloadFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache put: %v\n", err)
			return 2
		}
	}
	if len(payload) > vcacheFixtureMaxPayload {
		fmt.Fprintf(stderr, "fak vcache put: payload is %d bytes; maximum is %d\n", len(payload), vcacheFixtureMaxPayload)
		return 2
	}

	record, err := vcacheFixtureFillEvent(*tool, []byte(*argsText), payload, *witness, *fixtureMode)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache put: %v\n", err)
		return 1
	}
	eventsPath := filepath.Join(root, vcacheFixtureEventsFile)
	existing, err := readVCacheFixtureEvents(eventsPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache put: refused fixture ledger: %v\n", err)
		return 1
	}
	for _, ev := range existing {
		if ev.Digest == record.Digest {
			fmt.Fprintf(stderr, "fak vcache put: refused duplicate digest %s\n", record.Digest)
			return 1
		}
	}
	if err := appendVCacheFixtureEvent(eventsPath, record); err != nil {
		fmt.Fprintf(stderr, "fak vcache put: %v\n", err)
		return 1
	}
	report := vcacheFixturePutReport{Schema: vcacheFixturePutSchema, Stored: true, Digest: record.Digest, Metadata: record.Metadata}
	if *asJSON {
		return writeVCacheJSON(stdout, stderr, "put", report)
	}
	fmt.Fprintf(stdout, "stored fixture digest=%s bytes=%d tool=%s\n", report.Digest, report.Metadata.PayloadBytes, report.Metadata.Tool)
	return 0
}

func runVCacheInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "absolute caller-owned fixture directory")
	digestText := fs.String("digest", "", "entry SHA-256 (64 lowercase hex; sha256: prefix accepted)")
	showPayload := fs.Bool("show-payload", false, "include payload bytes (test fixture mode only)")
	fixtureMode := fs.String("fixture-mode", "", "payload policy gate: test")
	asJSON := fs.Bool("json", false, "emit JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak vcache inspect: unexpected positional arguments")
		return 2
	}
	if *showPayload && *fixtureMode != "test" {
		fmt.Fprintln(stderr, "fak vcache inspect: refused: --show-payload requires --fixture-mode test")
		return 2
	}
	if *fixtureMode != "" && *fixtureMode != "test" {
		fmt.Fprintln(stderr, "fak vcache inspect: --fixture-mode must be test when provided")
		return 2
	}
	digest, err := normalizeVCacheDigest(*digestText)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache inspect: %v\n", err)
		return 2
	}
	root, err := validateVCacheFixtureDir(*dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache inspect: %v\n", err)
		return 2
	}
	events, err := readVCacheFixtureEvents(filepath.Join(root, vcacheFixtureEventsFile))
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache inspect: refused fixture ledger: %v\n", err)
		return 1
	}
	var found *vcacheFixtureEvent
	for i := range events {
		if events[i].Digest != digest {
			continue
		}
		if found != nil {
			fmt.Fprintf(stderr, "fak vcache inspect: refused ambiguous duplicate digest %s\n", digest)
			return 1
		}
		found = &events[i]
	}
	if found == nil {
		fmt.Fprintf(stderr, "fak vcache inspect: digest %s not found\n", digest)
		return 1
	}
	report := vcacheInspectReport{Schema: vcacheInspectSchema, Digest: found.Digest, Metadata: found.Metadata, PayloadHidden: !*showPayload}
	if *showPayload {
		if utf8.Valid(found.Payload) {
			payload := string(found.Payload)
			report.Payload = &payload
		} else {
			report.PayloadBase64 = append([]byte(nil), found.Payload...)
		}
	}
	if *asJSON {
		return writeVCacheJSON(stdout, stderr, "inspect", report)
	}
	writeVCacheInspectText(stdout, report)
	return 0
}

// vcacheFixtureFillEvent deliberately seeds through VDSO.Emit rather than minting
// cache metadata by hand. The fixture therefore inherits the live read-only,
// idempotent eligibility gate and the same fill event shape as the runtime path.
func vcacheFixtureFillEvent(tool string, args, payload []byte, witness, fixtureMode string) (vcacheFixtureEvent, error) {
	v := vdso.New(1)
	var fills []vdso.CacheEvent
	v.SetCacheEventSink(func(ev vdso.CacheEvent) {
		if ev.Kind == vdso.CacheFill {
			fills = append(fills, ev)
		}
	})
	meta := map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
	if witness != "" {
		meta["witness"] = witness
	}
	call := &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), args...)}, Meta: meta}
	result := &abi.Result{
		Call: call, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), payload...), Len: int64(len(payload))},
	}
	v.Emit(abi.Event{Kind: abi.EvComplete, Call: call, Result: result})
	if len(fills) != 1 {
		return vcacheFixtureEvent{}, fmt.Errorf("normal vDSO event path emitted %d fill events, want 1", len(fills))
	}
	e := fills[0].Entry
	record := vcacheFixtureEvent{
		Schema: vcacheFixtureEventSchema,
		Event:  string(vdso.CacheFill),
		Digest: e.ID.Digest,
		Metadata: vcacheFixtureMetadata{
			Plane:            string(e.Plane),
			Producer:         e.Derivation.Producer,
			Tool:             e.Derivation.Tool,
			ArgsDigest:       e.Derivation.ArgsDigest,
			AdmittedAtEpoch:  e.Validity.AdmittedAtEpoch,
			Witness:          e.Validity.Witness,
			MediaType:        string(e.ID.MediaType),
			PayloadBytes:     e.ID.Length,
			LengthUnit:       string(e.ID.Unit),
			ResidencyTier:    string(e.Residency.Tier),
			InvalidationMode: string(e.Coherence.InvalidationMode),
			Eligibility:      "read_only+idempotent",
			FixtureMode:      fixtureMode,
		},
		Payload: append([]byte(nil), payload...),
	}
	if err := validateVCacheFixtureEvent(record); err != nil {
		return vcacheFixtureEvent{}, fmt.Errorf("emitted invalid fixture event: %w", err)
	}
	return record, nil
}

func validateVCacheFixtureEvent(ev vcacheFixtureEvent) error {
	if ev.Schema != vcacheFixtureEventSchema {
		return fmt.Errorf("unknown schema %q", ev.Schema)
	}
	if ev.Event != string(vdso.CacheFill) {
		return fmt.Errorf("event %q is not a vDSO fill", ev.Event)
	}
	digest, err := normalizeVCacheDigest(ev.Digest)
	if err != nil || digest != ev.Digest {
		return fmt.Errorf("non-canonical digest %q", ev.Digest)
	}
	if payloadSHA256(ev.Payload) != ev.Digest {
		return fmt.Errorf("payload digest mismatch for %s", ev.Digest)
	}
	m := ev.Metadata
	if m.Plane != "tool_result" || m.Producer != "vdso" || m.MediaType != "bytes" || m.LengthUnit != "bytes" || m.ResidencyTier != "dram" {
		return errors.New("metadata is not a resident vDSO tool-result byte entry")
	}
	if err := validateVCacheFixtureTool(m.Tool); err != nil {
		return err
	}
	if !isLowerHex(m.ArgsDigest, 24) {
		return fmt.Errorf("invalid args digest %q", m.ArgsDigest)
	}
	if m.AdmittedAtEpoch != "0" {
		return fmt.Errorf("unexpected fixture admission epoch %q", m.AdmittedAtEpoch)
	}
	if m.PayloadBytes != int64(len(ev.Payload)) || m.PayloadBytes < 0 || m.PayloadBytes > vcacheFixtureMaxPayload {
		return fmt.Errorf("payload length metadata %d does not match %d bytes", m.PayloadBytes, len(ev.Payload))
	}
	if m.Eligibility != "read_only+idempotent" {
		return fmt.Errorf("ineligible fixture entry %q", m.Eligibility)
	}
	if m.FixtureMode != "offline" && m.FixtureMode != "test" {
		return fmt.Errorf("invalid fixture mode %q", m.FixtureMode)
	}
	wantInvalidation := "write_epoch"
	if m.Witness != "" {
		wantInvalidation = "external_refutation"
	}
	if m.InvalidationMode != wantInvalidation {
		return fmt.Errorf("invalidation mode %q does not match witness boundary", m.InvalidationMode)
	}
	return nil
}

func validateVCacheFixtureDir(path string) (string, error) {
	if path == "" {
		return "", errors.New("--dir is required")
	}
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return "", errors.New("--dir must be an absolute clean path")
	}
	info, err := os.Lstat(path)
	if err != nil {
		return "", fmt.Errorf("open --dir: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return "", errors.New("--dir must name an existing non-symlink directory")
	}
	return path, nil
}

func readVCacheFixturePayload(path string) ([]byte, error) {
	if path == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, errors.New("--payload-file must be an absolute clean path")
	}
	info, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("inspect --payload-file: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return nil, errors.New("--payload-file must name a non-symlink regular file")
	}
	if info.Size() > vcacheFixtureMaxPayload {
		return nil, fmt.Errorf("payload file is %d bytes; maximum is %d", info.Size(), vcacheFixtureMaxPayload)
	}
	return os.ReadFile(path)
}

func readVCacheFixtureEvents(path string) ([]vcacheFixtureEvent, error) {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return nil, errors.New("event path must be a non-symlink regular file")
	}
	if info.Mode().Perm()&0o077 != 0 {
		return nil, errors.New("event ledger must have no group or world permissions")
	}
	if info.Size() > vcacheFixtureMaxLedger {
		return nil, fmt.Errorf("event ledger is %d bytes; maximum is %d", info.Size(), vcacheFixtureMaxLedger)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64<<10), vcacheFixtureMaxLine)
	var events []vcacheFixtureEvent
	for line := 1; scanner.Scan(); line++ {
		if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
			return nil, fmt.Errorf("line %d is empty", line)
		}
		var ev vcacheFixtureEvent
		dec := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&ev); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if err := requireJSONEOF(dec); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if err := validateVCacheFixtureEvent(ev); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		events = append(events, ev)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return events, nil
}

func appendVCacheFixtureEvent(path string, ev vcacheFixtureEvent) error {
	var line bytes.Buffer
	if err := json.NewEncoder(&line).Encode(ev); err != nil {
		return err
	}
	if line.Len() > vcacheFixtureMaxLine {
		return fmt.Errorf("encoded fixture event is %d bytes; maximum is %d", line.Len(), vcacheFixtureMaxLine)
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	closed := false
	defer func() {
		if !closed {
			_ = f.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Mode().Perm()&0o077 != 0 {
		return errors.New("event ledger must be a private regular file")
	}
	if _, err := f.Write(line.Bytes()); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	closed = true
	return nil
}

func normalizeVCacheDigest(value string) (string, error) {
	value = strings.TrimPrefix(value, "sha256:")
	if !isLowerHex(value, sha256.Size*2) {
		return "", errors.New("--digest must be 64 lowercase hexadecimal SHA-256 characters")
	}
	return value, nil
}

func payloadSHA256(payload []byte) string {
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

func isLowerHex(value string, size int) bool {
	if len(value) != size {
		return false
	}
	for _, c := range []byte(value) {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func validateVCacheFixtureTool(tool string) error {
	if tool == "" || len(tool) > 128 {
		return errors.New("--tool must contain 1 to 128 safe ASCII characters")
	}
	for _, c := range []byte(tool) {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' {
			continue
		}
		return errors.New("--tool may contain only ASCII letters, digits, dot, dash, and underscore")
	}
	return nil
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func writeVCacheJSON(stdout, stderr io.Writer, command string, value any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		fmt.Fprintf(stderr, "fak vcache %s: encode output: %v\n", command, err)
		return 1
	}
	return 0
}

func writeVCacheInspectText(w io.Writer, report vcacheInspectReport) {
	fmt.Fprintf(w, "digest: %s\n", report.Digest)
	fmt.Fprintln(w, "event: fill")
	fmt.Fprintf(w, "tool: %s\n", report.Metadata.Tool)
	fmt.Fprintf(w, "args_digest: %s\n", report.Metadata.ArgsDigest)
	fmt.Fprintf(w, "payload_bytes: %d\n", report.Metadata.PayloadBytes)
	fmt.Fprintf(w, "fixture_mode: %s\n", report.Metadata.FixtureMode)
	if report.Metadata.Witness != "" {
		fmt.Fprintf(w, "witness: %s\n", report.Metadata.Witness)
	}
	if report.PayloadHidden {
		fmt.Fprintln(w, "payload: hidden")
	} else if report.Payload != nil {
		fmt.Fprintf(w, "payload: %s\n", *report.Payload)
	} else {
		encoded, _ := json.Marshal(report.PayloadBase64)
		fmt.Fprintf(w, "payload_base64: %s\n", encoded)
	}
}
