package sharedtask

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// SchemaArtifactRef is the envelope schema name for a standalone disaggregated
// artifact reference — the sixth contract schema; the other five are declared in
// sharedtask.go and journal.go.
const SchemaArtifactRef = "fak.shared-artifact-ref.v1"

// contractSchemaFiles maps an envelope schema name to its JSON-schema file under
// the schema dir (tools/schemas in this repo). The schema files stay the single
// source of truth for the wire contract; this validator only interprets them.
var contractSchemaFiles = map[string]string{
	SchemaTask:        "shared-task.v1.json",
	SchemaEvent:       "shared-event.v1.json",
	SchemaPatch:       "shared-patch.v1.json",
	SchemaPatchResult: "shared-patch-result.v1.json",
	SchemaArtifactRef: "shared-artifact-ref.v1.json",
	SchemaTaskJournal: "shared-task-journal.v1.json",
}

// decodeContractJSON decodes one JSON value keeping numbers as json.Number, so the
// validator can distinguish an integer from a float the way the contract requires
// (a "bytes": 5.0 must fail an integer check even though Go would otherwise fold
// both into float64).
func decodeContractJSON(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

// LoadContractSchema reads and parses the named contract schema from schemaDir.
func LoadContractSchema(schemaDir, name string) (map[string]any, error) {
	rel, ok := contractSchemaFiles[name]
	if !ok {
		return nil, fmt.Errorf("unknown schema %q", name)
	}
	data, err := os.ReadFile(filepath.Join(schemaDir, rel))
	if err != nil {
		return nil, err
	}
	v, err := decodeContractJSON(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	schema, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s: schema is not an object", name)
	}
	return schema, nil
}

// resolveContractRef follows a local "#/definitions/..." $ref inside root. A schema
// without $ref resolves to itself.
func resolveContractRef(schema, root map[string]any) (map[string]any, error) {
	ref, ok := schema["$ref"].(string)
	if !ok || ref == "" {
		return schema, nil
	}
	var node any = root
	for _, part := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
		obj, ok := node.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("ref %q does not resolve to an object", ref)
		}
		node = obj[part]
	}
	resolved, ok := node.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("ref %q does not resolve to an object", ref)
	}
	return resolved, nil
}

// contractInt unwraps a JSON integer: a json.Number with an exact int64 value.
// A bool, a float, or any other type is not an integer.
func contractInt(v any) (int64, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	i, err := n.Int64()
	if err != nil {
		return 0, false
	}
	return i, true
}

// schemaInt reads an integer-valued keyword (minLength, minimum, minItems) from a
// schema, defaulting to def when absent or malformed.
func schemaInt(schema map[string]any, key string, def int64) int64 {
	if v, ok := contractInt(schema[key]); ok {
		return v
	}
	return def
}

// validateValue is the recursive core: it checks instance against schema (with
// $refs resolved against root) and returns the first violation as an error whose
// message carries the JSON path. It interprets exactly the keywords the six
// contract schemas use: type, const, enum, pattern, minLength, minimum, required,
// properties, minItems, items, $ref.
func validateValue(instance any, schema, root map[string]any, path string) error {
	schema, err := resolveContractRef(schema, root)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	switch schema["type"] {
	case "object":
		if _, ok := instance.(map[string]any); !ok {
			return fmt.Errorf("%s: want object", path)
		}
	case "array":
		if _, ok := instance.([]any); !ok {
			return fmt.Errorf("%s: want array", path)
		}
	case "string":
		if _, ok := instance.(string); !ok {
			return fmt.Errorf("%s: want string", path)
		}
	case "integer":
		if _, ok := contractInt(instance); !ok {
			return fmt.Errorf("%s: want integer", path)
		}
	}
	if want, ok := schema["const"]; ok && !contractEqual(instance, want) {
		return fmt.Errorf("%s: want %v, got %v", path, want, instance)
	}
	if enum, ok := schema["enum"].([]any); ok {
		found := false
		for _, allowed := range enum {
			if contractEqual(instance, allowed) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("%s: %v not in enum", path, instance)
		}
	}
	if s, ok := instance.(string); ok {
		if pattern, ok := schema["pattern"].(string); ok {
			re, err := regexp.Compile(pattern)
			if err != nil {
				return fmt.Errorf("%s: bad pattern %q: %w", path, pattern, err)
			}
			if !re.MatchString(s) {
				return fmt.Errorf("%s: %q does not match %q", path, s, pattern)
			}
		}
		if int64(utf8.RuneCountInString(s)) < schemaInt(schema, "minLength", 0) {
			return fmt.Errorf("%s: string too short", path)
		}
	}
	if i, ok := contractInt(instance); ok {
		if min, has := contractInt(schema["minimum"]); has && i < min {
			return fmt.Errorf("%s: below minimum", path)
		}
	}
	if obj, ok := instance.(map[string]any); ok {
		required, _ := schema["required"].([]any)
		var missing []string
		for _, key := range required {
			name, _ := key.(string)
			if _, present := obj[name]; !present {
				missing = append(missing, name)
			}
		}
		if len(missing) > 0 {
			return fmt.Errorf("%s: missing %v", path, missing)
		}
		props, _ := schema["properties"].(map[string]any)
		for _, key := range sortedContractKeys(obj) {
			propSchema, ok := props[key].(map[string]any)
			if !ok {
				continue
			}
			if err := validateValue(obj[key], propSchema, root, path+"."+key); err != nil {
				return err
			}
		}
	}
	if arr, ok := instance.([]any); ok {
		if int64(len(arr)) < schemaInt(schema, "minItems", 0) {
			return fmt.Errorf("%s: too few items", path)
		}
		if itemSchema, ok := schema["items"].(map[string]any); ok && len(itemSchema) > 0 {
			for i, item := range arr {
				if err := validateValue(item, itemSchema, root, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// contractEqual compares two decoded JSON values (const/enum semantics). The
// contract's const and enum values are all strings, but numbers compare by
// numeric string form for safety.
func contractEqual(a, b any) bool {
	if an, ok := a.(json.Number); ok {
		if bn, ok := b.(json.Number); ok {
			return an.String() == bn.String()
		}
	}
	return a == b
}

// sortedContractKeys returns the map keys in stable order so a multi-violation
// object always reports the same first error.
func sortedContractKeys(obj map[string]any) []string {
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ValidateEnvelope validates one decoded record against the schema named by its
// own "schema" field and returns that schema name.
func ValidateEnvelope(envelope map[string]any, schemaDir string) (string, error) {
	name, ok := envelope["schema"].(string)
	if !ok {
		return "", fmt.Errorf("envelope missing string schema")
	}
	schema, err := LoadContractSchema(schemaDir, name)
	if err != nil {
		return "", err
	}
	return name, validateValue(envelope, schema, schema, "$")
}

// ContractJSONFiles lists the JSON files to validate: the path itself when it is a
// file, else every *.json under the directory in stable sorted order.
func ContractJSONFiles(path string) ([]string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return []string{path}, nil
	}
	var out []string
	err = filepath.WalkDir(path, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".json") {
			out = append(out, p)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool { return filepath.ToSlash(out[i]) < filepath.ToSlash(out[j]) })
	return out, nil
}

// loadContractFile decodes one record file as a JSON object.
func loadContractFile(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	v, err := decodeContractJSON(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s: want object", path)
	}
	return obj, nil
}

// ValidateContractFiles envelope-validates each file and returns per-schema counts.
func ValidateContractFiles(schemaDir string, paths []string) (map[string]int, error) {
	counts := map[string]int{}
	for _, path := range paths {
		obj, err := loadContractFile(path)
		if err != nil {
			return nil, err
		}
		name, err := ValidateEnvelope(obj, schemaDir)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		counts[name]++
	}
	return counts, nil
}

var contractDocExampleRE = regexp.MustCompile("(?s)```json\n(.*?)\n```")

// ValidateContractDoc validates every ```json``` example block in the contract doc
// and returns per-schema counts, so the prose examples can never drift from the
// schemas they illustrate.
func ValidateContractDoc(schemaDir, docPath string) (map[string]int, error) {
	data, err := os.ReadFile(docPath)
	if err != nil {
		return nil, err
	}
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	counts := map[string]int{}
	for i, m := range contractDocExampleRE.FindAllStringSubmatch(text, -1) {
		v, err := decodeContractJSON([]byte(m[1]))
		if err != nil {
			return nil, fmt.Errorf("JSON example %d: %w", i+1, err)
		}
		obj, ok := v.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("JSON example %d is not an object", i+1)
		}
		name, err := ValidateEnvelope(obj, schemaDir)
		if err != nil {
			return nil, fmt.Errorf("JSON example %d: %w", i+1, err)
		}
		counts[name]++
	}
	return counts, nil
}

// loadValidatedContractDir validates every JSON file under path and returns the
// decoded records; a path with no JSON files is an error, not a silent pass.
func loadValidatedContractDir(schemaDir, path string) ([]map[string]any, map[string]int, error) {
	files, err := ContractJSONFiles(path)
	if err != nil {
		return nil, nil, err
	}
	var values []map[string]any
	counts := map[string]int{}
	for _, file := range files {
		obj, err := loadContractFile(file)
		if err != nil {
			return nil, nil, err
		}
		name, err := ValidateEnvelope(obj, schemaDir)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", file, err)
		}
		counts[name]++
		values = append(values, obj)
	}
	if len(values) == 0 {
		return nil, nil, fmt.Errorf("%s: no JSON files", path)
	}
	return values, counts, nil
}

// ValidateContractSequence checks the full collaborative-sequence fixture shape:
// a task record, at least one materialized journal bound to the same task with
// accepted-event snapshots, at least one accepted patch result, and a title
// replacement patch (the canonical co-edit the example narrates).
func ValidateContractSequence(schemaDir, path string) (map[string]int, error) {
	values, counts, err := loadValidatedContractDir(schemaDir, path)
	if err != nil {
		return nil, err
	}
	bySchema := map[string][]map[string]any{}
	for _, value := range values {
		name, _ := value["schema"].(string)
		bySchema[name] = append(bySchema[name], value)
	}
	tasks := bySchema[SchemaTask]
	journals := bySchema[SchemaTaskJournal]
	if len(tasks) == 0 {
		return nil, fmt.Errorf("sequence: missing task record")
	}
	if len(journals) == 0 {
		return nil, fmt.Errorf("sequence: missing materialized journal")
	}
	taskID, _ := tasks[0]["task_id"].(string)
	for _, journal := range journals {
		initial, _ := journal["initial"].(map[string]any)
		if journal["task_id"] != taskID || initial == nil || initial["task_id"] != taskID {
			return nil, fmt.Errorf("sequence: journal task mismatch")
		}
		if entries, _ := journal["entries"].([]any); len(entries) == 0 {
			return nil, fmt.Errorf("sequence: journal has no accepted-event snapshots")
		}
	}
	accepted := false
	for _, result := range bySchema[SchemaPatchResult] {
		if result["verdict"] == string(VerdictAccepted) {
			accepted = true
			break
		}
	}
	if !accepted {
		return nil, fmt.Errorf("sequence: no accepted patch result")
	}
	titlePatch := false
	for _, patch := range bySchema[SchemaPatch] {
		ops, _ := patch["ops"].([]any)
		for _, rawOp := range ops {
			op, _ := rawOp.(map[string]any)
			if op["op"] == "replace" && op["path"] == "/title" {
				titlePatch = true
			}
		}
	}
	if !titlePatch {
		return nil, fmt.Errorf("sequence: missing title replacement patch")
	}
	return counts, nil
}

// ValidateContractVerdicts checks the non-acceptance fixture shape: every
// non-accepted verdict a UI must render (needs_approval, denied, quarantined) is
// present, each carries a reason, and none advances the task revision.
func ValidateContractVerdicts(schemaDir, path string) (map[string]int, error) {
	values, counts, err := loadValidatedContractDir(schemaDir, path)
	if err != nil {
		return nil, err
	}
	verdicts := map[string]bool{}
	var results []map[string]any
	for _, value := range values {
		if value["schema"] == SchemaPatchResult {
			results = append(results, value)
			verdict, _ := value["verdict"].(string)
			verdicts[verdict] = true
		}
	}
	for _, want := range []Verdict{VerdictNeedsApproval, VerdictDenied, VerdictQuarantined} {
		if !verdicts[string(want)] {
			return nil, fmt.Errorf("verdicts: missing %s", want)
		}
	}
	for _, result := range results {
		if reason, _ := result["reason"].(string); reason == "" {
			return nil, fmt.Errorf("verdicts: %v missing reason", result["verdict"])
		}
		if result["current_rev"] != result["base_rev"] {
			return nil, fmt.Errorf("verdicts: %v advanced revision", result["verdict"])
		}
	}
	return counts, nil
}

// FormatContractCounts renders per-schema counts as the stable "name=n, ..."
// summary line the contract recipes print.
func FormatContractCounts(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, counts[k]))
	}
	return strings.Join(parts, ", ")
}
