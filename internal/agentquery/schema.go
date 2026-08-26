package agentquery

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/maputil"
)

const DescriptorSchema = "fak-agents-schema/1"

type FieldDescriptor struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Nullable bool   `json:"nullable"`
}
type Descriptor struct {
	Schema          string            `json:"schema"`
	RelationSchema  string            `json:"relation_schema"`
	GroupSchema     string            `json:"group_schema"`
	ListPlanSchema  string            `json:"list_plan_schema"`
	QueryPlanSchema string            `json:"query_plan_schema"`
	Fields          []FieldDescriptor `json:"fields"`
	Sources         []string          `json:"sources"`
	Filters         []string          `json:"filters"`
	Sorts           []string          `json:"sorts"`
	Aggregates      []string          `json:"aggregates"`
	MaxRows         int               `json:"max_rows"`
	QueryMaxBytes   int               `json:"query_max_bytes"`
}

func SchemaDescriptor() Descriptor {
	return Descriptor{Schema: DescriptorSchema, RelationSchema: Schema, GroupSchema: GroupSchema, ListPlanSchema: ListPlanSchema, QueryPlanSchema: QueryPlanSchema, Fields: rowFieldDescriptors(), Sources: []string{"live", "history", "union"}, Filters: []string{"state", "liveness", "owner", "host", "lane", "group", "model", "provider", "root_id", "parent_id", "started_after", "started_before"}, Sorts: []string{"elapsed_desc", "elapsed_asc", "progress_age_desc", "progress_age_asc", "started_desc", "started_asc", "ended_desc", "ended_asc", "cost_desc", "cost_asc", "identity_asc", "identity_desc"}, Aggregates: []string{"count", "min_elapsed_ms", "max_elapsed_ms", "sum_elapsed_ms", "avg_elapsed_ms"}, MaxRows: 10000, QueryMaxBytes: 4096}
}
func rowFieldDescriptors() []FieldDescriptor {
	t := reflect.TypeOf(Row{})
	out := make([]FieldDescriptor, 0, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		name := strings.Split(f.Tag.Get("json"), ",")[0]
		typ, nullable := fieldType(f.Type)
		out = append(out, FieldDescriptor{Name: name, Type: typ, Nullable: nullable})
	}
	return out
}
func fieldType(t reflect.Type) (string, bool) {
	nullable := t.Kind() == reflect.Pointer
	if nullable {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return "string", nullable
	case reflect.Bool:
		return "boolean", nullable
	case reflect.Int, reflect.Int64:
		return "integer", nullable
	case reflect.Float64:
		return "number", nullable
	}
	return t.String(), nullable
}

func ValidateResult(r Result) error {
	if r.Metadata.Schema != Schema {
		return fmt.Errorf("metadata schema: want %s", Schema)
	}
	validSource := false
	for _, s := range SchemaDescriptor().Sources {
		validSource = validSource || r.Metadata.Source == s
	}
	if !validSource {
		return fmt.Errorf("invalid source")
	}
	if _, err := time.Parse(time.RFC3339, r.Metadata.ObservedAt); err != nil {
		return fmt.Errorf("invalid observed_at")
	}
	if r.Metadata.Limit < 1 || r.Metadata.Limit > SchemaDescriptor().MaxRows {
		return fmt.Errorf("invalid limit")
	}
	if r.Metadata.ListPlan != nil {
		if err := ValidateListPlan(*r.Metadata.ListPlan); err != nil {
			return err
		}
	}
	for _, row := range r.Rows {
		if strings.TrimSpace(row.AgentID) == "" || strings.TrimSpace(row.LogicalSessionID) == "" {
			return fmt.Errorf("row identity is required")
		}
		if row.Source != "live" && row.Source != "history" {
			return fmt.Errorf("invalid row source")
		}
		if _, err := time.Parse(time.RFC3339, row.ObservedAt); err != nil {
			return fmt.Errorf("invalid row observed_at")
		}
	}
	return nil
}

// ValidateGroupResult keeps grouped adapters on the same versioned contract as the CLI.
func ValidateGroupResult(r GroupResult) error {
	if r.Metadata.Schema != GroupSchema {
		return fmt.Errorf("invalid group schema")
	}
	if !descriptorContains(SchemaDescriptor().Sources, r.Metadata.Source) {
		return fmt.Errorf("invalid source")
	}
	since, err := time.Parse(time.RFC3339, r.Metadata.Since)
	if err != nil {
		return fmt.Errorf("invalid since")
	}
	observed, err := time.Parse(time.RFC3339, r.Metadata.ObservedAt)
	if err != nil || since.After(observed) {
		return fmt.Errorf("invalid observed_at")
	}
	if err := validateGroupedPlan(r.Metadata.Plan); err != nil {
		return err
	}
	if r.Metadata.InputRows < 0 || r.Metadata.MatchedRows < 0 || r.Metadata.MatchedRows > r.Metadata.InputRows {
		return fmt.Errorf("invalid row counts")
	}
	matched := 0
	for _, row := range r.Rows {
		if strings.TrimSpace(row.State) == "" || row.Count < 1 {
			return fmt.Errorf("invalid group row")
		}
		if row.MaxElapsedMS != nil && *row.MaxElapsedMS < 0 {
			return fmt.Errorf("invalid max_elapsed_ms")
		}
		matched += row.Count
	}
	if matched != r.Metadata.MatchedRows {
		return fmt.Errorf("group counts do not match metadata")
	}
	return nil
}

func validateGroupedPlan(p QueryPlan) error {
	if p.Schema != QueryPlanSchema || p.Source != "history" || p.HistoryWindowSeconds < 1 || p.HistoryWindowSeconds > int64((3650*24*time.Hour)/time.Second) ||
		!reflect.DeepEqual(p.GroupBy, []string{"lane", "state"}) || (!reflect.DeepEqual(p.Aggregates, baseAggregates) && !reflect.DeepEqual(p.Aggregates, fullAggregates)) ||
		p.OrderBy != "max_elapsed_ms_desc" || (p.TimeColumn != "started_at" && p.TimeColumn != "observed_at") {
		return fmt.Errorf("invalid grouped query plan")
	}
	return nil
}

func descriptorContains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

// DescriptorFieldNames provides the canonical row-key set for adapters and drift tests.
func DescriptorFieldNames() []string {
	d := SchemaDescriptor()
	out := make([]string, len(d.Fields))
	for i, f := range d.Fields {
		out[i] = f.Name
	}
	return out
}
func MarshalRowKeys() ([]string, error) {
	b, _ := json.Marshal(Row{})
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	out := maputil.SortedKeys(m)
	return out, nil
}
