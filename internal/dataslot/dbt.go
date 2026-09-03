package dataslot

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// DBTModel describes a governed dbt model node with columns, description, and lineage.
type DBTModel struct {
	Name        string   `json:"name"`
	Package     string   `json:"package_name,omitempty"`
	Description string   `json:"description,omitempty"`
	DependsOn   []string `json:"depends_on,omitempty"`
	Columns     []string `json:"columns,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

// DBTSemanticsReceipt records the verified semantic query answers from local dbt artifacts.
type DBTSemanticsReceipt struct {
	ArtifactPath   string              `json:"artifact_path"`
	ArtifactSHA256 string              `json:"artifact_sha256"`
	RawSQLDormant  bool                `json:"raw_sql_dormant"`
	ZeroNetwork    bool                `json:"zero_network"`
	ModelCount     int                 `json:"model_count"`
	Models         map[string]DBTModel `json:"models"`
	LineageDown    map[string][]string `json:"lineage_downstream,omitempty"`
}

// ReadDBTSemantics parses a dbt manifest.json artifact and constructs the model and lineage graph
// entirely in-memory with zero network requests and raw SQL staying dormant.
func ReadDBTSemantics(manifestPath string) (*DBTSemanticsReceipt, error) {
	cleanPath := manifestPath
	f, err := os.Open(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("dataslot: failed to open dbt manifest: %w", err)
	}
	defer f.Close()

	h := sha256.New()
	tr := io.TeeReader(f, h)

	var root struct {
		Nodes map[string]struct {
			Name         string `json:"name"`
			PackageName  string `json:"package_name"`
			Description  string `json:"description"`
			ResourceType string `json:"resource_type"`
			DependsOn    struct {
				Nodes []string `json:"nodes"`
			} `json:"depends_on"`
			Columns map[string]struct {
				Name string `json:"name"`
			} `json:"columns"`
			Tags []string `json:"tags"`
		} `json:"nodes"`
	}

	if err := json.NewDecoder(tr).Decode(&root); err != nil {
		return nil, fmt.Errorf("dataslot: failed to decode dbt manifest JSON: %w", err)
	}

	digest := hex.EncodeToString(h.Sum(nil))

	receipt := &DBTSemanticsReceipt{
		ArtifactPath:   cleanPath,
		ArtifactSHA256: digest,
		RawSQLDormant:  true,
		ZeroNetwork:    true,
		Models:         make(map[string]DBTModel),
		LineageDown:    make(map[string][]string),
	}

	for _, node := range root.Nodes {
		if node.ResourceType != "model" {
			continue
		}

		var deps []string
		for _, dep := range node.DependsOn.Nodes {
			// Trim model.pkg. prefix to canonical model name
			parts := strings.Split(dep, ".")
			depName := parts[len(parts)-1]
			deps = append(deps, depName)
		}
		sort.Strings(deps)

		var cols []string
		for cName := range node.Columns {
			cols = append(cols, cName)
		}
		sort.Strings(cols)

		receipt.Models[node.Name] = DBTModel{
			Name:        node.Name,
			Package:     node.PackageName,
			Description: node.Description,
			DependsOn:   deps,
			Columns:     cols,
			Tags:        node.Tags,
		}

		for _, upstream := range deps {
			receipt.LineageDown[upstream] = append(receipt.LineageDown[upstream], node.Name)
		}
	}

	// Sort downstream lists
	for k := range receipt.LineageDown {
		sort.Strings(receipt.LineageDown[k])
	}

	receipt.ModelCount = len(receipt.Models)
	return receipt, nil
}

// Lineage returns upstream dependencies and downstream dependents for modelName.
func (r *DBTSemanticsReceipt) Lineage(modelName string) (upstream []string, downstream []string, ok bool) {
	if r == nil || r.Models == nil {
		return nil, nil, false
	}
	model, exists := r.Models[modelName]
	if !exists {
		return nil, nil, false
	}
	return model.DependsOn, r.LineageDown[modelName], true
}

// QueryModelColumns returns the column schema for modelName from the compiled semantic manifest.
func (r *DBTSemanticsReceipt) QueryModelColumns(modelName string) ([]string, error) {
	if r == nil || r.Models == nil {
		return nil, errors.New("dataslot: dbt semantics receipt is nil")
	}
	model, exists := r.Models[modelName]
	if !exists {
		return nil, fmt.Errorf("dataslot: model %q not found in dbt semantics", modelName)
	}
	return model.Columns, nil
}
