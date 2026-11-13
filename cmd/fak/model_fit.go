package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

const modelFitInputLimit = 4 << 20

var errModelFitMissingAccounting = errors.New("missing_accounting_field")

type modelFitArtifact struct {
	Schema        string                    `json:"schema"`
	ModeledOnly   bool                      `json:"modeled_only"`
	Qualification string                    `json:"qualification"`
	Reports       []fitplan.PlacementReport `json:"reports"`
}

func runModelFit(stdin io.Reader, stdout, stderr io.Writer, args []string) int {
	usage := func() {
		fmt.Fprintln(stderr, "usage: fak model fit compose|plan|check|compare|bands --input FILE|- [--json]\nDeclared memory accounting only; no model loading or hardware qualification. See docs/model-fit.md.")
	}
	if len(args) == 0 {
		usage()
		return 2
	}
	if args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
		usage()
		return 0
	}
	mode := args[0]
	if mode != "compose" && mode != "plan" && mode != "check" && mode != "compare" && mode != "bands" {
		usage()
		return 2
	}
	fs := flag.NewFlagSet("model fit "+mode, flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = usage
	input := fs.String("input", "", "JSON declaration file; - reads stdin")
	manifest := fs.String("manifest", "", "authenticated GGUF shard manifest JSON (bands only)")
	manifestSHA := fs.String("manifest-sha256", "", "trusted SHA-256 of canonical shard manifest (bands only)")
	fs.Bool("json", true, "emit a versioned JSON artifact (always enabled)")
	if rc, ok := parseFlagsOrHelp(fs, args[1:]); !ok {
		return rc
	}
	if *input == "" || fs.NArg() != 0 || (mode != "bands" && (*manifest != "" || *manifestSHA != "")) {
		usage()
		return 2
	}
	reader := stdin
	if *input != "-" {
		f, err := os.Open(*input)
		if err != nil {
			fmt.Fprintln(stderr, "fak model fit: input unavailable")
			return 2
		}
		defer f.Close()
		reader = f
	}
	raw, err := io.ReadAll(io.LimitReader(reader, modelFitInputLimit+1))
	if err != nil || len(raw) > modelFitInputLimit {
		fmt.Fprintln(stderr, "fak model fit: input unreadable or exceeds 4 MiB")
		return 2
	}
	var reports []fitplan.PlacementReport
	if mode == "bands" {
		var in fitplan.DeploymentInput
		if err := decodeModelFitInput(raw, &in); err != nil || *manifest == "" || *manifestSHA == "" {
			fmt.Fprintln(stderr, "fak model fit bands: invalid declaration or missing authenticated manifest")
			return 2
		}
		f, err := os.Open(*manifest)
		if err != nil {
			fmt.Fprintln(stderr, "fak model fit bands: manifest unavailable")
			return 2
		}
		data, err := io.ReadAll(io.LimitReader(f, modelFitInputLimit+1))
		closeErr := f.Close()
		if err != nil || closeErr != nil || len(data) > modelFitInputLimit {
			fmt.Fprintln(stderr, "fak model fit bands: manifest unreadable")
			return 2
		}
		var shards []ggufload.ShardIdentity
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&shards); err != nil {
			fmt.Fprintln(stderr, "fak model fit bands: invalid manifest")
			return 2
		}
		var extra any
		if dec.Decode(&extra) != io.EOF {
			fmt.Fprintln(stderr, "fak model fit bands: invalid manifest")
			return 2
		}
		ws, err := ggufload.OpenWeightsAuthenticated(context.Background(), shards, *manifestSHA)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		bands, err := fitLayerBands(ws, in)
		closeErr = ws.Close()
		if err != nil || closeErr != nil {
			fmt.Fprintln(stderr, errors.Join(err, closeErr))
			return 1
		}
		artifact := struct {
			Schema         string                  `json:"schema"`
			ModeledOnly    bool                    `json:"modeled_only"`
			Qualification  string                  `json:"qualification"`
			ManifestSHA256 string                  `json:"manifest_sha256"`
			Declaration    fitplan.DeploymentInput `json:"declaration"`
			LoaderProfile  string                  `json:"loader_profile"`
			Bands          []fitBand               `json:"bands"`
		}{"fak-model-fit-bands/1", true, "unverified", *manifestSHA, in, "q4k-dense-kquant-no-native-rows-no-mtp-no-packed-embedding/1", bands}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(artifact); err != nil {
			fmt.Fprintln(stderr, "fak model fit bands: output failed")
			return 1
		}
		return 0
	}
	if mode == "compose" {
		var in fitplan.DeploymentInput
		if err = decodeModelFitInput(raw, &in); err == nil {
			in, err = fitplan.ComposePlacement(in)
		}
		if err != nil {
			if errors.Is(err, fitplan.ErrPlacementCapacityUnknown) {
				fmt.Fprintln(stderr, "placement_capacity_unknown")
				return 1
			}
			if errors.Is(err, fitplan.ErrPlacementInsufficientCapacity) {
				fmt.Fprintln(stderr, "placement_insufficient_capacity")
				return 1
			}
			if errors.Is(err, fitplan.ErrInvalidPlacement) || errors.Is(err, errModelFitMissingAccounting) {
				fmt.Fprintln(stderr, err)
			} else {
				fmt.Fprintln(stderr, "fak model fit: invalid or incomplete JSON declaration")
			}
			return 2
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(in); err != nil {
			fmt.Fprintln(stderr, "fak model fit: output failed")
			return 1
		}
		return 0
	}
	if mode == "compare" {
		var in fitplan.DeploymentComparisonInput
		if err = decodeModelFitInput(raw, &in); err == nil {
			reports, err = fitplan.ComparePlacements(in)
		}
	} else {
		var in fitplan.DeploymentInput
		if err = decodeModelFitInput(raw, &in); err == nil {
			var report fitplan.PlacementReport
			report, err = fitplan.EvaluatePlacement(in)
			reports = []fitplan.PlacementReport{report}
		}
	}
	if err != nil {
		if errors.Is(err, fitplan.ErrInvalidPlacement) || errors.Is(err, errModelFitMissingAccounting) {
			fmt.Fprintln(stderr, err)
		} else {
			fmt.Fprintln(stderr, "fak model fit: invalid or incomplete JSON declaration")
		}
		return 2
	}
	artifact := modelFitArtifact{Schema: "fak-model-fit/1", ModeledOnly: true, Qualification: "unverified", Reports: reports}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(artifact); err != nil {
		fmt.Fprintln(stderr, "fak model fit: output failed")
		return 1
	}
	if mode == "check" && (len(reports) != 1 || !reports[0].Eligible) {
		return 1
	}
	return 0
}

func decodeModelFitInput(raw []byte, out any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return fmt.Errorf("expected one JSON declaration")
	}
	return requireModelFitAccounting(raw)
}

func requireModelFitAccounting(raw []byte) error {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return err
	}
	require := func(raw json.RawMessage, keys ...string) (map[string]json.RawMessage, error) {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(raw, &obj); err != nil {
			return nil, err
		}
		for _, key := range keys {
			if value, ok := obj[key]; !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return nil, fmt.Errorf("%w: %s", errModelFitMissingAccounting, key)
			}
		}
		return obj, nil
	}
	var pools []json.RawMessage
	if err := json.Unmarshal(root["pools"], &pools); err != nil {
		return err
	}
	for _, raw := range pools {
		if _, err := require(raw, "id", "usable_bytes"); err != nil {
			return err
		}
	}
	placements := []json.RawMessage{root["placement"]}
	if raw, ok := root["placements"]; ok {
		if err := json.Unmarshal(raw, &placements); err != nil {
			return err
		}
	}
	for _, raw := range placements {
		p, err := require(raw, "weights", "ranks")
		if err != nil {
			return err
		}
		if _, err := require(p["weights"], "replicated_bytes", "sharded_bytes", "shard_count"); err != nil {
			return err
		}
		var ranks []json.RawMessage
		if err := json.Unmarshal(p["ranks"], &ranks); err != nil {
			return err
		}
		for _, raw := range ranks {
			if _, err := require(raw, "rank", "pool_id", "shards", "kv_bytes", "workspace_bytes", "extra_bytes"); err != nil {
				return err
			}
		}
	}
	return nil
}
