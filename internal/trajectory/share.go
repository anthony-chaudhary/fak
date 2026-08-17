package trajectory

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

const (
	ShareManifestSchema = "fak-trajectory-share/1alpha1"
	ShareFormatJSONL    = "canonical-jsonl/1"
)

type ShareManifest struct {
	Schema           string `json:"schema"`
	Audience         string `json:"audience"`
	TrajectoryDigest string `json:"trajectory_digest"`
	ViewSpecDigest   string `json:"view_spec_digest"`
	ProjectionDigest string `json:"projection_digest"`
	RedactionProfile string `json:"redaction_profile"`
	Format           string `json:"format"`
	PayloadDigest    string `json:"encoded_bytes_digest"`
	ContentAddress   string `json:"content_address"`
	EventCount       int    `json:"event_count"`
}

type ShareBundle struct {
	Manifest ShareManifest `json:"manifest"`
	JSONL    []byte        `json:"jsonl"`
}

// ExportView packages only already-projected events and their matching transform receipt.
func ExportView(projected []ProjectedEvent, receipt ViewReceipt) (ShareBundle, error) {
	if err := receipt.Validate(); err != nil {
		return ShareBundle{}, err
	}
	projectionBytes, err := json.Marshal(projected)
	if err != nil {
		return ShareBundle{}, err
	}
	if viewDigest(projectionBytes) != receipt.ProjectionDigest {
		return ShareBundle{}, errors.New("projected events do not match view receipt")
	}
	if len(projected) != receipt.OutputEvents {
		return ShareBundle{}, errors.New("projected event count does not match view receipt")
	}

	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	encoder.SetEscapeHTML(false)
	for i, event := range projected {
		if err := event.Event.Validate(); err != nil {
			return ShareBundle{}, fmt.Errorf("projected event %d: %w", i, err)
		}
		if err := encoder.Encode(event); err != nil {
			return ShareBundle{}, err
		}
	}
	encoded := output.Bytes()
	manifest := ShareManifest{
		Schema: ShareManifestSchema, Audience: receipt.Audience, TrajectoryDigest: receipt.TrajectoryDigest,
		ViewSpecDigest: receipt.ViewSpecDigest, ProjectionDigest: receipt.ProjectionDigest, RedactionProfile: receipt.RedactionProfile,
		Format: ShareFormatJSONL, PayloadDigest: digestBytes(encoded), EventCount: len(projected),
	}
	addressSeed, _ := json.Marshal(manifest)
	manifest.ContentAddress = digestBytes(addressSeed)
	bundle := ShareBundle{Manifest: manifest, JSONL: append([]byte(nil), encoded...)}
	if err := bundle.Verify(); err != nil {
		return ShareBundle{}, err
	}
	return bundle, nil
}

func (bundle ShareBundle) Verify() error {
	manifest := bundle.Manifest
	if manifest.Schema != ShareManifestSchema || manifest.Format != ShareFormatJSONL || manifest.Audience == "" || manifest.TrajectoryDigest == "" || manifest.ViewSpecDigest == "" || manifest.ProjectionDigest == "" || manifest.RedactionProfile == "" {
		return errors.New("incomplete trajectory share manifest")
	}
	if digestBytes(bundle.JSONL) != manifest.PayloadDigest {
		return errors.New("trajectory share bytes digest mismatch")
	}
	address := manifest.ContentAddress
	manifest.ContentAddress = ""
	addressSeed, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	if digestBytes(addressSeed) != address {
		return errors.New("trajectory share content address mismatch")
	}
	decoder := json.NewDecoder(bytes.NewReader(bundle.JSONL))
	count := 0
	for decoder.More() {
		var projected ProjectedEvent
		if err := decoder.Decode(&projected); err != nil {
			return fmt.Errorf("decode shared event %d: %w", count, err)
		}
		if err := projected.Event.Validate(); err != nil {
			return fmt.Errorf("shared event %d: %w", count, err)
		}
		count++
	}
	if count != manifest.EventCount {
		return errors.New("trajectory share event count mismatch")
	}
	return nil
}
