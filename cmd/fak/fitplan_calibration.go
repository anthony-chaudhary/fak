// fitplan_calibration.go is the local fak.fitplan.calibration.v1 store behind `fak serve
// --plan-json`'s fit_plan: its file I/O, the hardware epoch a plan and a probe are keyed by, the
// receipt adapters, and `fak model-observe bandwidth calibrate`. internal/fitplan stays pure: it
// fits and resolves rows; this file reads and writes them.
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
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

// calibrationStoreEnv overrides the store path.
const calibrationStoreEnv = "FAK_FITPLAN_CALIBRATION"

var (
	errFitPlanReceiptUnsupported    = errors.New("fitplan calibration: receipt schema not ingestible")
	errFitPlanReceiptSimulated      = errors.New("fitplan calibration: simulated receipt")
	errFitPlanReceiptBelowDRAM      = errors.New("fitplan calibration: working set below the DRAM floor")
	errFitPlanReceiptNoMeasure      = errors.New("fitplan calibration: receipt carries no bandwidth measurement")
	errFitPlanReceiptPartialThreads = errors.New("fitplan calibration: probe ran on fewer threads than GOMAXPROCS")
	errFitPlanReceiptNoEpoch        = errors.New("fitplan calibration: receipt names no hardware epoch")
	errFitPlanReceiptForeignMachine = errors.New("fitplan calibration: receipt measured on another machine class")
)

// fitPlanCalibrationMinWorkingSet is the smallest host-roofline working set ingested: the
// collector's default, below which a copy measures cache, not memory.
var fitPlanCalibrationMinWorkingSet = modelperfobs.DefaultRooflineBenchmarkOptions().WorkingSetBytes

// fitPlanCalibrationPath is FAK_FITPLAN_CALIBRATION, else <UserConfigDir>/fak/fitplan/calibration.json.
func fitPlanCalibrationPath() string {
	if p := strings.TrimSpace(os.Getenv(calibrationStoreEnv)); p != "" {
		return p
	}
	if dir, err := os.UserConfigDir(); err == nil && dir != "" {
		return filepath.Join(dir, "fak", "fitplan", "calibration.json")
	}
	return filepath.Join(".fak", "fitplan", "calibration.json")
}

// loadFitPlanCalibration reads the store; a missing file is an empty store.
func loadFitPlanCalibration(path string) (fitplan.CalibrationStore, error) {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return fitplan.NewCalibrationStore(), nil
	}
	if err != nil {
		return fitplan.CalibrationStore{}, err
	}
	return fitplan.DecodeCalibrationStore(raw)
}

func saveFitPlanCalibration(path string, s fitplan.CalibrationStore) error {
	raw, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return writeFitPlanFileAtomic(path, append(raw, '\n'))
}

func writeFitPlanFileAtomic(path string, raw []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		os.Remove(name)
		return err
	}
	return nil
}

// fitPlanHardwareEpoch fingerprints the hardware a plan prices and a probe measures: the host
// CPU, thread count and RAM for a host serve, plus the device backend, name and capacity for a
// device serve. It carries no engine build: a probe measures hardware, and a decode row names
// its engine in the key's backend.
func fitPlanHardwareEpoch(be compute.Backend) compute.HardwareEpochFingerprint {
	hostTotal, _, _ := compute.HostSystemMemoryInfo()
	fp := compute.HardwareEpochFingerprint{
		BackendName:    "cpu",
		RuntimeVersion: runtime.GOOS + "/" + runtime.GOARCH,
		DeviceName:     fmt.Sprintf("%s threads=%d ram_gib=%d", hostCPUModel(), runtime.NumCPU(), max(hostTotal, 0)>>30),
	}
	if be != nil {
		devTotal, _, _ := compute.DeviceMemoryInfo(be)
		fp.BackendName = be.Name()
		fp.VisibleDevices = fmt.Sprintf("%s mem_gib=%d", be.Tier(), max(devTotal, 0)>>30)
	}
	return fp
}

// hostCPUModel names the CPU without spawning a process: /proc/cpuinfo's model name on Linux,
// PROCESSOR_IDENTIFIER on Windows, else GOARCH.
func hostCPUModel() string {
	if v := strings.TrimSpace(os.Getenv("PROCESSOR_IDENTIFIER")); runtime.GOOS == "windows" && v != "" {
		return v
	}
	if runtime.GOOS == "linux" {
		if f, err := os.Open("/proc/cpuinfo"); err == nil {
			defer f.Close()
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				if k, v, ok := strings.Cut(sc.Text(), ":"); ok && strings.TrimSpace(k) == "model name" {
					return strings.TrimSpace(v)
				}
			}
		}
	}
	return runtime.GOARCH
}

// fitPlanEngineBackend is a fak decode row's backend key; fitPlanProbeBackend the device a probe
// row for the same hardware is keyed by.
func fitPlanEngineBackend(be compute.Backend) string { return "fak-" + fitPlanProbeBackend(be) }

func fitPlanProbeBackend(be compute.Backend) string {
	if be == nil {
		return "cpu"
	}
	return strings.ToLower(be.Name())
}

// resolveServeCalibration is the calibration a plan prices with: the store's row for this
// epoch, engine, quant family and arm, else the generic EMPIRICAL default. An unreadable store
// is named in the source, never fatal: the plan still prices at the generic tier.
func resolveServeCalibration(be compute.Backend, m fitplan.Model, arm string) fitplan.Calibration {
	q := fitplan.CalibrationQuery{
		Key: fitplan.CalibrationKey{
			Epoch:       fitPlanHardwareEpoch(be).Digest(),
			Backend:     fitPlanEngineBackend(be),
			QuantFamily: m.QuantFamily,
			KernelPath:  arm,
			BatchBucket: fitplan.BatchBucket(fitplan.DefaultWorkload().Concurrency),
		},
		ProbeBackend: fitPlanProbeBackend(be),
	}
	path := fitPlanCalibrationPath()
	store, err := loadFitPlanCalibration(path)
	if err != nil {
		store = fitplan.NewCalibrationStore()
	}
	cal, ok := store.Resolve(q)
	if ok {
		return cal
	}
	generic := fitplan.GenericEmpirical()
	generic.Lookup = cal.Lookup
	if err != nil {
		generic.Source += "; calibration store " + path + " unreadable: " + err.Error()
	}
	return generic
}

// fitPlanReceiptsFromFile adapts one receipt file to calibration receipts. A host-memory
// measurement names no hardware of its own, so it is stamped with this host's epoch only when the
// operator states it was measured here (measuredHere); a keyed receipts file keeps its epochs.
func fitPlanReceiptsFromFile(path string, raw []byte, measuredAt time.Time, measuredHere bool) ([]fitplan.CalibrationReceipt, error) {
	var head struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("%w: %s: %v", errFitPlanReceiptUnsupported, path, err)
	}
	sum := sha256.Sum256(raw)
	prov := fitplan.Provenance{Path: path, SHA256: hex.EncodeToString(sum[:])}
	at := measuredAt.UTC().Format(time.RFC3339)
	hostMeasure := head.Schema == modelperfobs.RooflineMeasurementSchema || head.Schema == modelperfobs.RooflineSweepMeasurementSchema ||
		head.Schema == modelperfobs.BandwidthCollectionSchema
	if hostMeasure && !measuredHere {
		return nil, fmt.Errorf("%w: %s is a host-memory measurement with no hardware identity; pass --measured-here if this host took it", errFitPlanReceiptNoEpoch, path)
	}
	switch head.Schema {
	case fitplan.CalibrationReceiptsSchema:
		var doc fitplan.CalibrationReceipts
		if err := json.Unmarshal(raw, &doc); err != nil {
			return nil, fmt.Errorf("%w: %s: %v", errFitPlanReceiptUnsupported, path, err)
		}
		for i := range doc.Receipts {
			r := &doc.Receipts[i]
			if r.Provenance.SHA256 == "" {
				r.Provenance = prov
			}
			if r.MeasuredAt == "" {
				r.MeasuredAt = at
			}
			if err := r.Validate(); err != nil {
				return nil, fmt.Errorf("%s receipt %d: %w", path, i, err)
			}
		}
		return doc.Receipts, nil
	case modelperfobs.RooflineMeasurementSchema, modelperfobs.RooflineSweepMeasurementSchema:
		var m modelperfobs.RooflineMeasurement
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, fmt.Errorf("%w: %s: %v", errFitPlanReceiptUnsupported, path, err)
		}
		r, err := fitPlanHostRooflineReceipt(m, prov, at)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		return []fitplan.CalibrationReceipt{r}, nil
	case modelperfobs.BandwidthCollectionSchema:
		var c modelperfobs.BandwidthCollection
		if err := json.Unmarshal(raw, &c); err != nil {
			return nil, fmt.Errorf("%w: %s: %v", errFitPlanReceiptUnsupported, path, err)
		}
		if c.RooflineMeasurement == nil {
			return nil, fmt.Errorf("%w: %s has no roofline_measurement (collect with --measure-host-roofline)", errFitPlanReceiptNoMeasure, path)
		}
		r, err := fitPlanHostRooflineReceipt(*c.RooflineMeasurement, prov, at)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		return []fitplan.CalibrationReceipt{r}, nil
	case "fak.roofline.empirical/v1":
		var e struct {
			Simulated bool `json:"simulated"`
		}
		_ = json.Unmarshal(raw, &e)
		if e.Simulated {
			return nil, fmt.Errorf("%w: %s states simulated:true; only measured bandwidth calibrates", errFitPlanReceiptSimulated, path)
		}
		return nil, fmt.Errorf("%w: %s is a device roofline; device probes carry no device epoch yet", errFitPlanReceiptUnsupported, path)
	default:
		return nil, fmt.Errorf("%w: %s schema %q", errFitPlanReceiptUnsupported, path, head.Schema)
	}
}

func fitPlanHostRooflineReceipt(m modelperfobs.RooflineMeasurement, prov fitplan.Provenance, at string) (fitplan.CalibrationReceipt, error) {
	machine := runtime.GOOS + "/" + runtime.GOARCH
	switch {
	case m.Scope != "host-memory" || !(m.MeasuredSustainableGBS > 0):
		return fitplan.CalibrationReceipt{}, fmt.Errorf("%w: no host-memory measured_sustainable_gb_s", errFitPlanReceiptNoMeasure)
	case m.MachineClass != "" && m.MachineClass != machine:
		return fitplan.CalibrationReceipt{}, fmt.Errorf("%w: machine_class %s, this host %s", errFitPlanReceiptForeignMachine, m.MachineClass, machine)
	case m.WorkingSetBytes < fitPlanCalibrationMinWorkingSet:
		return fitplan.CalibrationReceipt{}, fmt.Errorf("%w: working_set_bytes %d < %d", errFitPlanReceiptBelowDRAM, m.WorkingSetBytes, fitPlanCalibrationMinWorkingSet)
	case m.Threads < runtime.GOMAXPROCS(0):
		// Decode streams on every core; a partial-thread copy under-reads the bandwidth it can reach.
		return fitplan.CalibrationReceipt{}, fmt.Errorf("%w: threads %d < %d", errFitPlanReceiptPartialThreads, m.Threads, runtime.GOMAXPROCS(0))
	}
	r := fitplan.CalibrationReceipt{
		Key:  fitplan.CalibrationKey{Epoch: fitPlanHardwareEpoch(nil).Digest(), Backend: fitPlanProbeBackend(nil)},
		Kind: fitplan.ReceiptBandwidth,
		// The copy counts read+write traffic and does not prove DRAM isolation; the label says so.
		Label:               fmt.Sprintf("host-memory %s threads=%d working_set=%d %s", m.Method, m.Threads, m.WorkingSetBytes, m.TrafficAccounting),
		AchievedBytesPerSec: m.MeasuredSustainableGBS * 1e9,
		MeasuredAt:          at,
		Provenance:          prov,
	}
	return r, r.Validate()
}

// ingestFitPlanCalibration adds receipts to the store at path and saves it.
func ingestFitPlanCalibration(path string, receipts []fitplan.CalibrationReceipt) (fitplan.CalibrationStore, int, error) {
	store, err := loadFitPlanCalibration(path)
	if err != nil {
		return fitplan.CalibrationStore{}, 0, err
	}
	added := store.Add(receipts...)
	if err := saveFitPlanCalibration(path, store); err != nil {
		return fitplan.CalibrationStore{}, 0, err
	}
	return store, added, nil
}

// fitPlanCalibrationReport is what `fak model-observe bandwidth calibrate` prints.
type fitPlanCalibrationReport struct {
	Schema string                   `json:"schema"`
	Store  string                   `json:"store"`
	Epoch  string                   `json:"host_epoch"`
	Added  int                      `json:"added"`
	Rows   []fitplan.CalibrationRow `json:"rows"`
}

// runModelObserveBandwidthCalibrate ingests receipt files into the local calibration store, all
// or none, and prints its rows; with no files it prints the store as it is.
func runModelObserveBandwidthCalibrate(args []string) error {
	return modelObserveBandwidthCalibrate(args, os.Stdout)
}

func modelObserveBandwidthCalibrate(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("model-observe bandwidth calibrate", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	store := fs.String("store", fitPlanCalibrationPath(), "calibration store path (env "+calibrationStoreEnv+")")
	measuredHere := fs.Bool("measured-here", false, "host-memory roofline files were measured on this host; stamp them with its hardware epoch")
	if err := fs.Parse(args); err != nil {
		return err
	}
	var receipts []fitplan.CalibrationReceipt
	for _, path := range fs.Args() {
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		measuredAt := time.Now()
		if st, err := os.Stat(path); err == nil {
			measuredAt = st.ModTime()
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			abs = path
		}
		rs, err := fitPlanReceiptsFromFile(abs, raw, measuredAt, *measuredHere)
		if err != nil {
			return err
		}
		receipts = append(receipts, rs...)
	}
	var s fitplan.CalibrationStore
	added := 0
	var err error
	if len(receipts) > 0 {
		s, added, err = ingestFitPlanCalibration(*store, receipts)
	} else {
		s, err = loadFitPlanCalibration(*store)
	}
	if err != nil {
		return err
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(fitPlanCalibrationReport{
		Schema: fitplan.CalibrationSchema,
		Store:  *store,
		Epoch:  fitPlanHardwareEpoch(nil).Digest(),
		Added:  added,
		Rows:   s.Rows,
	})
}

// ingestHostRooflineCalibration is collect --calibrate: the emitted collection bytes are the
// receipt. Written to --output, that file is the provenance; on stdout, a copy is kept beside
// the store.
func ingestHostRooflineCalibration(raw []byte, output string, m modelperfobs.RooflineMeasurement) error {
	path := fitPlanCalibrationPath()
	sum := sha256.Sum256(raw)
	digest := hex.EncodeToString(sum[:])
	receiptPath := output
	if receiptPath == "" {
		receiptPath = filepath.Join(filepath.Dir(path), "receipts", digest[:16]+".json")
		if err := writeFitPlanFileAtomic(receiptPath, raw); err != nil {
			return err
		}
	}
	if abs, err := filepath.Abs(receiptPath); err == nil {
		receiptPath = abs
	}
	r, err := fitPlanHostRooflineReceipt(m, fitplan.Provenance{Path: receiptPath, SHA256: digest}, time.Now().UTC().Format(time.RFC3339))
	if err != nil {
		return err
	}
	_, added, err := ingestFitPlanCalibration(path, []fitplan.CalibrationReceipt{r})
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "model-observe: calibration store %s: %d receipt added (%.1f GB/s host-memory, epoch %s)\n", path, added, m.MeasuredSustainableGBS, r.Key.Epoch[:12])
	return nil
}

// encodeModelObserveJSON is the bytes writeModelObserveJSON would write.
func encodeModelObserveJSON(pretty bool, value any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	if pretty {
		enc.SetIndent("", "  ")
	}
	if err := enc.Encode(value); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
