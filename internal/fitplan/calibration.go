package fitplan

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// CalibrationSchema stamps the local calibration store document.
const CalibrationSchema = "fak.fitplan.calibration.v1"

// CalibrationReceiptsSchema stamps a file of keyed decode or bandwidth receipts to ingest.
const CalibrationReceiptsSchema = "fak.fitplan.calibration-receipts.v1"

// CalibrationReceipts is a receipts file: what a non-fak producer (another engine's bench) emits.
type CalibrationReceipts struct {
	Schema   string               `json:"schema"`
	Receipts []CalibrationReceipt `json:"receipts"`
}

const (
	// DriftThreshold is the |t_pred - t_meas| / t_meas above which a receipt disagrees with
	// its prediction and the row stops ranking.
	DriftThreshold = 0.15
	// LineMinSpread is the bytes max/min a least-squares t = a*bytes + c needs; below it the
	// intercept is not identifiable and the receipts pool into one point.
	LineMinSpread = 3.0
)

// ReceiptKind names what a calibration receipt measured.
type ReceiptKind string

const (
	ReceiptDecode    ReceiptKind = "decode"    // a decode step: bytes streamed and step seconds
	ReceiptBandwidth ReceiptKind = "bandwidth" // an achieved memory-bandwidth probe
	// ReceiptConfirm names the post-load two-phase fit confirmation (FitConfirmRecord in
	// confirm.go): planned-vs-allocated bytes per class plus the unaccounted residual.
	// It is a RECORD kind, not a fitter input: a confirm record never flows into
	// FitCalibration, so CalibrationReceipt.Validate keeps refusing it by design.
	ReceiptConfirm ReceiptKind = "fit_confirm"
)

// FitKind names how a row's numbers were derived from its receipts.
type FitKind string

const (
	FitOnePoint          FitKind = "one_point"          // BW = bytes/t; t_ovh folded into BW
	FitLeastSquares      FitKind = "least_squares"      // t = a*bytes + c; BW = 1/a, t_ovh = c
	FitThroughOrigin     FitKind = "through_origin"     // least squares gave c < 0: t = a*bytes, t_ovh = 0
	FitAchievedBandwidth FitKind = "achieved_bandwidth" // median probe reading; t_ovh unknown, 0
)

// RowState is whether a row may rank.
type RowState string

const (
	RowFit   RowState = "fit"
	RowDrift RowState = "drift" // a receipt disagrees by > DriftThreshold; demoted to EMPIRICAL, excluded from ranking
)

// DriftReason is the closed vocabulary of why a row drifted.
type DriftReason string

const (
	DriftOwnFit      DriftReason = "residual_vs_own_fit"      // a receipt off the row's own fit
	DriftSiblingLine DriftReason = "residual_vs_sibling_line" // a zero-dof row off its backend's line
	DriftProbeSpread DriftReason = "probe_spread"             // bandwidth probes on one epoch disagree
)

var (
	ErrCalibrationDrift          = errors.New("fitplan: calibration_drift")
	ErrCalibrationSchema         = errors.New("fitplan: calibration_schema")
	ErrCalibrationReceiptInvalid = errors.New("fitplan: calibration_receipt_invalid")
)

// DriftError is a drifted row's typed verdict; errors.Is matches ErrCalibrationDrift.
type DriftError struct {
	Key      CalibrationKey
	Reason   DriftReason
	Residual float64
}

func (e *DriftError) Error() string {
	return fmt.Sprintf("fitplan: calibration_drift %s (%+.1f%%)", e.Reason, 100*e.Residual)
}
func (e *DriftError) Unwrap() error { return ErrCalibrationDrift }

// CalibrationKey is one row's identity: hardware epoch, engine backend, weight quant
// family, kernel path and batch bucket. A bandwidth probe row sets Epoch and Backend only.
type CalibrationKey struct {
	Epoch       string `json:"epoch"`
	Backend     string `json:"backend,omitempty"`
	QuantFamily string `json:"quant_family,omitempty"`
	KernelPath  string `json:"kernel_path,omitempty"`
	BatchBucket string `json:"batch_bucket,omitempty"`
}

func (k CalibrationKey) normalized() CalibrationKey {
	norm := func(s string) string { return strings.ToLower(strings.TrimSpace(s)) }
	return CalibrationKey{
		Epoch:       strings.TrimSpace(k.Epoch),
		Backend:     norm(k.Backend),
		QuantFamily: norm(k.QuantFamily),
		KernelPath:  norm(k.KernelPath),
		BatchBucket: norm(k.BatchBucket),
	}
}

// lineKey is the key a backend's least-squares line is shared under: every field but the family.
func (k CalibrationKey) lineKey() CalibrationKey {
	k.QuantFamily = ""
	return k
}

// String renders the key as its non-empty fields, for a provenance line.
func (k CalibrationKey) String() string {
	parts := []string{"epoch=" + k.Epoch}
	for _, f := range [][2]string{{"backend", k.Backend}, {"quant", k.QuantFamily}, {"path", k.KernelPath}, {"b", k.BatchBucket}} {
		if f[1] != "" {
			parts = append(parts, f[0]+"="+f[1])
		}
	}
	return strings.Join(parts, " ")
}

// Provenance names the receipt file a number came from.
type Provenance struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// CalibrationReceipt is one measured observation the fitter consumes.
type CalibrationReceipt struct {
	Key   CalibrationKey `json:"key"`
	Kind  ReceiptKind    `json:"kind"`
	Label string         `json:"label,omitempty"` // arm or model the receipt measured
	// TierHint, when set, is the evidence tier the row fitted from this receipt adopts
	// instead of the default calibrated tier: a silicon-measured decode confirm marks its
	// row TierSilicon so the store can tell host-fitted rows apart from on-silicon ones.
	// Zero value keeps every pre-existing receipt byte-for-byte unchanged.
	TierHint Tier `json:"tier_hint,omitempty"`
	// Decode: bytes one step streams and the measured step seconds.
	BytesPerStep float64 `json:"bytes_per_step,omitempty"`
	StepSeconds  float64 `json:"step_s,omitempty"`
	// Bandwidth: the achieved reading.
	AchievedBytesPerSec float64    `json:"achieved_bytes_per_s,omitempty"`
	MeasuredAt          string     `json:"measured_at,omitempty"` // RFC3339; orders epochs for carry-over
	Provenance          Provenance `json:"provenance"`
}

// Validate reports a receipt the fitter cannot use; errors.Is matches ErrCalibrationReceiptInvalid.
func (r CalibrationReceipt) Validate() error {
	bad := func(why string) error { return fmt.Errorf("%w: %s", ErrCalibrationReceiptInvalid, why) }
	switch {
	case strings.TrimSpace(r.Key.Epoch) == "":
		return bad("no hardware epoch")
	case strings.TrimSpace(r.Provenance.SHA256) == "":
		return bad("no receipt sha256")
	}
	switch r.Kind {
	case ReceiptDecode:
		if !positiveFinite(r.BytesPerStep) || !positiveFinite(r.StepSeconds) {
			return bad("decode receipt needs bytes_per_step and step_s > 0")
		}
	case ReceiptBandwidth:
		if !positiveFinite(r.AchievedBytesPerSec) {
			return bad("bandwidth receipt needs achieved_bytes_per_s > 0")
		}
	default:
		return bad("unknown kind " + string(r.Kind))
	}
	return nil
}

// identity dedupes a re-ingested receipt. The measured values are part of it: one file may hold
// several unlabelled receipts on one key, and those are the points a line is fitted through.
func (r CalibrationReceipt) identity() string {
	num := func(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }
	return strings.Join([]string{r.Provenance.SHA256, string(r.Kind), r.Key.normalized().String(), r.Label,
		num(r.BytesPerStep), num(r.StepSeconds), num(r.AchievedBytesPerSec)}, "\x00")
}

// Residual is one receipt's disagreement with a prediction, in step time.
type Residual struct {
	Label     string  `json:"label,omitempty"`
	SHA256    string  `json:"sha256"`
	Predicted float64 `json:"predicted_step_s"`
	Measured  float64 `json:"measured_step_s"`
	Relative  float64 `json:"relative"`          // (predicted - measured) / measured
	Against   string  `json:"against,omitempty"` // "own_fit" or "line:<quant family>"
}

// CalibrationRow is one fitted store row.
type CalibrationRow struct {
	Key                 CalibrationKey `json:"key"`
	Tier                Tier           `json:"tier"`
	State               RowState       `json:"state"`
	Fit                 FitKind        `json:"fit"`
	BytesPerSec         float64        `json:"bytes_per_s"` // effective bandwidth the row predicts with
	OverheadSeconds     float64        `json:"overhead_s"`  // t_ovh per decode step
	Receipts            int            `json:"receipts"`    // receipts the fit consumed
	DriftReason         DriftReason    `json:"drift_reason,omitempty"`
	Residuals           []Residual     `json:"residuals,omitempty"`
	Provenance          []Provenance   `json:"provenance"`
	MeasuredAt          string         `json:"measured_at,omitempty"`    // latest receipt
	PriorEpoch          string         `json:"prior_epoch,omitempty"`    // set when carried from an older epoch
	InheritedFromFamily string         `json:"inherited_from,omitempty"` // set when a sibling family's line stands in
}

// Err is the row's typed drift verdict, nil for a row that may rank.
func (r CalibrationRow) Err() error {
	if r.State != RowDrift {
		return nil
	}
	worst := 0.0
	for _, res := range r.Residuals {
		if math.Abs(res.Relative) > math.Abs(worst) {
			worst = res.Relative
		}
	}
	return &DriftError{Key: r.Key, Reason: r.DriftReason, Residual: worst}
}

// predict is the row's step time for a decode streaming bytes.
func (r CalibrationRow) predict(bytes float64) float64 {
	if !positiveFinite(r.BytesPerSec) {
		return 0
	}
	return bytes/r.BytesPerSec + r.OverheadSeconds
}

// CalibrationStore is the persisted local store: the receipts and the rows fitted from them.
type CalibrationStore struct {
	Schema   string               `json:"schema"`
	Receipts []CalibrationReceipt `json:"receipts"`
	Rows     []CalibrationRow     `json:"rows"`
}

// NewCalibrationStore is an empty store.
func NewCalibrationStore() CalibrationStore {
	return CalibrationStore{Schema: CalibrationSchema, Receipts: []CalibrationReceipt{}, Rows: []CalibrationRow{}}
}

// DecodeCalibrationStore parses a store document; a foreign schema is ErrCalibrationSchema.
// Rows are refitted from the receipts, so a hand-edited row cannot outlive its evidence.
func DecodeCalibrationStore(b []byte) (CalibrationStore, error) {
	var s CalibrationStore
	if err := json.Unmarshal(b, &s); err != nil {
		return CalibrationStore{}, fmt.Errorf("%w: %v", ErrCalibrationSchema, err)
	}
	if s.Schema != CalibrationSchema {
		return CalibrationStore{}, fmt.Errorf("%w: schema %q, want %q", ErrCalibrationSchema, s.Schema, CalibrationSchema)
	}
	out := NewCalibrationStore()
	out.Add(s.Receipts...)
	return out, nil
}

// Add appends the valid receipts not already present and refits every row. It returns how many
// were added; an invalid receipt is skipped (Validate names why).
func (s *CalibrationStore) Add(receipts ...CalibrationReceipt) int {
	s.Schema = CalibrationSchema
	seen := make(map[string]bool, len(s.Receipts))
	for _, r := range s.Receipts {
		seen[r.identity()] = true
	}
	added := 0
	for _, r := range receipts {
		if r.Validate() != nil {
			continue
		}
		r.Key = r.Key.normalized()
		if r.Kind == ReceiptBandwidth {
			r.Key = CalibrationKey{Epoch: r.Key.Epoch, Backend: r.Key.Backend}
		}
		// MeasuredAt orders epochs by string; only a UTC RFC3339 stamp orders correctly.
		if at, err := time.Parse(time.RFC3339, r.MeasuredAt); err == nil {
			r.MeasuredAt = at.UTC().Format(time.RFC3339)
		} else {
			r.MeasuredAt = ""
		}
		if id := r.identity(); !seen[id] {
			seen[id] = true
			s.Receipts = append(s.Receipts, r)
			added++
		}
	}
	if s.Receipts == nil {
		s.Receipts = []CalibrationReceipt{}
	}
	s.Rows = FitCalibration(s.Receipts)
	return added
}

// FitCalibration fits one row per key: one decode receipt gives BW = bytes/t with the
// overhead folded in; two or more spanning >= LineMinSpread x bytes give least squares
// t = a*bytes + c; narrower spreads pool into one point. A row with no residual degrees of
// freedom is checked against its backend's line (a least-squares row of another quant family on
// the same epoch, backend, path and bucket); bandwidth probes take the median reading. Any
// residual above DriftThreshold sets the row to drift. Rows are sorted by key.
func FitCalibration(receipts []CalibrationReceipt) []CalibrationRow {
	groups := map[CalibrationKey][]CalibrationReceipt{}
	var keys []CalibrationKey
	for _, r := range receipts {
		if r.Validate() != nil {
			continue
		}
		k := r.Key.normalized()
		if r.Kind == ReceiptBandwidth {
			k = CalibrationKey{Epoch: k.Epoch, Backend: k.Backend}
		}
		gk := k
		if r.Kind == ReceiptBandwidth {
			gk.KernelPath = "\x00probe" // probes never pool with decode receipts
		}
		if _, ok := groups[gk]; !ok {
			keys = append(keys, gk)
		}
		groups[gk] = append(groups[gk], r)
	}
	rows := make([]CalibrationRow, 0, len(keys))
	for _, gk := range keys {
		rs := groups[gk]
		var row CalibrationRow
		if rs[0].Kind == ReceiptBandwidth {
			row = fitProbe(CalibrationKey{Epoch: gk.Epoch, Backend: gk.Backend}, rs)
		} else {
			row = fitDecode(gk, rs)
		}
		// A row the store could not save or a plan could not marshal is no row.
		if positiveFinite(row.BytesPerSec) && row.OverheadSeconds >= 0 && !math.IsInf(row.OverheadSeconds, 0) {
			rows = append(rows, row)
		}
	}
	checkSiblingLines(rows, groups)
	sort.SliceStable(rows, func(i, j int) bool { return rowLess(rows[i], rows[j]) })
	return rows
}

func rowLess(a, b CalibrationRow) bool {
	ka, kb := a.Key.String(), b.Key.String()
	if ka != kb {
		return ka < kb
	}
	return a.Fit < b.Fit
}

func newRow(k CalibrationKey, fit FitKind, rs []CalibrationReceipt) CalibrationRow {
	tier := TierCalibrated
	for _, r := range rs {
		if r.TierHint != "" {
			tier = r.TierHint
		}
	}
	row := CalibrationRow{Key: k, Tier: tier, State: RowFit, Fit: fit, Receipts: len(rs)}
	seen := map[Provenance]bool{}
	for _, r := range rs {
		if !seen[r.Provenance] {
			seen[r.Provenance] = true
			row.Provenance = append(row.Provenance, r.Provenance)
		}
		if r.MeasuredAt > row.MeasuredAt {
			row.MeasuredAt = r.MeasuredAt
		}
	}
	sort.Slice(row.Provenance, func(i, j int) bool { return row.Provenance[i].Path < row.Provenance[j].Path })
	return row
}

// fitProbe takes the median probe reading. A probe measures the hardware, not the engine, so its
// row is EMPIRICAL: an achieved-bandwidth upper bound with no engine efficiency or overhead fitted.
func fitProbe(k CalibrationKey, rs []CalibrationReceipt) CalibrationRow {
	row := newRow(k, FitAchievedBandwidth, rs)
	row.Tier = TierEmpirical
	vals := make([]float64, len(rs))
	for i, r := range rs {
		vals[i] = r.AchievedBytesPerSec
	}
	sort.Float64s(vals)
	med := vals[len(vals)/2]
	if len(vals)%2 == 0 {
		med = (vals[len(vals)/2-1] + vals[len(vals)/2]) / 2
	}
	row.BytesPerSec = med
	for _, r := range rs {
		// A probe's residual is the median's time per byte against the reading's.
		row.Residuals = append(row.Residuals, Residual{Label: r.Label, SHA256: r.Provenance.SHA256,
			Predicted: 1 / med, Measured: 1 / r.AchievedBytesPerSec, Relative: r.AchievedBytesPerSec/med - 1, Against: "own_fit"})
	}
	// Any two readings, not each against the median: the median hides a pair up to ~35% apart.
	if vals[len(vals)-1]/vals[0]-1 > DriftThreshold {
		row.State, row.DriftReason = RowDrift, DriftProbeSpread
	}
	return row
}

func fitDecode(k CalibrationKey, rs []CalibrationReceipt) CalibrationRow {
	lo, hi := math.Inf(1), 0.0
	for _, r := range rs {
		lo, hi = math.Min(lo, r.BytesPerStep), math.Max(hi, r.BytesPerStep)
	}
	var row CalibrationRow
	if a, c, origin, ok := leastSquares(rs); len(rs) >= 2 && hi >= LineMinSpread*lo && ok {
		fit := FitLeastSquares
		if origin {
			fit = FitThroughOrigin
		}
		row = newRow(k, fit, rs)
		row.BytesPerSec, row.OverheadSeconds = 1/a, c
	} else {
		var b, t float64
		for _, r := range rs {
			b, t = b+r.BytesPerStep, t+r.StepSeconds
		}
		row = newRow(k, FitOnePoint, rs)
		row.BytesPerSec = b / t
	}
	if row.dof() > 0 {
		for _, r := range rs {
			row.addResidual(r, row, "own_fit", DriftOwnFit)
		}
	}
	return row
}

// dof is the residual degrees of freedom: receipts beyond the parameters the fit spent.
func (r CalibrationRow) dof() int {
	switch r.Fit {
	case FitLeastSquares:
		return r.Receipts - 2
	case FitOnePoint, FitThroughOrigin:
		return r.Receipts - 1
	default:
		return 0
	}
}

func (r *CalibrationRow) addResidual(rc CalibrationReceipt, model CalibrationRow, against string, reason DriftReason) {
	pred := model.predict(rc.BytesPerStep)
	rel := (pred - rc.StepSeconds) / rc.StepSeconds
	r.Residuals = append(r.Residuals, Residual{Label: rc.Label, SHA256: rc.Provenance.SHA256,
		Predicted: pred, Measured: rc.StepSeconds, Relative: rel, Against: against})
	if math.Abs(rel) > DriftThreshold && r.State != RowDrift {
		r.State, r.DriftReason, r.Tier = RowDrift, reason, TierEmpirical
	}
}

// leastSquares fits t = a*bytes + c on centred data. It refuses a non-physical fit (a <= 0), and a
// negative intercept refits through the origin (origin true: one parameter spent, not two), since
// overhead cannot be negative.
func leastSquares(rs []CalibrationReceipt) (a, c float64, origin, ok bool) {
	n := float64(len(rs))
	if n < 2 {
		return 0, 0, false, false
	}
	var mb, mt float64
	for _, r := range rs {
		mb, mt = mb+r.BytesPerStep, mt+r.StepSeconds
	}
	mb, mt = mb/n, mt/n
	var sxy, sxx float64
	for _, r := range rs {
		dx := r.BytesPerStep - mb
		sxy += dx * (r.StepSeconds - mt)
		sxx += dx * dx
	}
	if !(sxx > 0) {
		return 0, 0, false, false
	}
	a = sxy / sxx
	c = mt - a*mb
	if c < 0 {
		var bt, bb float64
		for _, r := range rs {
			bt += r.BytesPerStep * r.StepSeconds
			bb += r.BytesPerStep * r.BytesPerStep
		}
		a, c, origin = bt/bb, 0, true
	}
	if !positiveFinite(a) || !positiveFinite(1/a) || math.IsNaN(c) || math.IsInf(c, 0) {
		return 0, 0, false, false
	}
	return a, c, origin, true
}

// checkSiblingLines tests every zero-dof decode row against its backend's line, the least-squares
// row of another quant family on the same epoch, backend, path and bucket: a single receipt cannot
// test itself, but it can test whether the line transfers to its family.
func checkSiblingLines(rows []CalibrationRow, groups map[CalibrationKey][]CalibrationReceipt) {
	lines := map[CalibrationKey]int{}
	for i, r := range rows {
		if r.Fit != FitLeastSquares || r.State != RowFit {
			continue
		}
		lk := r.Key.lineKey()
		if j, ok := lines[lk]; !ok || r.Receipts > rows[j].Receipts || (r.Receipts == rows[j].Receipts && r.Key.QuantFamily < rows[j].Key.QuantFamily) {
			lines[lk] = i
		}
	}
	for i := range rows {
		r := &rows[i]
		if r.Fit == FitAchievedBandwidth || r.dof() > 0 {
			continue
		}
		j, ok := lines[r.Key.lineKey()]
		if !ok || rows[j].Key.QuantFamily == r.Key.QuantFamily {
			continue
		}
		line := rows[j]
		for _, rc := range groups[r.Key] {
			r.addResidual(rc, line, "line:"+line.Key.QuantFamily, DriftSiblingLine)
		}
	}
}

// CalibrationQuery is what a plan resolves a calibration for.
type CalibrationQuery struct {
	Key          CalibrationKey // the engine row the plan would run on
	ProbeBackend string         // the device a bandwidth probe row for this epoch is keyed by
}

// Resolve picks the calibration a plan prices with, highest tier first:
//  1. the exact row on this epoch (a drifted row is stated with its drift verdict);
//  2. this epoch's backend line from another quant family (inherited);
//  3. this epoch's bandwidth probe for the device (achieved-bandwidth override);
//  4. the same rows from the latest prior epoch, demoted to an EMPIRICAL prior.
//
// ok is false when no row applies; the caller keeps its generic EMPIRICAL default. Compute
// efficiency is never fitted here, so it stays GenericEmpirical's, labelled with its own tier.
func (s CalibrationStore) Resolve(q CalibrationQuery) (Calibration, bool) {
	k := q.Key.normalized()
	cal, ok := s.resolve(k, strings.ToLower(strings.TrimSpace(q.ProbeBackend)))
	cal.Lookup = &k
	return cal, ok
}

func (s CalibrationStore) resolve(k CalibrationKey, probe string) (Calibration, bool) {
	if row, ok := s.resolveEpoch(k, probe); ok {
		return row.calibration(), true
	}
	var best *CalibrationRow
	for _, epoch := range s.epochs() {
		if epoch == k.Epoch {
			continue
		}
		pk := k
		pk.Epoch = epoch
		row, ok := s.resolveEpoch(pk, probe)
		if !ok || row.State != RowFit {
			continue
		}
		if best == nil || row.MeasuredAt > best.MeasuredAt {
			r := row
			best = &r
		}
	}
	if best == nil {
		return Calibration{}, false
	}
	best.PriorEpoch = best.Key.Epoch
	best.Tier = TierEmpirical
	return best.calibration(), true
}

func (s CalibrationStore) epochs() []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range s.Rows {
		if !seen[r.Key.Epoch] {
			seen[r.Key.Epoch] = true
			out = append(out, r.Key.Epoch)
		}
	}
	sort.Strings(out)
	return out
}

func (s CalibrationStore) resolveEpoch(k CalibrationKey, probe string) (CalibrationRow, bool) {
	var line *CalibrationRow
	for i := range s.Rows {
		r := s.Rows[i]
		if r.Fit == FitAchievedBandwidth {
			continue
		}
		if r.Key == k {
			return r, true
		}
		if r.Fit == FitLeastSquares && r.State == RowFit && r.Key.lineKey() == k.lineKey() &&
			(line == nil || r.Receipts > line.Receipts) {
			line = &s.Rows[i]
		}
	}
	if line != nil {
		r := *line
		r.InheritedFromFamily = r.Key.QuantFamily
		return r, true
	}
	if probe == "" {
		return CalibrationRow{}, false
	}
	for _, r := range s.Rows {
		if r.Fit == FitAchievedBandwidth && r.Key == (CalibrationKey{Epoch: k.Epoch, Backend: probe}) {
			return r, true
		}
	}
	return CalibrationRow{}, false
}

func (r CalibrationRow) calibration() Calibration {
	g := GenericEmpirical()
	src := fmt.Sprintf("%s %s fit over %d receipt(s), %s", r.Tier, r.Fit, r.Receipts, r.Key)
	switch {
	case r.PriorEpoch != "":
		src += "; prior epoch carried as EMPIRICAL prior"
	case r.InheritedFromFamily != "":
		src += "; inherited from the " + r.InheritedFromFamily + " line"
	}
	if r.Fit == FitAchievedBandwidth {
		src += "; achieved-bandwidth upper bound: a hardware probe overrides peak x efficiency, no engine efficiency or overhead fitted"
	}
	if r.State == RowDrift {
		src += "; drift (" + string(r.DriftReason) + "): stated, not fit to rank"
	}
	row := r
	return Calibration{
		Tier:                r.Tier,
		ComputeEfficiency:   g.ComputeEfficiency,
		Source:              src,
		AchievedBytesPerSec: r.BytesPerSec,
		OverheadSeconds:     r.OverheadSeconds,
		ComputeTier:         g.Tier,
		ComputeSource:       g.Source,
		State:               r.State,
		Row:                 &row,
	}
}

// BatchBucket is the batch-size bucket a row is keyed by: 1, 2, 3, 4, 5-8, 9-64, >64.
func BatchBucket(b int) string {
	switch {
	case b <= 1:
		return "1"
	case b <= 4:
		return fmt.Sprint(b)
	case b <= 8:
		return "5-8"
	case b <= 64:
		return "9-64"
	default:
		return ">64"
	}
}

// QuantFamily folds a GGUF tensor type name into the family a kernel's dequant cost tracks:
// "k" (Q2_K..Q8_K), "iq" (IQ*), "q8_0", "legacy" (Q4_0/Q4_1/Q5_0/Q5_1), "float" (F32/F16/BF16),
// else the lowercased name.
func QuantFamily(typeName string) string {
	t := strings.ToUpper(strings.TrimSpace(typeName))
	switch {
	case t == "":
		return ""
	case t == "F32" || t == "F16" || t == "BF16" || t == "F64":
		return "float"
	case t == "Q8_0":
		return "q8_0"
	case strings.HasPrefix(t, "IQ"):
		return "iq"
	case strings.HasPrefix(t, "Q") && strings.HasSuffix(t, "_K"):
		return "k"
	case t == "Q4_0" || t == "Q4_1" || t == "Q5_0" || t == "Q5_1" || t == "Q8_1":
		return "legacy"
	default:
		return strings.ToLower(t)
	}
}
