package modelengine

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/fabricmap"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var ErrNativePDReceiver = errors.New("modelengine: receiver-ready transfer refused")

// NativePDReceiverSnapshot binds the observed provider snapshot used for this
// admission; no timeless or generation-free topology is consumer-ready evidence.
type NativePDReceiverSnapshot struct {
	Provider   string
	Generation uint64
	ValidUntil time.Time
	SHA256     string
}
type NativePDReceiverReceipt struct {
	fabricmap.ExecutionReceipt
	Snapshot    NativePDReceiverSnapshot
	FrameSHA256 string
}

// NativePDReceiverTransaction is owned by the configured receiver, not a request.
// Materialize returns an independently owned frame at the consumer-ready endpoint;
// a successful hop receipt alone is never evidence that those bytes exist.
type NativePDReceiverTransaction interface {
	fabricmap.Executor
	Materialize(context.Context) ([]byte, error)
	Close() error
}

// NativePDReceiverFabric binds one decode worker's trusted topology, policy and
// byte adapter. Configure before use and do not mutate it while the cluster runs.
// The adapter is explicit: this does not claim RDMA, GPU or physical-device work.
// Snapshots/rechecks are point-in-time evidence, not a distributed topology lock.
// The receiver owner must keep adapter resources/authority valid through the call.
type NativePDReceiverFabric struct {
	Provider                                       fabricmap.Provider
	Authorizer                                     fabricmap.Authorizer
	SourceEndpoint, LandingEndpoint, ReadyEndpoint string
	Access                                         fabricmap.Access
	Open                                           func(context.Context, []byte) (NativePDReceiverTransaction, error)
	once                                           sync.Once
	slot                                           chan struct{}
	snapshots                                      fabricmap.SnapshotSet
}

// WithNativePDReceiverFabric opts into receiver-owned two-hop admission. A nil
// map retains the existing direct transport. A selected missing worker refuses;
// it cannot silently fall back to a request-chosen transfer backend.
func WithNativePDReceiverFabric(receivers map[string]*NativePDReceiverFabric) NativePDOption {
	return func(c *NativePDCluster) {
		if receivers == nil {
			return
		}
		c.receiverFabric = make(map[string]*NativePDReceiverFabric, len(receivers))
		for worker, receiver := range receivers {
			c.receiverFabric[worker] = receiver
		}
	}
}

func (r *NativePDReceiverFabric) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.once.Do(func() { r.slot = make(chan struct{}, 1) })
	select {
	case <-ctx.Done():
		return ctx.Err()
	case r.slot <- struct{}{}:
	}
	if err := ctx.Err(); err != nil {
		<-r.slot
		return err
	}
	return nil
}

func (r *NativePDReceiverFabric) route(ctx context.Context, size uint64) (fabricmap.Route, NativePDReceiverSnapshot, error) {
	var snapshotIdentity NativePDReceiverSnapshot
	snapshot, err := r.Provider.Snapshot(ctx)
	if err != nil {
		return fabricmap.Route{}, snapshotIdentity, err
	}
	if snapshot.ValidUntil.IsZero() {
		return fabricmap.Route{}, snapshotIdentity, errors.New("receiver snapshot requires an explicit expiry")
	}
	// SnapshotSet compares capabilities independent of endpoint/link order and
	// may sort its input slices. Own those slices and their labels, then use the
	// same canonical ID order for validation and the snapshot digest. Never let
	// a recheck mutate provider-owned metadata or change identity only by order.
	snapshot.Endpoints = slices.Clone(snapshot.Endpoints)
	for i := range snapshot.Endpoints {
		snapshot.Endpoints[i].Labels = maps.Clone(snapshot.Endpoints[i].Labels)
	}
	snapshot.Links = slices.Clone(snapshot.Links)
	for i := range snapshot.Links {
		snapshot.Links[i].Labels = maps.Clone(snapshot.Links[i].Labels)
	}
	slices.SortFunc(snapshot.Endpoints, func(a, b fabricmap.Endpoint) int { return cmp.Compare(a.ID, b.ID) })
	slices.SortFunc(snapshot.Links, func(a, b fabricmap.Link) int { return cmp.Compare(a.ID, b.ID) })
	now := time.Now()
	if err = r.snapshots.Update(snapshot, now); err != nil {
		return fabricmap.Route{}, snapshotIdentity, err
	}
	// Validate generations through SnapshotSet, but construct routes ONLY from this snapshot.
	// Retained snapshots from a previous provider identity must not supply a route
	// attributed to the current provider's unrelated snapshot receipt.
	graph := fabricmap.Graph{Endpoints: snapshot.Endpoints, Links: snapshot.Links}
	route, err := graph.Plan(fabricmap.Request{From: r.SourceEndpoint, To: r.ReadyEndpoint, Bytes: size})
	if err != nil {
		return route, snapshotIdentity, err
	}
	if len(route.Links) != 2 || route.Links[0].To != r.LandingEndpoint || route.Links[1].From != r.LandingEndpoint {
		return route, snapshotIdentity, errors.New("route must land then materialize at the configured receiver endpoint")
	}
	raw, err := json.Marshal(snapshot)
	if err != nil {
		return route, snapshotIdentity, err
	}
	hash := sha256.Sum256(raw)
	snapshotIdentity = NativePDReceiverSnapshot{Provider: snapshot.Provider, Generation: snapshot.Generation, ValidUntil: snapshot.ValidUntil, SHA256: hex.EncodeToString(hash[:])}
	return route, snapshotIdentity, nil
}

func (r *NativePDReceiverFabric) access(digest string) fabricmap.Access {
	access := r.Access
	access.Data = make(map[string]string, len(r.Access.Data)+1)
	for k, v := range r.Access.Data {
		access.Data[k] = v
	}
	access.Data["frame_sha256"] = digest
	return access
}

// validateAdmission runs under the receiver gate immediately before scheduler
// enqueue, including after CPU decoding/cloning of the received frame.
func (r *NativePDReceiverFabric) validateAdmission(ctx context.Context, receipt *NativePDReceiverReceipt) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	route, snapshotIdentity, err := r.route(ctx, receipt.Hops[0].Bytes)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(snapshotIdentity, receipt.Snapshot) {
		return errors.New("receiver snapshot identity changed before admission")
	}
	if _, err = fabricmap.AuthorizeRoute(route, r.access(receipt.FrameSHA256), r.Authorizer); err != nil {
		return err
	}
	return ctx.Err()
}

// receiverTransfer returns owned decoded KV and the receiver transaction gate.
// The caller must hold that gate through scheduler/residency admission and free
// the KV on refusal. Adapter cleanup and topology revalidation finish first.
func (c *NativePDCluster) receiverTransfer(ctx context.Context, req NativePDRequest, decode *NativePDWorker, src *model.PagedKV, tokens int) (out model.PagedKVTransferReceipt, execution *NativePDReceiverReceipt, release func(), err error) {
	fail := func(cause error) (model.PagedKVTransferReceipt, *NativePDReceiverReceipt, func(), error) {
		return model.PagedKVTransferReceipt{}, nil, nil, fmt.Errorf("%w: %w", ErrNativePDReceiver, cause)
	}
	r := c.receiverFabric[decode.name]
	if r == nil || r.Provider == nil || r.Authorizer == nil || r.Open == nil || r.SourceEndpoint == "" || r.LandingEndpoint == "" || r.ReadyEndpoint == "" || r.SourceEndpoint == r.LandingEndpoint || r.LandingEndpoint == r.ReadyEndpoint || r.SourceEndpoint == r.ReadyEndpoint || r.Access.Operation != fabricmap.OperationCopy {
		return fail(errors.New("incomplete receiver-owned configuration"))
	}
	if req.TransferBackend != "" {
		return fail(errors.New("request cannot override the receiver-owned transfer"))
	}
	transfer := c.transferDescriptor(req, decode.name, tokens)
	if transfer.AdmissionVerdict != cachemeta.AdmissionAllow {
		return fail(errors.New("KV governance does not admit the receiver transfer"))
	}
	transfer.Backend = "receiver-fabric"
	frame, e := model.MarshalPagedKVTransfer(src, transfer, 0, tokens)
	if e != nil {
		return fail(e)
	}
	sourceHash := sha256.Sum256(frame)
	digest := hex.EncodeToString(sourceHash[:])
	if e = r.acquire(ctx); e != nil {
		return fail(e)
	}
	owned := true
	defer func() {
		if owned {
			<-r.slot
		}
	}()
	route, snapshotIdentity, e := r.route(ctx, uint64(len(frame)))
	if e != nil {
		return fail(e)
	}
	access := r.access(digest)
	authorized, e := fabricmap.AuthorizeRoute(route, access, r.Authorizer)
	if e != nil {
		return fail(e)
	}
	tx, e := r.Open(ctx, append([]byte(nil), frame...))
	if e != nil {
		if tx != nil {
			_ = tx.Close()
		}
		return fail(e)
	}
	if tx == nil {
		return fail(errors.New("receiver adapter returned no transaction"))
	}
	closed := false
	defer func() {
		if !closed {
			_ = tx.Close()
		}
	}()
	observed, e := fabricmap.Execute(ctx, authorized, uint64(len(frame)), tx)
	if e != nil {
		return fail(e)
	}
	for _, hop := range observed.Hops {
		if hop.Integrity.Algorithm != "sha256" || hop.Integrity.Digest != digest {
			return fail(errors.New("hop does not bind the source frame"))
		}
	}
	ready, e := tx.Materialize(ctx)
	if e != nil {
		return fail(e)
	}
	ready = append([]byte(nil), ready...)
	if len(ready) != len(frame) || sha256.Sum256(ready) != sourceHash {
		return fail(errors.New("receiver materialization does not match source frame"))
	}
	if e = tx.Close(); e != nil {
		closed = true
		return fail(e)
	}
	closed = true
	revalidated, finalSnapshot, e := r.route(ctx, uint64(len(frame)))
	if e != nil {
		return fail(e)
	}
	if !reflect.DeepEqual(route, revalidated) || !reflect.DeepEqual(snapshotIdentity, finalSnapshot) {
		return fail(errors.New("receiver route changed before admission"))
	}
	if _, e = fabricmap.AuthorizeRoute(revalidated, access, r.Authorizer); e != nil {
		return fail(e)
	}
	if e = ctx.Err(); e != nil {
		return fail(e)
	}
	out, e = model.UnmarshalPagedKVTransfer(decode.pool, ready)
	if e != nil {
		return fail(e)
	}
	if out.Transfer.Outcome != cachemeta.KVTransferOK {
		out.KV.Free()
		return fail(errors.New("receiver transfer outcome is not complete"))
	}
	// Join consumer-ready identity into the existing normalized transfer entry,
	// which the decode worker stores. Keep native KV bytes distinct from frame bytes.
	out.Entry.Labels["receiver_provider"] = snapshotIdentity.Provider
	out.Entry.Labels["receiver_generation"] = strconv.FormatUint(snapshotIdentity.Generation, 10)
	out.Entry.Labels["receiver_snapshot_sha256"] = snapshotIdentity.SHA256
	out.Entry.Labels["receiver_snapshot_valid_until"] = snapshotIdentity.ValidUntil.UTC().Format(time.RFC3339Nano)
	out.Entry.Labels["receiver_ready_endpoint"] = r.ReadyEndpoint
	out.Entry.Labels["receiver_landing_endpoint"] = r.LandingEndpoint
	out.Entry.Labels["receiver_frame_sha256"] = digest
	out.Entry.Labels["receiver_wire_bytes"] = strconv.Itoa(len(frame))
	owned = false
	return out, &NativePDReceiverReceipt{ExecutionReceipt: observed, Snapshot: snapshotIdentity, FrameSHA256: digest}, func() { <-r.slot }, nil
}
