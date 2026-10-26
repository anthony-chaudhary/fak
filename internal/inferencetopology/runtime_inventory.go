package inferencetopology

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"time"
)

const RuntimeTopologySchema = "fak-runtime-topology/1"

const (
	RuntimeRolePrefill = "prefill"
	RuntimeRoleDecode  = "decode"
	RuntimeRoleBoth    = "both"
)

// RuntimeTopology is an immutable, epoch-bound account of configured rank
// placement and the observations that make that placement eligible to run.
// The four evidence planes are deliberately separate: configuration is never
// treated as proof that a device was observed, supported, or available.
type RuntimeTopology struct {
	Schema     string                     `json:"schema"`
	ID         string                     `json:"id"`
	Configured RuntimeTopologyConfigured  `json:"configured"`
	Observed   RuntimeTopologyObserved    `json:"observed"`
	Supported  []RuntimeTopologySupport   `json:"supported"`
	Available  []RuntimeTopologyAvailable `json:"available"`
}

type RuntimeTopologyConfigured struct {
	Groups []RuntimeTopologyGroup `json:"groups"`
	Ranks  []RuntimeTopologyRank  `json:"ranks"`
}

type RuntimeTopologyGroup struct {
	ID       string   `json:"id"`
	Role     string   `json:"role"`
	TP       int      `json:"tp"`
	PP       int      `json:"pp"`
	EP       int      `json:"ep"`
	Layers   int      `json:"layers"`
	KVFormat string   `json:"kv_format"`
	RankIDs  []string `json:"rank_ids"`
}

type RuntimeTopologyRank struct {
	ID                string    `json:"id"`
	Incarnation       string    `json:"incarnation"`
	GroupID           string    `json:"group_id"`
	Role              string    `json:"role"`
	DeviceID          string    `json:"device_id"`
	DeviceIncarnation string    `json:"device_incarnation"`
	TPRank            int       `json:"tp_rank"`
	PPStage           int       `json:"pp_stage"`
	EPRank            int       `json:"ep_rank"`
	LayerStart        int       `json:"layer_start"`
	LayerEnd          int       `json:"layer_end"`
	KVOwnership       []string  `json:"kv_ownership"`
	Commitment        Resources `json:"commitment"`
}

type RuntimeTopologyObserved struct {
	At         string                  `json:"at"`
	TTLSeconds int64                   `json:"ttl_seconds"`
	Epoch      uint64                  `json:"epoch"`
	Provenance string                  `json:"provenance"`
	Visibility []string                `json:"visibility"`
	Devices    []RuntimeObservedDevice `json:"devices"`
	Links      []RuntimeObservedLink   `json:"links"`
}

type RuntimeObservedDevice struct {
	ID             string  `json:"id"`
	Incarnation    string  `json:"incarnation"`
	Lifecycle      string  `json:"lifecycle"`
	Health         string  `json:"health"`
	CapacityBytes  float64 `json:"capacity_bytes"`
	FreeBytes      float64 `json:"free_bytes"`
	FreeBytesKnown bool    `json:"free_bytes_known"`
}

type RuntimeObservedLink struct {
	FromDeviceID          string  `json:"from_device_id"`
	FromDeviceIncarnation string  `json:"from_device_incarnation"`
	ToDeviceID            string  `json:"to_device_id"`
	ToDeviceIncarnation   string  `json:"to_device_incarnation"`
	Kind                  string  `json:"kind"`
	State                 string  `json:"state"`
	BytesPerSecond        float64 `json:"bytes_per_second"`
}

type RuntimeTopologySupport struct {
	DeviceID    string `json:"device_id"`
	Incarnation string `json:"incarnation"`
	State       string `json:"state"`
}

type RuntimeTopologyAvailable struct {
	DeviceID    string `json:"device_id"`
	Incarnation string `json:"incarnation"`
	State       string `json:"state"`
}

// RuntimeTopologyEligibility is a canonical eligibility receipt. Its slices
// are fresh copies, sorted by stable identity, so discovery order cannot affect
// a decision and later caller mutations cannot change the validated receipt.
type RuntimeTopologyEligibility struct {
	Schema           string   `json:"schema"`
	TopologyID       string   `json:"topology_id"`
	ObservationEpoch uint64   `json:"observation_epoch"`
	Configured       []string `json:"configured"`
	Observed         []string `json:"observed"`
	Supported        []string `json:"supported"`
	Available        []string `json:"available"`
}

// DecodeRuntimeTopology decodes one bounded, strict runtime topology document.
func DecodeRuntimeTopology(r io.Reader) (RuntimeTopology, error) {
	var topology RuntimeTopology
	raw, err := io.ReadAll(io.LimitReader(r, (16<<20)+1))
	if err != nil {
		return topology, fmt.Errorf("%w: read: %v", ErrInvalid, err)
	}
	if len(raw) > 16<<20 {
		return topology, invalid("runtime topology exceeds 16 MiB")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&topology); err != nil {
		return topology, fmt.Errorf("%w: decode: %v", ErrInvalid, err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return topology, fmt.Errorf("%w: trailing JSON", ErrInvalid)
	}
	return topology, nil
}

// ValidateRuntimeTopology fails closed unless every configured rank has a
// visible, current, active, supported and available device incarnation and the
// complete TP/PP/EP and ownership graph is internally consistent.
func ValidateRuntimeTopology(t RuntimeTopology, now time.Time) (RuntimeTopologyEligibility, error) {
	if t.Schema != RuntimeTopologySchema || t.ID == "" {
		return RuntimeTopologyEligibility{}, invalid("runtime topology schema and id required")
	}
	observedAt, err := time.Parse(time.RFC3339Nano, t.Observed.At)
	maxTTLSeconds := int64(^uint64(0)>>1) / int64(time.Second)
	if err != nil || t.Observed.TTLSeconds <= 0 || t.Observed.TTLSeconds > maxTTLSeconds || t.Observed.Epoch == 0 || t.Observed.Provenance == "" {
		return RuntimeTopologyEligibility{}, invalid("observation time, ttl, epoch and provenance required")
	}
	if now.Before(observedAt) || !now.Before(observedAt.Add(time.Duration(t.Observed.TTLSeconds)*time.Second)) {
		return RuntimeTopologyEligibility{}, invalid("observation is future or stale")
	}
	if len(t.Observed.Visibility) == 0 {
		return RuntimeTopologyEligibility{}, invalid("explicit nonempty visibility required")
	}

	visible := make(map[string]bool, len(t.Observed.Visibility))
	for _, id := range t.Observed.Visibility {
		if id == "" || visible[id] {
			return RuntimeTopologyEligibility{}, invalid("visibility contains empty or duplicate device")
		}
		visible[id] = true
	}
	devices := make(map[[2]string]RuntimeObservedDevice, len(t.Observed.Devices))
	observedDeviceIDs := make(map[string]bool, len(t.Observed.Devices))
	for _, d := range t.Observed.Devices {
		key := [2]string{d.ID, d.Incarnation}
		if d.ID == "" || d.Incarnation == "" || devices[key].ID != "" || observedDeviceIDs[d.ID] {
			return RuntimeTopologyEligibility{}, invalid("observed device %q identity", d.ID)
		}
		devices[key] = d
		observedDeviceIDs[d.ID] = true
	}
	for id := range visible {
		if !observedDeviceIDs[id] {
			return RuntimeTopologyEligibility{}, invalid("visible device %q was not observed", id)
		}
	}

	supported := make(map[[2]string]RuntimeTopologySupport, len(t.Supported))
	for _, s := range t.Supported {
		key := [2]string{s.DeviceID, s.Incarnation}
		if s.DeviceID == "" || s.Incarnation == "" || supported[key].DeviceID != "" {
			return RuntimeTopologyEligibility{}, invalid("device %q support is unknown or duplicated", s.DeviceID)
		}
		supported[key] = s
	}
	available := make(map[[2]string]RuntimeTopologyAvailable, len(t.Available))
	for _, a := range t.Available {
		key := [2]string{a.DeviceID, a.Incarnation}
		if a.DeviceID == "" || a.Incarnation == "" || available[key].DeviceID != "" {
			return RuntimeTopologyEligibility{}, invalid("device %q availability is unknown or duplicated", a.DeviceID)
		}
		available[key] = a
	}

	groups := make(map[string]RuntimeTopologyGroup, len(t.Configured.Groups))
	if len(t.Configured.Groups) == 0 || len(t.Configured.Ranks) == 0 {
		return RuntimeTopologyEligibility{}, invalid("configured groups and ranks required")
	}
	for _, g := range t.Configured.Groups {
		rankCount, countOK := runtimeRankCount(g.TP, g.PP, g.EP)
		if g.ID == "" || groups[g.ID].ID != "" || !runtimeRole(g.Role) || !countOK || g.Layers < 1 || g.KVFormat == "" || len(g.RankIDs) != rankCount {
			return RuntimeTopologyEligibility{}, invalid("group %q dimensions or identity", g.ID)
		}
		groups[g.ID] = g
	}

	ranks := make(map[string]RuntimeTopologyRank, len(t.Configured.Ranks))
	deviceOwner := map[string]string{}
	kvOwner := map[string]string{}
	groupCoordinates := map[string]map[[3]int]bool{}
	groupMembers := map[string]map[string]bool{}
	receipt := RuntimeTopologyEligibility{Schema: "fak-runtime-topology-eligibility/1", TopologyID: t.ID, ObservationEpoch: t.Observed.Epoch}
	for _, r := range t.Configured.Ranks {
		g, ok := groups[r.GroupID]
		if r.ID == "" || r.Incarnation == "" || ranks[r.ID].ID != "" || !ok || r.Role != g.Role || r.DeviceID == "" || r.DeviceIncarnation == "" || r.TPRank < 0 || r.TPRank >= g.TP || r.PPStage < 0 || r.PPStage >= g.PP || r.EPRank < 0 || r.EPRank >= g.EP || r.LayerStart < 0 || r.LayerEnd <= r.LayerStart || r.LayerEnd > g.Layers || len(r.KVOwnership) == 0 {
			return RuntimeTopologyEligibility{}, invalid("rank %q identity, role, group or coordinates", r.ID)
		}
		if owner := deviceOwner[r.DeviceID]; owner != "" {
			return RuntimeTopologyEligibility{}, invalid("device %q ownership overlaps ranks %q and %q", r.DeviceID, owner, r.ID)
		}
		deviceOwner[r.DeviceID] = r.ID
		for _, shard := range r.KVOwnership {
			if shard == "" || kvOwner[shard] != "" {
				return RuntimeTopologyEligibility{}, invalid("kv ownership %q is empty or overlapping", shard)
			}
			kvOwner[shard] = r.ID
		}
		if err := validateRuntimeResources(r.Commitment); err != nil {
			return RuntimeTopologyEligibility{}, invalid("rank %q commitment: %v", r.ID, err)
		}
		deviceKey := [2]string{r.DeviceID, r.DeviceIncarnation}
		d, observed := devices[deviceKey]
		s, supportKnown := supported[deviceKey]
		a, availabilityKnown := available[deviceKey]
		if !observed || !visible[r.DeviceID] || d.Lifecycle != "active" || d.Health != "healthy" || !nonnegative(d.CapacityBytes) || d.CapacityBytes <= 0 || !d.FreeBytesKnown || !nonnegative(d.FreeBytes) || d.FreeBytes > d.CapacityBytes {
			return RuntimeTopologyEligibility{}, invalid("rank %q device is unobserved, hidden, retired or changed incarnation", r.ID)
		}
		if !supportKnown || s.State != "supported" {
			return RuntimeTopologyEligibility{}, invalid("rank %q device is not explicitly supported", r.ID)
		}
		if !availabilityKnown || a.State != "available" {
			return RuntimeTopologyEligibility{}, invalid("rank %q device is not explicitly available", r.ID)
		}
		committed := r.Commitment.ResidentWeightBytes + r.Commitment.KVResidentBytes
		if !nonnegative(committed) || committed > d.FreeBytes {
			return RuntimeTopologyEligibility{}, invalid("rank %q commitment exceeds observed free capacity", r.ID)
		}
		coordinate := [3]int{r.TPRank, r.PPStage, r.EPRank}
		if groupCoordinates[g.ID] == nil {
			groupCoordinates[g.ID] = map[[3]int]bool{}
			groupMembers[g.ID] = map[string]bool{}
		}
		if groupCoordinates[g.ID][coordinate] {
			return RuntimeTopologyEligibility{}, invalid("group %q has duplicate TP/PP/EP coordinate", g.ID)
		}
		groupCoordinates[g.ID][coordinate] = true
		groupMembers[g.ID][r.ID] = true
		ranks[r.ID] = r
		receipt.Configured = append(receipt.Configured, r.ID)
		receipt.Observed = append(receipt.Observed, r.ID)
		receipt.Supported = append(receipt.Supported, r.ID)
		receipt.Available = append(receipt.Available, r.ID)
	}
	for _, g := range t.Configured.Groups {
		listed := map[string]bool{}
		for _, id := range g.RankIDs {
			if id == "" || listed[id] || !groupMembers[g.ID][id] {
				return RuntimeTopologyEligibility{}, invalid("group %q rank list is incomplete or duplicated", g.ID)
			}
			listed[id] = true
		}
		rankCount, _ := runtimeRankCount(g.TP, g.PP, g.EP)
		if len(groupCoordinates[g.ID]) != rankCount {
			return RuntimeTopologyEligibility{}, invalid("group %q TP/PP/EP membership is incomplete", g.ID)
		}
		for pp := 0; pp < g.PP; pp++ {
			stageStart, stageEnd := -1, -1
			for tp := 0; tp < g.TP; tp++ {
				for ep := 0; ep < g.EP; ep++ {
					if !groupCoordinates[g.ID][[3]int{tp, pp, ep}] {
						return RuntimeTopologyEligibility{}, invalid("group %q pipeline stages are noncontiguous", g.ID)
					}
					for _, rank := range ranks {
						if rank.GroupID == g.ID && rank.TPRank == tp && rank.PPStage == pp && rank.EPRank == ep {
							if stageStart < 0 {
								stageStart, stageEnd = rank.LayerStart, rank.LayerEnd
							} else if rank.LayerStart != stageStart || rank.LayerEnd != stageEnd {
								return RuntimeTopologyEligibility{}, invalid("group %q pipeline stage %d has inconsistent layer cuts", g.ID, pp)
							}
						}
					}
				}
			}
			if pp == 0 && stageStart != 0 {
				return RuntimeTopologyEligibility{}, invalid("group %q pipeline layers do not start at zero", g.ID)
			}
			if pp > 0 {
				previousEnd := -1
				for _, rank := range ranks {
					if rank.GroupID == g.ID && rank.PPStage == pp-1 {
						previousEnd = rank.LayerEnd
						break
					}
				}
				if stageStart != previousEnd {
					return RuntimeTopologyEligibility{}, invalid("group %q pipeline layer cuts have a gap or overlap", g.ID)
				}
			}
			if pp == g.PP-1 && stageEnd != g.Layers {
				return RuntimeTopologyEligibility{}, invalid("group %q pipeline layers do not cover all layers", g.ID)
			}
		}
		if err := validateRuntimeGroupLinks(g, ranks, observedDeviceIDs, t.Observed.Links); err != nil {
			return RuntimeTopologyEligibility{}, err
		}
	}
	for _, values := range [][]string{receipt.Configured, receipt.Observed, receipt.Supported, receipt.Available} {
		sort.Strings(values)
	}
	return receipt, nil
}

func runtimeRole(role string) bool {
	return role == RuntimeRolePrefill || role == RuntimeRoleDecode || role == RuntimeRoleBoth
}

func runtimeRankCount(tp, pp, ep int) (int, bool) {
	if tp < 1 || pp < 1 || ep < 1 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if tp > maxInt/pp || tp*pp > maxInt/ep {
		return 0, false
	}
	return tp * pp * ep, true
}

func validateRuntimeResources(r Resources) error {
	for _, v := range []float64{r.ResidentWeightBytes, r.WeightStreamBytesPerStep, r.KVResidentBytes, r.KVReadBytesPerStep, r.KVWriteBytesPerStep, r.ActivationTrafficBytesPerStep, r.FLOPsPerStep} {
		if !nonnegative(v) {
			return invalid("values must be finite and nonnegative")
		}
	}
	return nil
}

func validateRuntimeGroupLinks(g RuntimeTopologyGroup, ranks map[string]RuntimeTopologyRank, observedDeviceIDs map[string]bool, links []RuntimeObservedLink) error {
	if len(g.RankIDs) == 1 {
		return nil
	}
	pairs := map[runtimeDevicePair]bool{}
	pairKinds := map[runtimeDevicePair]map[string]bool{}
	for _, link := range links {
		if link.FromDeviceID == "" || link.FromDeviceIncarnation == "" || link.ToDeviceID == "" || link.ToDeviceIncarnation == "" || (link.FromDeviceID == link.ToDeviceID && link.FromDeviceIncarnation == link.ToDeviceIncarnation) || !observedDeviceIDs[link.FromDeviceID] || !observedDeviceIDs[link.ToDeviceID] || link.Kind == "" || link.State != "available" || !nonnegative(link.BytesPerSecond) || link.BytesPerSecond <= 0 {
			continue
		}
		key := orderedRuntimeDevicePair(runtimeDeviceKey{link.FromDeviceID, link.FromDeviceIncarnation}, runtimeDeviceKey{link.ToDeviceID, link.ToDeviceIncarnation})
		if pairKinds[key] == nil {
			pairKinds[key] = map[string]bool{}
		}
		if pairKinds[key][link.Kind] {
			return invalid("duplicate observed link %q-%q kind %q", key[0][0], key[1][0], link.Kind)
		}
		pairKinds[key][link.Kind] = true
		pairs[key] = true
	}
	for i, leftID := range g.RankIDs {
		for _, rightID := range g.RankIDs[i+1:] {
			left, leftOK := ranks[leftID]
			right, rightOK := ranks[rightID]
			required := left.PPStage == right.PPStage || (absRuntimeInt(left.PPStage-right.PPStage) == 1 && left.TPRank == right.TPRank && left.EPRank == right.EPRank)
			pair := orderedRuntimeDevicePair(runtimeDeviceKey{left.DeviceID, left.DeviceIncarnation}, runtimeDeviceKey{right.DeviceID, right.DeviceIncarnation})
			if required && (!leftOK || !rightOK || !pairs[pair]) {
				return invalid("group %q lacks observed link for ranks %q and %q", g.ID, leftID, rightID)
			}
		}
	}
	return nil
}

func absRuntimeInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

type runtimeDeviceKey [2]string
type runtimeDevicePair [2]runtimeDeviceKey

func orderedRuntimeDevicePair(a, b runtimeDeviceKey) runtimeDevicePair {
	if a[0] > b[0] || (a[0] == b[0] && a[1] > b[1]) {
		a, b = b, a
	}
	return runtimeDevicePair{a, b}
}
