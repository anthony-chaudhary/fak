package model

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// TestV41DSparkIntakeWitness is the real-artifact intake witness for the
// bundled DSpark/MTP drafter: every official model-*.safetensors shard in
// FAK_V41_DSPARK_SHARD_DIR must open through OpenV41DSparkPackedShard with the
// pinned config, and together the shards must hold each required draft weight
// exactly once, each optional role (e.g. the router bias_vl) at most once, and
// no unaccounted mtp.* tensor. Only headers are read. This attests artifact
// admission, never runtime DSpark support.
// fak-test:runtime fast est=200ms lane=optin
func TestV41DSparkIntakeWitness(t *testing.T) {
	dir := os.Getenv("FAK_V41_DSPARK_SHARD_DIR")
	if dir == "" {
		t.Skip("set FAK_V41_DSPARK_SHARD_DIR to a directory of official V4.1 draft shards")
	}
	raw, cfg := readDeepSeekV41Config(t)
	if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != deepSeekV41ConfigSHA256 {
		t.Fatalf("config is not the pinned %s@%s", DeepSeekV41FlashModelID, DeepSeekV41FlashRevision)
	}
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		t.Fatal(err)
	}
	paths, err := filepath.Glob(filepath.Join(dir, "model-*.safetensors"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no model-*.safetensors under %s: %v", dir, err)
	}
	sort.Strings(paths)

	required := map[string]bool{}
	for _, name := range inv.Names() {
		required[name] = true
	}
	owners, optionalOwners := map[string][]string{}, map[string][]string{}
	dtypes := map[string]int{}
	stages := map[int]bool{}
	var unexpected []string
	scales, aliases, optional := 0, 0, 0
	for _, path := range paths {
		base := filepath.Base(path)
		shard, err := OpenV41DSparkPackedShard(path, cfg)
		if err != nil {
			t.Fatalf("%s: %v", base, err)
		}
		absent := map[string]bool{}
		for _, name := range shard.Missing() {
			absent[name] = true
		}
		shardStages := map[int]bool{}
		shardDtypes := map[string]int{}
		weights, shardScales := 0, 0
		for _, name := range inv.Names() {
			if absent[name] {
				continue
			}
			owners[name] = append(owners[name], base)
			spec, _ := inv.Lookup(name)
			ref, err := shard.Reference(name)
			if err != nil {
				t.Fatalf("%s: present role %s has no reference: %v", base, name, err)
			}
			if ref.HasScale() != (len(spec.ScaleShape) > 0) {
				t.Fatalf("%s: %s scale presence %v, inventory requires %v", base, name, ref.HasScale(), len(spec.ScaleShape) > 0)
			}
			weights++
			shardStages[spec.Stage] = true
			shardDtypes[ref.Ref.Dtype]++
			if ref.HasScale() {
				shardScales++
				shardDtypes[ref.Ref.ScaleDtype+"(scale)"]++
			}
		}
		draftHeader, shardAliases, shardOptional := 0, 0, 0
		for _, name := range safetensorsTensorNames(shard.packed.file.hdr) {
			canonical, draft := v41DSparkCanonicalName(name)
			if !draft {
				continue
			}
			draftHeader++
			if v41DSparkTargetOwnedAlias(canonical) {
				shardAliases++
				continue
			}
			if _, ok := inv.Lookup(canonical); ok && !required[canonical] {
				ref, err := shard.Reference(canonical)
				if err != nil || ref.Ref.Name != name {
					t.Fatalf("%s: optional role %s not bound to %s: %+v, %v", base, canonical, name, ref, err)
				}
				optionalOwners[canonical] = append(optionalOwners[canonical], base)
				shardDtypes[ref.Ref.Dtype]++
				shardOptional++
				continue
			}
			weight := strings.TrimSuffix(canonical, ".scale") + ".weight"
			if _, ok := inv.Lookup(canonical); ok && !absent[canonical] {
				continue
			}
			if spec, ok := inv.Lookup(weight); ok && strings.HasSuffix(canonical, ".scale") && len(spec.ScaleShape) > 0 && !absent[weight] {
				continue
			}
			unexpected = append(unexpected, base+":"+name)
		}
		if draftHeader != weights+shardOptional+shardScales+shardAliases {
			t.Fatalf("%s: %d mtp.* header tensors, accounted %d weights + %d optional + %d scales + %d aliases",
				base, draftHeader, weights, shardOptional, shardScales, shardAliases)
		}
		var stageList []int
		for stage := range shardStages {
			stageList = append(stageList, stage)
			stages[stage] = true
		}
		sort.Ints(stageList)
		t.Logf("%s: stages=%v weights=%d optional=%d scales=%d target_aliases=%d non_draft=%d dtypes=%v",
			base, stageList, weights, shardOptional, shardScales, shardAliases, len(safetensorsTensorNames(shard.packed.file.hdr))-draftHeader, shardDtypes)
		for k, v := range shardDtypes {
			dtypes[k] += v
		}
		scales += shardScales
		aliases += shardAliases
		optional += shardOptional
		if err := shard.Close(); err != nil {
			t.Fatalf("%s: close: %v", base, err)
		}
	}

	var missing, duplicate []string
	for _, name := range inv.Names() {
		switch len(owners[name]) {
		case 1:
		case 0:
			missing = append(missing, name)
		default:
			duplicate = append(duplicate, name+"@"+strings.Join(owners[name], ","))
		}
	}
	for name, shards := range optionalOwners {
		if len(shards) > 1 {
			duplicate = append(duplicate, name+"@"+strings.Join(shards, ","))
		}
	}
	sort.Strings(duplicate)
	var stageList []int
	for stage := range stages {
		stageList = append(stageList, stage)
	}
	sort.Ints(stageList)
	t.Logf("intake %s@%s: shards=%d stages=%v weights=%d/%d optional=%d scales=%d target_aliases=%d missing=%d duplicate=%d unexpected=%d dtypes=%v",
		DeepSeekV41FlashModelID, DeepSeekV41FlashRevision, len(paths), stageList,
		len(inv.Names())-len(missing), len(inv.Names()), optional, scales, aliases, len(missing), len(duplicate), len(unexpected), dtypes)
	if len(missing) != 0 || len(duplicate) != 0 || len(unexpected) != 0 {
		t.Fatalf("incomplete drafter: missing=%v duplicate=%v unexpected=%v", missing, duplicate, unexpected)
	}
}
