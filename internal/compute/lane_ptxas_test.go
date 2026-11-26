package compute

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"testing/iotest"
)

// ptxasParseText parses a literal diagnostic stream and fails the test on a read error.
func ptxasParseText(t *testing.T, text string) PtxasReport {
	t.Helper()
	r, err := ParsePtxasLog(strings.NewReader(text))
	if err != nil {
		t.Fatalf("ParsePtxasLog: %v", err)
	}
	return r
}

// ptxasEntryFor returns the entry with the given mangled name, failing the test when it is absent.
func ptxasEntryFor(t *testing.T, r PtxasReport, mangled string) PtxasEntry {
	t.Helper()
	for _, e := range r.Entries {
		if e.Mangled == mangled {
			return e
		}
	}
	t.Fatalf("no entry %q in report", mangled)
	return PtxasEntry{}
}

const ptxasSampleLog = "testdata/lane_ptxas/sample.log"

func TestPtxasParseSampleFixture(t *testing.T) {
	data, err := os.ReadFile(ptxasSampleLog)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	r := ptxasParseText(t, text)

	if want := strings.Count(text, "Compiling entry function"); r.Counts.Entries != want {
		t.Errorf("entries = %d, want %d (one per Compiling line)", r.Counts.Entries, want)
	}
	if want := strings.Count(text, "Function properties for __internal"); r.Counts.CalleeBlocks != want {
		t.Errorf("callee blocks = %d, want %d", r.Counts.CalleeBlocks, want)
	}
	toolErrors := 0
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "ptxas ") && !strings.HasPrefix(line, "ptxas info") {
			toolErrors++
		}
	}
	if r.Counts.ToolErrorLines != toolErrors || toolErrors == 0 {
		t.Errorf("tool error lines = %d, want %d (non-zero)", r.Counts.ToolErrorLines, toolErrors)
	}
	if r.Counts.Incomplete != 0 || r.Counts.Undemangled != 0 || r.Counts.UnparsedClauses != 0 {
		t.Errorf("counts = %+v, want no incomplete, undemangled or unparsed clauses", r.Counts)
	}
	if len(r.ArchsSeen) != 1 || r.ArchsSeen[0] != "sm_120" {
		t.Errorf("archs = %v, want [sm_120]", r.ArchsSeen)
	}

	// Values read from the sample lines: 165 registers, no smem clause, a zero stack frame.
	plain := ptxasEntryFor(t, r, "_ZN6ink2pa17k_ink2p_attend_v2ILi2ELi16ELi1ELi1EEEvPKfS2_S2_S2_S2_S2_Pfiiiiifii")
	if plain.BaseName != "ink2pa::k_ink2p_attend_v2" || plain.Registers != 165 || plain.Barriers != 1 ||
		plain.SmemBytes != 0 || !plain.HasFrame || plain.StackFrameBytes != 0 || !plain.HasRegisters {
		t.Errorf("plain entry = %+v", plain)
	}

	// "48 bytes stack frame, 112 bytes spill stores, 48 bytes spill loads" with the stack-then-smem clause order
	// ptxas actually prints.
	spill := ptxasEntryFor(t, r, "_ZN5ink2d11k_ink2d_mmqILi6ELi128EEEvPKhPKfPKaPK6float2Pfiii")
	if spill.Registers != 128 || spill.SmemBytes != 33792 || spill.CumulativeStackBytes != 48 ||
		spill.StackFrameBytes != 48 || spill.SpillStoreBytes != 112 || spill.SpillLoadBytes != 48 {
		t.Errorf("spill entry = %+v", spill)
	}

	// "Used 255 registers, used 1 barriers, 56 bytes cumulative stack size" with an entry-level frame of 56.
	capped := ptxasEntryFor(t, r, "_ZN12fak_q56k_mmq9mul_mat_qILi6ELi112ELb0EEEvPKcPKiPfS5_5uint3iiiiiS6_")
	if capped.BaseName != "fak_q56k_mmq::mul_mat_q" || capped.Registers != 255 || capped.SmemBytes != 0 ||
		capped.CumulativeStackBytes != 56 || capped.SpillStoreBytes != 72 || capped.SpillLoadBytes != 64 {
		t.Errorf("capped entry = %+v", capped)
	}
}

func TestPtxasTolerantUsedGrammar(t *testing.T) {
	const compiling = "ptxas info    : Compiling entry function '_Z3fooi' for 'sm_120'\n"
	const props = "ptxas info    : Function properties for _Z3fooi\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
	tests := []struct {
		name         string
		text         string
		wantRegs     int
		wantHas      bool
		wantUnparsed []string
		wantUnattach int
		check        func(t *testing.T, e PtxasEntry)
	}{
		{
			name:     "clauses in any order",
			text:     compiling + props + "ptxas info    : Used 128 registers, used 1 barriers, 33792 bytes smem, 48 bytes cumulative stack size\n",
			wantRegs: 128,
			wantHas:  true,
			check: func(t *testing.T, e PtxasEntry) {
				if e.SmemBytes != 33792 || e.CumulativeStackBytes != 48 || e.Barriers != 1 {
					t.Errorf("entry = %+v", e)
				}
			},
		},
		{
			name:         "unknown clause kept raw",
			text:         compiling + props + "ptxas info    : Used 64 registers, used 1 barriers, 12 bytes frobnicate\n",
			wantRegs:     64,
			wantHas:      true,
			wantUnparsed: []string{"12 bytes frobnicate"},
		},
		{
			name:     "cmem clauses by bank",
			text:     compiling + props + "ptxas info    : Used 40 registers, used 1 barriers, 380 bytes cmem[0], 16 bytes cmem[2]\n",
			wantRegs: 40,
			wantHas:  true,
			check: func(t *testing.T, e PtxasEntry) {
				if e.CmemBytes[0] != 380 || e.CmemBytes[2] != 16 || len(e.CmemBytes) != 2 {
					t.Errorf("cmem = %v", e.CmemBytes)
				}
			},
		},
		{
			name:     "missing registers clause is incomplete, not zero",
			text:     compiling + props + "ptxas info    : Used 0 bytes smem\n",
			wantRegs: 0,
			wantHas:  false,
		},
		{
			name:         "duplicate clause is unparsed",
			text:         compiling + props + "ptxas info    : Used 8 registers, used 1 barriers, used 2 barriers\n",
			wantRegs:     8,
			wantHas:      true,
			wantUnparsed: []string{"used 2 barriers"},
		},
		{
			name:         "second Used line for one entry is unparsed",
			text:         compiling + props + "ptxas info    : Used 8 registers, used 1 barriers\nptxas info    : Used 9 registers, used 1 barriers\n",
			wantRegs:     8,
			wantHas:      true,
			wantUnparsed: []string{"ptxas info    : Used 9 registers, used 1 barriers"},
		},
		{
			name:         "malformed stack line is unparsed and frame stays absent",
			text:         compiling + "ptxas info    : Function properties for _Z3fooi\n    0 bytes stack frame, x bytes spill stores, 0 bytes spill loads\nptxas info    : Used 8 registers, used 1 barriers\n",
			wantRegs:     8,
			wantHas:      true,
			wantUnparsed: []string{"    0 bytes stack frame, x bytes spill stores, 0 bytes spill loads"},
			check: func(t *testing.T, e PtxasEntry) {
				if e.HasFrame {
					t.Errorf("HasFrame set from a malformed stack line")
				}
			},
		},
		{
			name:     "entry without a properties block has no frame",
			text:     compiling + "ptxas info    : Used 8 registers, used 1 barriers\n",
			wantRegs: 8,
			wantHas:  true,
			check: func(t *testing.T, e PtxasEntry) {
				if e.HasFrame {
					t.Errorf("HasFrame set without a properties block")
				}
			},
		},
		{
			name:     "callee block does not take the next entry's frame",
			text:     "ptxas info    : Function properties for __internal_accurate_pow\n    16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n" + compiling + props + "ptxas info    : Used 8 registers, used 1 barriers\n",
			wantRegs: 8,
			wantHas:  true,
			check: func(t *testing.T, e PtxasEntry) {
				if e.StackFrameBytes != 0 {
					t.Errorf("stack frame = %d, want the entry's own 0", e.StackFrameBytes)
				}
			},
		},
		{
			name:         "malformed Compiling line detaches following Used line",
			text:         compiling + props + "ptxas info    : Used 8 registers, used 1 barriers\nptxas info    : Compiling entry function '_Z3fooi' for sm_120\nptxas info    : Used 99 registers, used 1 barriers\n",
			wantRegs:     8,
			wantHas:      true,
			wantUnattach: 2,
		},
		{
			name:         "unknown info line inside an entry",
			text:         compiling + "ptxas info    : something new\n" + props + "ptxas info    : Used 8 registers, used 1 barriers\n",
			wantRegs:     8,
			wantHas:      true,
			wantUnparsed: []string{"ptxas info    : something new"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := ptxasParseText(t, tt.text)
			if len(r.Entries) != 1 {
				t.Fatalf("entries = %d, want 1", len(r.Entries))
			}
			e := r.Entries[0]
			if e.Registers != tt.wantRegs || e.HasRegisters != tt.wantHas {
				t.Errorf("registers = %d has=%v, want %d has=%v", e.Registers, e.HasRegisters, tt.wantRegs, tt.wantHas)
			}
			if strings.Join(e.Unparsed, "|") != strings.Join(tt.wantUnparsed, "|") {
				t.Errorf("unparsed = %q, want %q", e.Unparsed, tt.wantUnparsed)
			}
			if len(r.Unattached) != tt.wantUnattach {
				t.Errorf("unattached = %q, want %d", r.Unattached, tt.wantUnattach)
			}
			if tt.check != nil {
				tt.check(t, e)
			}
		})
	}
}

func TestPtxasCountsAndEdges(t *testing.T) {
	t.Run("empty stream has no archs and no entries", func(t *testing.T) {
		r := ptxasParseText(t, "")
		if len(r.Entries) != 0 || r.ArchsSeen == nil || len(r.ArchsSeen) != 0 {
			t.Errorf("report = %+v", r)
		}
	})
	t.Run("incomplete entry is counted", func(t *testing.T) {
		r := ptxasParseText(t, "ptxas info    : Compiling entry function '_Z3fooi' for 'sm_120'\n")
		if r.Counts.Incomplete != 1 || r.Counts.Entries != 1 {
			t.Errorf("counts = %+v", r.Counts)
		}
	})
	t.Run("tool error lines are counted, not entries", func(t *testing.T) {
		r := ptxasParseText(t, "ptxas fatal   : Ptx assembly aborted due to errors\n")
		if r.Counts.ToolErrorLines != 1 || r.Counts.Entries != 0 {
			t.Errorf("counts = %+v", r.Counts)
		}
	})
	t.Run("two architectures are both reported", func(t *testing.T) {
		r := ptxasParseText(t, "ptxas info    : Compiling entry function '_Z3fooi' for 'sm_120'\n"+
			"ptxas info    : Compiling entry function '_Z3bari' for 'sm_100'\n")
		if strings.Join(r.ArchsSeen, ",") != "sm_100,sm_120" {
			t.Errorf("archs = %v", r.ArchsSeen)
		}
	})
	t.Run("read error is returned, not swallowed", func(t *testing.T) {
		_, err := ParsePtxasLog(iotest.ErrReader(errors.New("disk gone")))
		if err == nil || !strings.Contains(err.Error(), "disk gone") {
			t.Errorf("err = %v, want the reader error", err)
		}
	})
}

// ptxasDemangleCases are real mangled kernel names from the sm_120 log, with the qualified base name a
// demangler gives them. Every one is checked; the excerpt test below checks the demangled names too.
var ptxasDemangleCases = []struct{ mangled, base string }{
	{"_ZN6ink2pa17k_ink2p_attend_v2ILi2ELi16ELi1ELi1EEEvPKfS2_S2_S2_S2_S2_Pfiiiiifii", "ink2pa::k_ink2p_attend_v2"},
	{"_ZN5ink2d11k_ink2d_mmqILi6ELi128EEEvPKhPKfPKaPK6float2Pfiii", "ink2d::k_ink2d_mmq"},
	{"_Z10k_ink2_mixILi1EEv10Ink2StatesiPKfPfi", "k_ink2_mix"},
	{"_Z11k_ink2_mixcILi4EEv10Ink2StatesiPKf9Ink2NormsifPfi", "k_ink2_mixc"},
	{"_Z12k_ink2d_fuse9Ink2DTapsiPKfPfif", "k_ink2d_fuse"},
	{"_Z12k_f32_to_f16PKfP6__halfi", "k_f32_to_f16"},
	{"_Z11k_gptq_gemvPKjS0_PKfPKiS2_Pfiiiii", "k_gptq_gemv"},
	{"_ZN13fak_fattn_mma24k_qwen38_fattn_mma_hd256EPKfPK7__half2S4_Pfiiiif", "fak_fattn_mma::k_qwen38_fattn_mma_hd256"},
	{"_ZN12fak_q56k_mmq9mul_mat_qILi6ELi112ELb0EEEvPKcPKiPfS5_5uint3iiiiiS6_", "fak_q56k_mmq::mul_mat_q"},
	{"_ZN12fak_gdn_vllm27k_qwen35_gdn_post_conv_vllmILi1ELb0EEEvPKfS2_S2_S2_S2_PfS2_S2_S3_iiiifNS_21IndependentLaneStatesE", "fak_gdn_vllm::k_qwen35_gdn_post_conv_vllm"},
	{"_Z39k_qwen35_gdn_recurrent_gated_norm_lanesILb0EEvPKfS1_S1_S1_S1_S1_S1_S1_S1_19Qwen35GDNLaneStatesPfiiiif", "k_qwen35_gdn_recurrent_gated_norm_lanes"},
	{"_Z29k_qwen35_gdn_conv_state_lanesPKfS0_19Qwen35GDNLaneStatesPfii", "k_qwen35_gdn_conv_state_lanes"},
	{"_Z12k_ink2m_convILi8EEvPKfS1_S1_PfS2_S1_iifi", "k_ink2m_conv"},
	{"_Z11k_attentionPKfS0_S0_PfS1_iiiif", "k_attention"},
	{"_Z10k_add_biasPfPKfii", "k_add_bias"},
	{"_Z10k_situ_actPKfiS0_Pfiiidd", "k_situ_act"},
	{"_Z12k_fp16_splitPKfP6__halfS2_m", "k_fp16_split"},
	{"_ZN5ink2d11k_ink2d_mmqILi5ELi64EEEvPKhPKfPKaPK6float2Pfiii", "ink2d::k_ink2d_mmq"},
	{"_Z12k_ink2v_mixcILi4EEv10Ink2StatesiPKf9Ink2NormsifPfii", "k_ink2v_mixc"},
	{"_Z11k_iq3s_gemvPKhPKfPfiii", "k_iq3s_gemv"},
	{"_Z12k_iq4nl_gemvPKhPKfPfiii", "k_iq4nl_gemv"},
	{"_Z12k_q4k_coeffsPKhP6float2ii", "k_q4k_coeffs"},
	{"_ZL10k_internalPfi", "k_internal"},
	{"_ZNSt3fooEv", "std::foo"},
	{"_ZN3fooIiE3barEv", "foo::bar"},
	{"_Z3fooIiEv", "foo"},
	{"_Z3fooPFvvEv", "foo"},
	{"_Z3fooDnv", "foo"},
	{"_Z3fooDv4_fv", "foo"},
	{"_Z3fooA4_iv", "foo"},
	{"_ZN3foo3barIN3bazEEEv", "foo::bar"},
	{"_ZN1AD2Ev", "A::~A"},
	{"_ZN2ns1CIiED1Ev", "ns::C::~C"},
	{"_ZNSt6vectorIiSaIiEED1Ev", "std::vector::~vector"},
	{"_ZN3foo3bar3bazEv", "foo::bar::baz"},
	{"_ZN3foo3barC2Ev", "foo::bar::bar"},
	{"_ZN3foo3barIS_EEv", "foo::bar"},
	{"_ZN12_GLOBAL__N_13fooEv", "(anonymous namespace)::foo"},
	{"repack", "repack"},
}

func TestPtxasDemangleRealAndSyntheticNames(t *testing.T) {
	for _, tt := range ptxasDemangleCases {
		got, ok := ptxasDemangleBaseName(tt.mangled)
		if !ok {
			t.Errorf("%s: refused, want %q", tt.mangled, tt.base)
			continue
		}
		if got != tt.base {
			t.Errorf("%s = %q, want %q", tt.mangled, got, tt.base)
		}
	}
}

// ptxasMalformedNames are names the demangler must refuse. Each one is outside the subset or breaks the
// grammar, so the entry is reported undemangled rather than given a guessed base name.
var ptxasMalformedNames = []string{
	"",
	"_Z",
	"_ZN",
	"_ZN3fooE",
	"_Z3foo",
	"_Z9abc",
	"_Z03foov",
	"_Z3fooZ",
	"_ZN3fooEE",
	"_ZN3foo3barEv.",
	"_Z3fooi.isra.0",
	"_Z3fooiE",
	"_Z3fooIi",
	"_Z3fooILi1EEE",
	"_Z3fooLi1EEv",
	"_Z3foo_",
	"_ZTV3foo",
	"_ZZ3foovE3bar",
	"_ZN3foopLEi",
	"_ZN3fooS1_v",
	"_ZN3fooS0_3barEv",
	"_ZN3foo3barIXadL_Z3bazEEEEv",
	"_Z3fooFvi",
	"_ZN3foo3barEC1",
	"_ZN3foo3barE",
	"_Z3fooX",
	"1abc",
	"_Z3fooDz",
	"_ZN3fooS_3barEv",
	"_ZN2ns1AS_1fEv",
}

func TestPtxasDemangleRefusesMalformed(t *testing.T) {
	for _, name := range ptxasMalformedNames {
		if got, ok := ptxasDemangleBaseName(name); ok {
			t.Errorf("%q = %q, want refusal", name, got)
		}
	}
}

// TestPtxasDemangleMatchesRegistersExcerpt checks the demangler against the c++filt output the register table
// carries for the same kernels. The excerpt is a committed copy of 47 of its 519 rows.
func TestPtxasDemangleMatchesRegistersExcerpt(t *testing.T) {
	data, err := os.ReadFile("testdata/lane_ptxas/registers_excerpt.json")
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		Mangled   string `json:"kernel_mangled"`
		Demangled string `json:"kernel"`
	}
	if err := json.Unmarshal(data, &rows); err != nil {
		t.Fatal(err)
	}
	if len(rows) < 40 {
		t.Fatalf("excerpt has %d rows, want at least 40", len(rows))
	}
	for _, row := range rows {
		want := strings.TrimPrefix(row.Demangled, "void ")
		if i := strings.IndexAny(want, "(<"); i >= 0 {
			want = want[:i]
		}
		got, ok := ptxasDemangleBaseName(row.Mangled)
		if !ok || got != want {
			t.Errorf("%s = (%q, %v), want %q", row.Mangled, got, ok, want)
		}
	}
}

func TestPtxasSiteBaseNameStripsTemplates(t *testing.T) {
	cases := map[string]string{
		"k_qwen35_gdn_post_conv_vllm<Ratio, true>": "k_qwen35_gdn_post_conv_vllm",
		"fak_fattn_mma::k_qwen38_fattn_mma_hd256":  "fak_fattn_mma::k_qwen38_fattn_mma_hd256",
		"mul_mat_q<J, fallback>":                   "mul_mat_q",
		"k_flash_attention<false>":                 "k_flash_attention",
	}
	for in, want := range cases {
		if got := ptxasSiteBaseName(in); got != want {
			t.Errorf("ptxasSiteBaseName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPtxasJoinCases(t *testing.T) {
	report := PtxasReport{
		ArchsSeen: []string{"sm_120"},
		Entries: []PtxasEntry{
			{Mangled: "_ZN2ns4kernILi6EEv", BaseName: "ns::kern", Arch: "sm_120", HasRegisters: true, Registers: 99},
			{Mangled: "_ZN2ns4kernILi7EEv", BaseName: "ns::kern", Arch: "sm_120", HasRegisters: true, Registers: 113},
			{Mangled: "_Z5other", BaseName: "other", Arch: "sm_120", HasRegisters: true, Registers: 30},
			{Mangled: "_ZZ3foovE3bar", BaseName: "", Arch: "sm_120", HasRegisters: true, Registers: 7},
		},
	}
	sites := []LaunchSite{
		{File: "a.cu", Kernel: "ns::kern<Ratio>", ThreadsPerBlock: 256, SmemExpr: "0"},
		{File: "b.cu", Kernel: "missing<1>", ThreadsPerBlock: 0, SmemExpr: "0"},
		{File: "c.cu", Kernel: "other", ThreadsPerBlock: 128, SmemExpr: "0"},
	}
	join := PtxasJoinSites(report, sites)

	if join.SiteCount != 3 || join.JoinedSiteCount != 2 || join.AmbiguousSiteCount != 1 {
		t.Errorf("counts = sites %d joined %d ambiguous %d, want 3/2/1", join.SiteCount, join.JoinedSiteCount, join.AmbiguousSiteCount)
	}
	if join.JoinRate < 0.666 || join.JoinRate > 0.667 {
		t.Errorf("join rate = %v, want 2/3", join.JoinRate)
	}
	if len(join.Sites) != 2 || len(join.Sites[0].Entries) != 2 || !join.Sites[0].Ambiguous {
		t.Errorf("ambiguous site must keep both instantiations: %+v", join.Sites)
	}
	if join.Sites[0].Entries[0].Registers != 99 || join.Sites[0].Entries[1].Registers != 113 {
		t.Errorf("instantiation registers = %d, %d, want 99, 113", join.Sites[0].Entries[0].Registers, join.Sites[0].Entries[1].Registers)
	}
	if len(join.UnjoinedSites) != 1 || join.UnjoinedSites[0].Kernel != "missing<1>" {
		t.Errorf("unjoined sites = %+v", join.UnjoinedSites)
	}
	if len(join.UnjoinedEntries) != 1 || join.UnjoinedEntries[0].Mangled != "_ZZ3foovE3bar" {
		t.Errorf("unjoined entries = %+v, want only the undemangled entry", join.UnjoinedEntries)
	}
	if join.JoinedEntryCount != 3 || join.EntryCount != 4 {
		t.Errorf("entry counts joined %d of %d, want 3 of 4", join.JoinedEntryCount, join.EntryCount)
	}
	if !strings.HasPrefix(join.Provenance, ptxasProvenance) || !strings.HasSuffix(join.Provenance, "sm_120") {
		t.Errorf("provenance = %q", join.Provenance)
	}
}

// TestPtxasSweepHandComputed prices one kernel whose numbers are worked by hand on the sm_120 row.
// 137 registers, 256 threads, 84 SMs, grid 84:
//
//	warps/block = 256/32 = 8; blocks by warps = 48/8 = 6
//	per-warp registers = roundup(137*32 = 4384, 256) = 4608
//	warps the file fits = floor(65536/4608) = 14; granted = roundown(14, 4) = 12
//	blocks by registers = 12/8 = 1, so 1 block per SM, 8 of 48 warps
//	resource-capped lanes per SM = 1536 - 1*256 = 1280; x84 SMs = 107520
//	capacity = 1 wave x 84 SMs x 1536 lanes = 129024; active = 84*256 = 21504
//	register file: useful 8*4384 = 35072; rounding 8*224 = 1792; granularity 2*4608 = 9216;
//	headroom 4*4608 = 18432 (not a finding under the register limiter); remainder 65536-64512 = 1024
func TestPtxasSweepHandComputed(t *testing.T) {
	a, ok := LookupNVArch(SM120)
	if !ok {
		t.Fatal("sm_120 row missing")
	}
	report := PtxasReport{
		ArchsSeen: []string{"sm_120"},
		Entries: []PtxasEntry{{
			Mangled: "_ZN3ink8q8_gemmILi1EEvv", BaseName: "ink::k_q8_gemm_rowtile", Arch: "sm_120",
			HasRegisters: true, Registers: 137,
		}},
	}
	sites := []LaunchSite{{File: "quant_matmul.cuh", Kernel: "ink::k_q8_gemm_rowtile", ThreadsPerBlock: 256, SmemExpr: "0"}}
	rows, err := a.PtxasSweep(PtxasJoinSites(report, sites), 84, 84)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	r := rows[0]
	if r.Refusal != "" || r.Limiter != "register" || r.BlocksPerSM != 1 || r.WarpsPerBlock != 8 ||
		r.WarpsPerSM != 8 || r.MaxWarpsPerSM != 48 || r.Waves != 1 {
		t.Errorf("row occupancy = %+v", r)
	}
	if r.LaneWaveCapacity != 129024 || r.ActiveLaneSlotWaves != 21504 || r.ResourceCappedLaneSlotWaves != 107520 {
		t.Errorf("lane-slot-waves capacity %d active %d resource-capped %d, want 129024 21504 107520",
			r.LaneWaveCapacity, r.ActiveLaneSlotWaves, r.ResourceCappedLaneSlotWaves)
	}
	if len(r.LaneFindings) != 1 || r.LaneFindings[0].Name != "resource_capped" || r.LaneFindings[0].LaneSlotWaves != 107520 {
		t.Errorf("lane findings = %+v", r.LaneFindings)
	}
	want := []struct {
		name string
		regs int64
	}{{"register_granularity", 9216}, {"register_rounding", 1792}}
	if len(r.RegisterFindings) != len(want) {
		t.Fatalf("register findings = %+v, want %+v", r.RegisterFindings, want)
	}
	for i, w := range want {
		if r.RegisterFindings[i].Name != w.name || r.RegisterFindings[i].RegistersPerSM != w.regs {
			t.Errorf("register finding[%d] = %+v, want %s %d", i, r.RegisterFindings[i], w.name, w.regs)
		}
	}
	if r.SmemDynamicUnknown {
		t.Error("literal smem 0 must not be flagged dynamic")
	}
}

func TestPtxasSweepRefusals(t *testing.T) {
	a, _ := LookupNVArch(SM120)
	good := PtxasReport{
		ArchsSeen: []string{"sm_120"},
		Entries: []PtxasEntry{
			{Mangled: "k1", BaseName: "k1", Arch: "sm_120", HasRegisters: true, Registers: 40},
			{Mangled: "k2", BaseName: "k2", Arch: "sm_120", HasRegisters: false},
			{Mangled: "k3", BaseName: "k3", Arch: "sm_120", HasRegisters: true, Registers: 300},
		},
	}
	sites := []LaunchSite{
		{Kernel: "k1", ThreadsPerBlock: 128, SmemExpr: "shmem"},
		{Kernel: "k2", ThreadsPerBlock: 128, SmemExpr: "0"},
		{Kernel: "k3", ThreadsPerBlock: 128, SmemExpr: "0"},
		{Kernel: "k1", ThreadsPerBlock: 0, SmemExpr: "0"},
	}
	join := PtxasJoinSites(good, sites)

	t.Run("arch mismatch refuses the whole sweep", func(t *testing.T) {
		mismatch := join
		mismatch.ArchsSeen = []string{"sm_100"}
		if _, err := a.PtxasSweep(mismatch, 84, 84); !errors.Is(err, ErrPtxasArchMismatch) {
			t.Errorf("err = %v, want ErrPtxasArchMismatch", err)
		}
	})
	t.Run("no arch refuses", func(t *testing.T) {
		none := join
		none.ArchsSeen = nil
		if _, err := a.PtxasSweep(none, 84, 84); !errors.Is(err, ErrPtxasNoArch) {
			t.Errorf("err = %v, want ErrPtxasNoArch", err)
		}
	})
	t.Run("deviceSMs must be positive", func(t *testing.T) {
		if _, err := a.PtxasSweep(join, 0, 84); !errors.Is(err, ErrPtxasDeviceSMs) {
			t.Errorf("err = %v, want ErrPtxasDeviceSMs", err)
		}
	})
	t.Run("gridBlocks must be positive", func(t *testing.T) {
		if _, err := a.PtxasSweep(join, 84, -1); !errors.Is(err, ErrPtxasGridBlocks) {
			t.Errorf("err = %v, want ErrPtxasGridBlocks", err)
		}
	})

	rows, err := a.PtxasSweep(join, 84, 84)
	if err != nil {
		t.Fatal(err)
	}
	byKey := map[string]PtxasSweepRow{}
	for _, r := range rows {
		byKey[fmt.Sprintf("%s/%d", r.Kernel, r.ThreadsPerBlock)] = r
	}
	if r := byKey["k1/128"]; r.Refusal != "" || !r.SmemDynamicUnknown {
		t.Errorf("k1 row = %+v, want priced and dynamic smem flagged", r)
	}
	if r := byKey["k2/128"]; r.Refusal != ErrPtxasIncomplete.Error() {
		t.Errorf("incomplete row refusal = %q", r.Refusal)
	}
	if r := byKey["k3/128"]; r.Refusal != ErrLaneRegsOverLimit.Error() {
		t.Errorf("over-limit row refusal = %q", r.Refusal)
	}
	if r := byKey["k1/0"]; r.Refusal != ErrPtxasSymbolicThreads.Error() {
		t.Errorf("symbolic-threads row refusal = %q", r.Refusal)
	}
}

func TestPtxasRankSweepRows(t *testing.T) {
	rows := []PtxasSweepRow{
		{Kernel: "b", Mangled: "b", ResourceCappedLaneSlotWaves: 10},
		{Kernel: "a", Mangled: "a", ResourceCappedLaneSlotWaves: 10},
		{Kernel: "z", Mangled: "z", ResourceCappedLaneSlotWaves: 99},
		{Kernel: "r", Mangled: "r", Refusal: "x"},
	}
	ranked := RankPtxasSweepRows(rows)
	order := []string{"z", "a", "b", "r"}
	for i, w := range order {
		if ranked[i].Kernel != w {
			t.Errorf("rank[%d] = %s, want %s", i, ranked[i].Kernel, w)
		}
	}
	if rows[0].Kernel != "b" {
		t.Error("RankPtxasSweepRows must not reorder its input")
	}
}

// TestPtxasFullLogFromEnv runs the parser over the whole real sm_120 log and compares every register count
// with the parsed register table. It is gated on environment variables so the committed suite does not
// depend on a scratch path: PTXAS_FULL_LOG names the raw log and PTXAS_FULL_REGISTERS the table.
func TestPtxasFullLogFromEnv(t *testing.T) {
	logPath := os.Getenv("PTXAS_FULL_LOG")
	if logPath == "" {
		t.Skip("PTXAS_FULL_LOG not set")
	}
	f, err := os.Open(logPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := ParsePtxasLog(f)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("entries=%d incomplete=%d undemangled=%d unparsed_clauses=%d callee_blocks=%d tool_error_lines=%d archs=%v",
		r.Counts.Entries, r.Counts.Incomplete, r.Counts.Undemangled, r.Counts.UnparsedClauses,
		r.Counts.CalleeBlocks, r.Counts.ToolErrorLines, r.ArchsSeen)

	regPath := os.Getenv("PTXAS_FULL_REGISTERS")
	if regPath == "" {
		return
	}
	data, err := os.ReadFile(regPath)
	if err != nil {
		t.Fatal(err)
	}
	var table []struct {
		Mangled    string `json:"kernel_mangled"`
		Registers  int    `json:"registers"`
		SmemBytes  int64  `json:"smem_bytes"`
		StackBytes int64  `json:"stack_bytes"`
	}
	if err := json.Unmarshal(data, &table); err != nil {
		t.Fatal(err)
	}
	byName := map[string]PtxasEntry{}
	for _, e := range r.Entries {
		byName[e.Mangled] = e
	}
	diffs, smemDiffs, stackDiffs := 0, 0, 0
	for _, row := range table {
		e, ok := byName[row.Mangled]
		if !ok || !e.HasRegisters || e.Registers != row.Registers {
			diffs++
			t.Errorf("%s: parsed (%d, present=%v, has=%v) vs table %d", row.Mangled, e.Registers, ok, e.HasRegisters, row.Registers)
		}
		if e.SmemBytes != row.SmemBytes {
			smemDiffs++
		}
		if e.CumulativeStackBytes != row.StackBytes {
			stackDiffs++
		}
	}
	t.Logf("register table rows=%d register_diffs=%d smem_diffs=%d cumulative_stack_diffs=%d",
		len(table), diffs, smemDiffs, stackDiffs)
	ptxasLogFullJoinAndSweep(t, r)
}

// ptxasLogFullJoinAndSweep joins a full parse against the real launch inventory and prices the joined
// sites on sm_120 at the 84 SMs the device properties file reports, one block per site. It logs the
// measured join rate and refusal breakdown; nothing here asserts a fixed count.
func ptxasLogFullJoinAndSweep(t *testing.T, r PtxasReport) {
	t.Helper()
	a, ok := LookupNVArch(SM120)
	if !ok {
		t.Fatal("sm_120 row missing")
	}
	join := PtxasJoinSites(r, LaunchSites())
	t.Logf("join: sites=%d joined=%d rate=%.3f ambiguous=%d unjoined_entries=%d",
		join.SiteCount, join.JoinedSiteCount, join.JoinRate, join.AmbiguousSiteCount, len(join.UnjoinedEntries))
	rows, err := a.PtxasSweep(join, 84, 84)
	if err != nil {
		t.Fatal(err)
	}
	refusals := map[string]int{}
	priced := 0
	for _, row := range rows {
		if row.Refusal != "" {
			refusals[row.Refusal]++
		} else {
			priced++
		}
	}
	t.Logf("sweep: rows=%d priced=%d", len(rows), priced)
	for reason, n := range refusals {
		t.Logf("sweep refusal x%d: %s", n, reason)
	}
}

func TestPtxasDemangleTemplateArgs(t *testing.T) {
	cases := []struct {
		mangled string
		base    string
		args    []string
	}{
		{"_Z17k_flash_attentionILb0EEvPKfS1_S1_PfiiiifPKi", "k_flash_attention", []string{"false"}},
		{"_Z17k_ink2p_nv2_quantILi16EEvPKiS1_S1_PKfiiPaPfi", "k_ink2p_nv2_quant", []string{"16"}},
		{"_Z6k_ropeILb1EEvPfiiidPKi", "k_rope", []string{"true"}},
		{"_ZN2ns4kernILin5EEEv", "ns::kern", []string{"-5"}},
		{"_ZN2ns4kernIfEEv", "ns::kern", []string{ptxasTypeArg}},
		{"_Z5otherv", "other", nil},
	}
	for _, tc := range cases {
		base, args, ok := ptxasDemangle(tc.mangled)
		if !ok || base != tc.base || !slices.Equal(args, tc.args) {
			t.Errorf("ptxasDemangle(%q) = (%q, %q, %v), want (%q, %q)", tc.mangled, base, args, ok, tc.base, tc.args)
		}
	}
}

func TestPtxasSiteTemplateArgsSplitsTopLevel(t *testing.T) {
	cases := map[string][]string{
		"k_flash_attention<false>": {"false"},
		"k<Ratio, true>":           {"Ratio", "true"},
		"f<g<1, 2>, 3>":            {"g<1, 2>", "3"},
		"mul_mat_q<J, fallback>":   {"J", "fallback"},
		"plain":                    nil,
	}
	for in, want := range cases {
		if got := ptxasSiteTemplateArgs(in); !slices.Equal(got, want) {
			t.Errorf("ptxasSiteTemplateArgs(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestPtxasJoinRefusesLiteralMismatch pins the measured failure: a site whose literal template argument
// differs from the only compiled instantiation must not be priced with that instantiation's registers.
func TestPtxasJoinRefusesLiteralMismatch(t *testing.T) {
	report := PtxasReport{
		ArchsSeen: []string{"sm_120"},
		Entries: []PtxasEntry{
			{Mangled: "_Z6k_ropeILb0EEvPfiiidPKi", BaseName: "k_rope", TemplateArgs: []string{"false"}, HasRegisters: true, Registers: 32},
			{Mangled: "_Z17k_ink2p_nv2_quantILi16EEvPKi", BaseName: "k_ink2p_nv2_quant", TemplateArgs: []string{"16"}, HasRegisters: true, Registers: 27},
		},
	}
	sites := []LaunchSite{
		{File: "a.cuh", Kernel: "k_rope<true>", ThreadsPerBlock: 256, SmemExpr: "0"},
		{File: "a.cuh", Kernel: "k_rope<false>", ThreadsPerBlock: 256, SmemExpr: "0"},
		{File: "b.cuh", Kernel: "k_ink2p_nv2_quant<32>", ThreadsPerBlock: 256, SmemExpr: "0"},
		{File: "b.cuh", Kernel: "k_ink2p_nv2_quant<16>", ThreadsPerBlock: 256, SmemExpr: "0"},
	}
	join := PtxasJoinSites(report, sites)
	if join.JoinedSiteCount != 2 || join.AmbiguousSiteCount != 0 {
		t.Fatalf("joined %d ambiguous %d, want 2 and 0", join.JoinedSiteCount, join.AmbiguousSiteCount)
	}
	if len(join.UnjoinedSites) != 2 || join.UnjoinedSites[0].Kernel != "k_rope<true>" || join.UnjoinedSites[1].Kernel != "k_ink2p_nv2_quant<32>" {
		t.Errorf("unjoined sites = %+v, want the true and the 32 sites", join.UnjoinedSites)
	}
	for _, js := range join.Sites {
		if len(js.Entries) != 1 {
			t.Errorf("site %s joined %d entries, want 1", js.Site.Kernel, len(js.Entries))
		}
	}
}

func TestPtxasParseCountsFailedUnits(t *testing.T) {
	log := "##### UNIT a.cu  exit=1\n##### UNIT b.cu  exit=0\n##### UNIT c.cu\n"
	report, err := ParsePtxasLog(strings.NewReader(log))
	if err != nil {
		t.Fatal(err)
	}
	if report.Counts.Units != 2 || report.Counts.FailedUnits != 1 {
		t.Errorf("units %d failed %d, want 2 and 1", report.Counts.Units, report.Counts.FailedUnits)
	}
	if len(report.FailedUnits) != 1 || report.FailedUnits[0] != (PtxasUnitExit{Unit: "a.cu", Exit: 1}) {
		t.Errorf("failed units = %+v", report.FailedUnits)
	}
	if len(report.Unattached) != 1 || report.Unattached[0] != "##### UNIT c.cu" {
		t.Errorf("unattached = %q, want the malformed marker kept verbatim", report.Unattached)
	}
}

func TestPtxasRankRefusedRowsLast(t *testing.T) {
	rows := RankPtxasSweepRows([]PtxasSweepRow{
		{Kernel: "a_refused", Refusal: "boom", RefusalKind: PtxasRefusalLedger},
		{Kernel: "z_priced", ResourceCappedLaneSlotWaves: 0},
	})
	if rows[0].Kernel != "z_priced" || rows[1].Kernel != "a_refused" {
		t.Errorf("order = %s, %s; want the priced zero row first", rows[0].Kernel, rows[1].Kernel)
	}
}
