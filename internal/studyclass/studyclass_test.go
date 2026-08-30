package studyclass

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

const testDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func testCorpus() studyforge.Corpus {
	return studyforge.Corpus{Receipt: studyforge.Receipt{
		Repository: "owner/repo", Revision: "abc123", Cutoff: "2026-08-26T22:35:00Z", IndexChecksum: testDigest,
	}, Records: []studyforge.Record{
		{Source: "issues", Kind: "issue", ID: 2, Number: 2, State: "open", Title: "KV cache regression", Labels: []string{"bug", "kv-cache-manager"}, CreatedAt: "2026-01-01T00:00:00Z"},
		{Source: "pulls", Kind: "pull", ID: 1, Number: 1, State: "closed", Title: "scheduler kernel", MergedAt: "2026-02-01T00:00:00Z"},
		{Source: "releases", Kind: "release", ID: 3, TagName: "v1.0", PublishedAt: "2026-03-01T00:00:00Z"},
	}}
}

func TestCoverageDeterminismAndSummary(t *testing.T) {
	first, err := Classify(testCorpus(), testDigest)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Records) != len(testCorpus().Records) {
		t.Fatalf("coverage=%d", len(first.Records))
	}
	if first.Summary.RecordCount != 3 || first.Summary.ByDisposition[string(DispositionMergedLanded)] != 1 || first.Summary.ByMechanism[string(MechanismExplicitNonCandidate)] != 1 {
		t.Fatalf("summary=%+v", first.Summary)
	}
	var a, b bytes.Buffer
	if err := WriteJSON(&a, first); err != nil {
		t.Fatal(err)
	}
	second, err := Classify(testCorpus(), testDigest)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteJSON(&b, second); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Fatal("output is not byte deterministic")
	}
	if _, err := ReadJSON(bytes.NewReader(a.Bytes())); err != nil {
		t.Fatal(err)
	}
}

func TestValidationCorruptionFailures(t *testing.T) {
	out, err := Classify(testCorpus(), testDigest)
	if err != nil {
		t.Fatal(err)
	}
	out.Records[1].Identity = out.Records[0].Identity
	if err := Validate(out); err == nil {
		t.Fatal("duplicate/corrupt identity accepted")
	}
	out, _ = Classify(testCorpus(), testDigest)
	out.Records[0].Disposition = ""
	if err := Validate(out); err == nil {
		t.Fatal("unclassified record accepted")
	}
	out, _ = Classify(testCorpus(), testDigest)
	out.Records[0].Mechanisms[0].Name = "invalid"
	if err := Validate(out); err == nil {
		t.Fatal("invalid mechanism accepted")
	}
	out, _ = Classify(testCorpus(), testDigest)
	out.Records = out.Records[1:]
	if err := Validate(out); err == nil {
		t.Fatal("missing classified record accepted")
	}
	out, _ = Classify(testCorpus(), testDigest)
	out.Records[1].State = "open"
	if err := Validate(out); err == nil {
		t.Fatal("impossible open/merged state accepted")
	}
}

func TestEvidenceBasedDispositionsAreStateIndependent(t *testing.T) {
	var errs []error
	record := Classification{}
	record.Disposition = DispositionRegressionBug
	validateDispositionCoherence(record, &errs)
	record.Disposition = DispositionDuplicate
	validateDispositionCoherence(record, &errs)
	record.Disposition = DispositionSupportQuestion
	validateDispositionCoherence(record, &errs)
	record.Disposition = DispositionStaleSuperseded
	validateDispositionCoherence(record, &errs)
	if len(errs) != 0 {
		t.Fatalf("evidence-based dispositions gained state-only errors: %v", errs)
	}
}

func TestActionableClusterRequiresEvidence(t *testing.T) {
	out, err := Classify(testCorpus(), testDigest)
	if err != nil {
		t.Fatal(err)
	}
	for i := range out.Clusters {
		if out.Clusters[i].Actionable {
			out.Clusters[i].Representative.Evidence = nil
			break
		}
	}
	if err := Validate(out); err == nil {
		t.Fatal("actionable cluster without evidence accepted")
	}
}

func TestRelationshipDirectionAndLexicalHonesty(t *testing.T) {
	corpus := testCorpus()
	corpus.Records = []studyforge.Record{
		{Source: "issues", Kind: "issue", ID: 1, State: "closed", Title: "Duplicate token handling"},
		{Source: "pulls", Kind: "pull", ID: 2, State: "closed", Body: "Duplicated by #9"},
		{Source: "pulls", Kind: "pull", ID: 3, State: "closed", Body: "Duplicate of #8"},
		{Source: "pulls", Kind: "pull", ID: 4, State: "closed", Body: "Superseded by #7"},
	}
	out, err := Classify(corpus, testDigest)
	if err != nil {
		t.Fatal(err)
	}
	got := map[int64]Disposition{}
	for _, record := range out.Records {
		got[record.ID] = record.Disposition
	}
	if got[1] != DispositionClosedUnmerged || got[2] != DispositionClosedUnmerged ||
		got[3] != DispositionDuplicate || got[4] != DispositionStaleSuperseded {
		t.Fatalf("relationship dispositions=%v", got)
	}
}

func TestDuplicateCorpusIdentityRejected(t *testing.T) {
	corpus := testCorpus()
	corpus.Records = append(corpus.Records, corpus.Records[0])
	if _, err := Classify(corpus, testDigest); err == nil {
		t.Fatal("duplicate corpus identity accepted")
	}
}

func TestSchemaDocumentVocabularies(t *testing.T) {
	var schema map[string]any
	if err := json.Unmarshal(SchemaDocument(), &schema); err != nil {
		t.Fatal(err)
	}
	if schema["$schema"] != "https://json-schema.org/draft/2020-12/schema" || schema["$id"] != JSONSchemaID {
		t.Fatalf("schema identity=%v", schema)
	}
	blob := string(SchemaDocument())
	for _, d := range Dispositions {
		if !bytes.Contains([]byte(blob), []byte(`"`+d+`"`)) {
			t.Fatalf("missing disposition %s", d)
		}
	}
	for _, m := range Mechanisms {
		if !bytes.Contains([]byte(blob), []byte(`"`+m+`"`)) {
			t.Fatalf("missing mechanism %s", m)
		}
	}
	obj := SchemaObject()
	obj["title"] = "mutated"
	if SchemaObject()["title"] == "mutated" {
		t.Fatal("schema object was not independent")
	}
}

func TestCompactBoundAndRoundTrip(t *testing.T) {
	out, err := Classify(testCorpus(), testDigest)
	if err != nil {
		t.Fatal(err)
	}
	index, err := Compact(out, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, cluster := range index.Clusters {
		if len(cluster.RelatedSamples) > 1 {
			t.Fatal("sample bound exceeded")
		}
	}
	var buf bytes.Buffer
	if err := WriteCompactJSON(&buf, index); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadCompactJSON(&buf); err != nil {
		t.Fatal(err)
	}
	if err := ValidateCompactAgainst(index, out); err != nil {
		t.Fatal(err)
	}
	index.FullOutputChecksum = testDigest
	if err := ValidateCompactAgainst(index, out); err == nil {
		t.Fatal("tampered compact full-output checksum accepted")
	}
}

func TestLiveCorpusSmoke(t *testing.T) {
	path := os.Getenv("FAK_STUDYCLASS_CORPUS")
	if path == "" {
		t.Skip("set FAK_STUDYCLASS_CORPUS for the immutable full-corpus smoke")
	}
	corpus, err := studyforge.Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := studyforge.Validate(corpus); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	out, err := Classify(corpus, "sha256:"+hex.EncodeToString(sum[:]))
	if err != nil {
		t.Fatal(err)
	}
	if out.Summary.RecordCount != len(corpus.Records) {
		t.Fatalf("classified %d of %d", out.Summary.RecordCount, len(corpus.Records))
	}
}
