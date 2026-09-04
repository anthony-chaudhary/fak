package deletioncert

import (
	"crypto/ed25519"
	"testing"
)

func deterministicBenchmarkKey() ed25519.PrivateKey {
	seed := make([]byte, ed25519.SeedSize)
	for i := range seed {
		seed[i] = byte(i + 1)
	}
	return ed25519.NewKeyFromSeed(seed)
}

func BenchmarkDeletionCert(b *testing.B) {
	priv := deterministicBenchmarkKey()
	certInput := baseCert()
	journal := goodJournal()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cert, err := Mint(priv, certInput)
		if err != nil {
			b.Fatal(err)
		}
		result := Verify(cert, journal)
		if !result.Valid {
			b.Fatalf("expected valid certificate: %+v", result)
		}
		encoded, err := cert.Marshal()
		if err != nil {
			b.Fatal(err)
		}
		parsed, err := Parse(encoded)
		if err != nil {
			b.Fatal(err)
		}
		if parsed.Signature != cert.Signature {
			b.Fatal("signature mismatch after parse")
		}
	}
}

func BenchmarkMint(b *testing.B) {
	priv := deterministicBenchmarkKey()
	certInput := baseCert()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cert, err := Mint(priv, certInput)
		if err != nil {
			b.Fatal(err)
		}
		if cert.Signature == "" {
			b.Fatal("empty signature")
		}
	}
}

func BenchmarkVerify(b *testing.B) {
	priv := deterministicBenchmarkKey()
	cert, err := Mint(priv, baseCert())
	if err != nil {
		b.Fatal(err)
	}
	journal := goodJournal()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Verify(cert, journal)
		if !res.Valid {
			b.Fatalf("verify failed: %+v", res)
		}
	}
}

func BenchmarkMarshal(b *testing.B) {
	priv := deterministicBenchmarkKey()
	cert, err := Mint(priv, baseCert())
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := cert.Marshal()
		if err != nil {
			b.Fatal(err)
		}
		if len(data) == 0 {
			b.Fatal("empty marshaled data")
		}
	}
}

func BenchmarkParse(b *testing.B) {
	priv := deterministicBenchmarkKey()
	cert, err := Mint(priv, baseCert())
	if err != nil {
		b.Fatal(err)
	}
	data, err := cert.Marshal()
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parsed, err := Parse(data)
		if err != nil {
			b.Fatal(err)
		}
		if parsed.Schema != SchemaVersion {
			b.Fatal("unexpected schema")
		}
	}
}

func TestBenchmarkDeletionCertExecution(t *testing.T) {
	priv := deterministicBenchmarkKey()
	cert, err := Mint(priv, baseCert())
	if err != nil {
		t.Fatalf("Mint failed: %v", err)
	}
	res := Verify(cert, goodJournal())
	if !res.Valid {
		t.Fatalf("Verify failed: %+v", res)
	}
	data, err := cert.Marshal()
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	parsed, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	if parsed.Signature != cert.Signature {
		t.Fatalf("signature mismatch: %s vs %s", parsed.Signature, cert.Signature)
	}
}
