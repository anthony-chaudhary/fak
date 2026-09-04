package edgequal

import (
	"encoding/json"
	"testing"
)

func BenchmarkPackBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = PackBytes()
	}
}

func BenchmarkPackSHA256(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = PackSHA256()
	}
}

func BenchmarkParse(b *testing.B) {
	r := validReceipt("android_arm64_phone")
	raw, err := json.Marshal(r)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Parse(raw)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkValidatePass(b *testing.B) {
	r := validReceipt("android_arm64_phone")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(r); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkValidateRefusal(b *testing.B) {
	r := validReceipt("android_arm64_phone")
	r.Status = "refused"
	r.RefusalCode = "OOM"
	r.Metrics = Metrics{}
	r.Cases = nil
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(r); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkValidatePair(b *testing.B) {
	phone := validReceipt("android_arm64_phone")
	laptop := validReceipt("laptop_8gib")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidatePair(phone, laptop); err != nil {
			b.Fatal(err)
		}
	}
}
