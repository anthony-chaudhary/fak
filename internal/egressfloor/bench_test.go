package egressfloor

import "testing"

func BenchmarkEgressFloor(b *testing.B) {
	blockedArgs := map[string]any{
		"url": "http://169.254.169.254/latest/meta-data/",
	}
	allowedArgs := map[string]any{
		"url": "https://api.github.com/repos/anthony-chaudhary/fak",
	}
	policy := DeliveryPolicy{
		Allow: map[Platform][]string{
			PlatformDiscord: {"#general"},
		},
	}
	del := Delivery{
		Platform:    PlatformDiscord,
		Destination: "#general",
		Message:     "Status update: all systems operational.",
	}

	b.Run("ClassifyBlocked", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			host, label := Classify("web_fetch", blockedArgs)
			if host == "" || label == "" {
				b.Fatal("expected egress block")
			}
		}
	})

	b.Run("ClassifyAllowed", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			host, _ := Classify("web_fetch", allowedArgs)
			if host != "" {
				b.Fatal("expected host to be allowed")
			}
		}
	})

	b.Run("DeliveryAdjudicate", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			rec := policy.Adjudicate(del)
			if !rec.Allowed {
				b.Fatal("expected delivery allowed")
			}
		}
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Classify("web_fetch", blockedArgs)
	}
}
