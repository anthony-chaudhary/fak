package computebuild

import (
	"testing"
)

// ParseCUDAArchs parses architecture declarations and generates nvcc -gencode flags.
func ParseCUDAArchs(archText string, targetArch string) ([]string, string, error) {
	return ParseCUDAArchMatrix(archText, targetArch)
}

var (
	benchGencodeSink   []string
	benchBuildArchSink string
	benchToolchainSink *Toolchain
	benchEnvSink       map[string]string
)

func BenchmarkParseCUDAArchs(b *testing.B) {
	const sampleArchMatrix = "sm_80\nsm_89\nsm_90\nsm_100\nsm_120\n"
	const crlfArchMatrix = "sm_80\r\nsm_89\r\nsm_90\r\nsm_100\r\nsm_120\r\n"

	b.Run("FatbinAll", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			gencode, buildArchs, err := ParseCUDAArchs(sampleArchMatrix, "")
			if err != nil {
				b.Fatalf("ParseCUDAArchs failed: %v", err)
			}
			benchGencodeSink = gencode
			benchBuildArchSink = buildArchs
		}
	})

	b.Run("SingleTargetPrefixed", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			gencode, buildArchs, err := ParseCUDAArchs(sampleArchMatrix, "sm_89")
			if err != nil {
				b.Fatalf("ParseCUDAArchs failed: %v", err)
			}
			benchGencodeSink = gencode
			benchBuildArchSink = buildArchs
		}
	})

	b.Run("SingleTargetNumeric", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			gencode, buildArchs, err := ParseCUDAArchs(sampleArchMatrix, "90")
			if err != nil {
				b.Fatalf("ParseCUDAArchs failed: %v", err)
			}
			benchGencodeSink = gencode
			benchBuildArchSink = buildArchs
		}
	})

	b.Run("CRLFMatrix", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			gencode, buildArchs, err := ParseCUDAArchs(crlfArchMatrix, "")
			if err != nil {
				b.Fatalf("ParseCUDAArchs failed: %v", err)
			}
			benchGencodeSink = gencode
			benchBuildArchSink = buildArchs
		}
	})
}

func BenchmarkToolchainDiscovery(b *testing.B) {
	b.Run("DiscoverToolchain", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			tc, err := DiscoverToolchain()
			if err != nil {
				b.Fatalf("DiscoverToolchain failed: %v", err)
			}
			benchToolchainSink = tc
		}
	})

	tc, err := DiscoverToolchain()
	if err != nil {
		b.Fatalf("setup DiscoverToolchain failed: %v", err)
	}

	b.Run("SynthesizeCgoEnv", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			vkEnv := SynthesizeVulkanCgoEnv(tc, ".")
			cudaEnv := SynthesizeCUDACgoEnv(tc, ".", false, nil)
			benchEnvSink = vkEnv
			if cudaEnv != nil {
				benchEnvSink = cudaEnv
			}
		}
	})

	overrides := &Toolchain{
		CXX: "custom-g++",
		CC:  "custom-gcc",
	}
	b.Run("MergeToolchainOverrides", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			merged := mergeToolchainOverrides(tc, overrides)
			benchToolchainSink = merged
		}
	})

	rawEnv := []byte("CC=gcc\nCXX=g++\nCUDA_PATH=C:\\CUDA\nVULKAN_SDK=C:\\VulkanSDK\nPATH=C:\\bin\n")
	b.Run("ParseEnvBlock", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			env := ParseEnvBlock(rawEnv)
			benchEnvSink = env
		}
	})

	b.Run("CompareVersionStrings", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			cmp := compareVersionStrings("1.4.350.0", "1.3.290.0")
			if cmp <= 0 {
				b.Fatal("unexpected compareVersionStrings result")
			}
		}
	})
}
