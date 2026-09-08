# typed: false
# frozen_string_literal: true

# Formula for installing fak from prebuilt releases.
class Fak < Formula
  desc "Fused Agent Kernel: single-binary agent governor, gateway, and runtime"
  homepage "https://github.com/anthony-chaudhary/fak"
  version "0.53.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.53.0/fak_0.53.0_darwin_arm64.tar.gz"
      sha256 "dcaef3674b6ce59a32e320e54c88ac8d6165de9bb60ad767c9c56fd91a68c183"
    else
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.53.0/fak_0.53.0_darwin_amd64.tar.gz"
      sha256 "5385d297158c7479c84dfcc0cb4f12a1e7e020046d30b87eb6e618557b1d44e1"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.53.0/fak_0.53.0_linux_arm64.tar.gz"
      sha256 "10762f52559696efe99aa83e74964bce8ef3c7189dfd5b111658fb0ecfee78d8"
    else
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.53.0/fak_0.53.0_linux_amd64.tar.gz"
      sha256 "aa11869cc9e00c766ff7894a674661ef7fcae71603d3ff7f65bb8e88647bf23e"
    end
  end

  def install
    bin.install "fak"
    generate_completions_from_executable(bin/"fak", "completion")
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/fak version")
  end
end
