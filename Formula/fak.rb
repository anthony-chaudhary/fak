# typed: false
# frozen_string_literal: true

# Formula for installing fak from prebuilt releases.
class Fak < Formula
  desc "Fused Agent Kernel: single-binary agent governor, gateway, and runtime"
  homepage "https://github.com/anthony-chaudhary/fak"
  version "0.59.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.59.0/fak_0.59.0_darwin_arm64.tar.gz"
      sha256 "9eea040183190962dc5274ee1a35edb1499f920cb03c8ebe1a3b5c66de467c1f"
    else
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.59.0/fak_0.59.0_darwin_amd64.tar.gz"
      sha256 "8c0b5245c99ccfb86e3f1c9ca9beb4087cf35b9ce7e12c7b585afcfe90a1d7bc"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.59.0/fak_0.59.0_linux_arm64.tar.gz"
      sha256 "ed59e019a8d5ad5b7e7b53575d8dbe92fba3b9b87720cc976f13c7a1b43076ac"
    else
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.59.0/fak_0.59.0_linux_amd64.tar.gz"
      sha256 "71f7a55e39a8dd0b49493f5c3ca8e09ee7a447a8be1028422a971d06382c5262"
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
