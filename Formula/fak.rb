# typed: false
# frozen_string_literal: true

# Formula for installing fak from prebuilt releases.
class Fak < Formula
  desc "Fused Agent Kernel: single-binary agent governor, gateway, and runtime"
  homepage "https://github.com/anthony-chaudhary/fak"
  version "0.56.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.56.0/fak_0.56.0_darwin_arm64.tar.gz"
      sha256 "d4912e6f4843927a7305d4ea21e268ae5a048b550f9f480139d45e2589bb6a34"
    else
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.56.0/fak_0.56.0_darwin_amd64.tar.gz"
      sha256 "ec2166b8e851745e44c58c0eccd8853e09d1a5d89618c2f115de45c0a40a11f5"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.56.0/fak_0.56.0_linux_arm64.tar.gz"
      sha256 "be421cfadeacf81294370ae1b3611b1f451d1fd3028f7837d20a8a1c105d78fa"
    else
      url "https://github.com/anthony-chaudhary/fak/releases/download/v0.56.0/fak_0.56.0_linux_amd64.tar.gz"
      sha256 "24b9f58551ff945394e8cc110a30e1b451eae6ed2d0db670089f0052d6d680c9"
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
