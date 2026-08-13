---
title: "fak — Fused Agent Kernel (pengantar Bahasa Indonesia / Indonesian introduction)"
description: "Halaman masuk fak dalam Bahasa Indonesia: satu Go binary yang duduk di antara agen AI dan tool call-nya — memeriksa setiap tool call sebelum dijalankan, memakai ulang kerja berulang di sesi panjang; self-host, ramah UU PDP, Apache-2.0."
---

# fak — Fused Agent Kernel (pengantar Bahasa Indonesia)

> Ini adalah **halaman masuk (entry point) yang dilokalkan**, bukan terjemahan
> dokumen lengkap. Dokumentasi lengkap berbahasa Inggris — halaman ini memberi
> inti fak, bukti 60 detik, dan jalur instalasi, lalu mengarahkan Anda ke
> [dokumentasi Inggris](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Catatan:** terjemahan ini dibuat oleh mesin dan masih menunggu tinjauan
> penutur asli — bila menemukan kekeliruan, silakan buka issue/PR.
> **Bahasa lain:** lihat [i18n hub](../README.md).

## fak dalam satu kalimat

**fak adalah satu Go binary** yang duduk di antara agen AI Anda dan tool call
yang dipanggilnya — memeriksa setiap tool call *sebelum* dijalankan, dan memakai
ulang kerja bersama yang berulang di sepanjang sesi panjang. Hasilnya: loop agen
yang sama menjadi **lebih aman, lebih murah, dan lebih cepat**, tanpa menulis
ulang apa pun.

Anda tidak menulis ulang agen — cukup arahkan satu base URL ke `fak serve`, dan
setiap tool call lebih dulu melewati capability floor.

```bash
fak manage claude      # membungkus agen Anda yang sudah ada dengan satu perintah
```

## Mengapa tim di Indonesia perlu peduli

- **Tagihan token dalam USD, tapi margin Anda dihitung dalam Rupiah.** Untuk
  fleet yang self-host dan padat pembacaan (read-heavy), fak memakai ulang kerja
  bersama di sesi panjang (KV cache dari system prompt dan daftar tool) —
  **sekitar 4,1× lebih sedikit kerja** dibanding stack warm-cache yang sudah
  di-tuning, diukur pada sesi 50 giliran × 5 agen. (Angka ~60× — sekitar 19 jam
  menjadi sekitar 19 menit — benar *hanya* dibanding pola naif yang mengirim
  ulang semuanya, jadi jangan dijadikan headline.) Ini tuas langsung ke margin
  dalam IDR.
- **Data tetap di dalam negeri (UU PDP 2022).** fak mengutamakan self-host: satu
  static binary yang duduk di depan **model lokal** atau provider domestik, dengan
  residensi yang **fail-closed** di setiap backend, **capability floor
  default-deny**, dan **audit log tamper-evident** untuk setiap tool call. Data
  tidak keluar dari mesin Anda.
- **Tanpa pembayaran lintas batas.** fak berlisensi **Apache-2.0**, gratis, dan
  self-host — tanpa kartu kredit, tanpa invoice lintas negara, tanpa badan hukum.
  `git clone` lalu `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`
  adalah keseluruhan jalurnya.
- **Satu static binary, tanpa dependensi eksternal.** Ops sederhana untuk tim
  kecil — tanpa sidecar, tanpa authorizer terpisah. Dari laptop hingga fleet
  memakai artifact yang sama; Anda menambah flag, bukan komponen.

## Masalah yang diselesaikan fak

- **Sesi panjang tetap murah.** Diskon prompt-cache dari provider hanya bertahan
  bila prefix yang ter-cache tetap sama byte demi byte. fak membuang giliran lama
  di tengah namun **menjaga prefix byte-identical** — fak *menjamin* keidentikan
  byte pada prefix; apakah provider benar-benar memakai ulang cache adalah
  keputusan provider, yang diteruskan fak, bukan diklaim fak.
- **Keamanan default-deny.** Batas kapabilitas diperiksa **di dalam kernel, di
  jalur panggilan yang sama** — **fail-closed**. Aksi yang tidak pernah ada di
  allow-list tidak bisa berjalan, tak peduli model dibujuk untuk apa. Ini
  struktur, bukan classifier.
- **Karantina prompt injection / hasil teracuni.** *Hasil* tool yang mencurigakan
  ditahan sepenuhnya di luar konteks model — lewat struktur, bukan detektor.
  Detektor yang menandainya dianggap ~100% bisa dihindari secara desain: sebuah
  bonus, bukan lantai pertahanan. Pada uji langsung, prompt injection menembus
  baseline tanpa proteksi 5/5; fak menutupnya 5/5.

## Bukti 60 detik (tanpa key, tanpa model, tanpa GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Dengan model Anda

fak tidak mengganti model Anda — ia mengatur (govern) dan meng-cache-nya.
**Qwen2/Qwen3 dan GLM-MoE** terbukti bit-exact di in-kernel reference engine;
selebihnya (DeepSeek, Mistral, model open-weights mana pun) tersambung lewat
OpenAI-compatible wire — melalui Ollama / vLLM / SGLang / llama.cpp / LM Studio
atau API apa pun yang kompatibel dengan OpenAI.

## Langkah berikutnya

- [Mulai cepat (Bahasa Indonesia) — 10 menit menuju model lokal](./quickstart.md)
- [Instalasi (Bahasa Indonesia) — pasang binary-nya](./install.md)
- [FAQ (Bahasa Indonesia) — pertanyaan kontak pertama](./faq.md)
- [README (ikhtisar lengkap)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — model lokal dalam 10 menit](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — instal binary](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — sambungkan agen Anda](../../integrations/README.md)
- [Residensi data & kepatuhan — untuk UU PDP 2022](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — sumber setiap angka](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — apa yang shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
