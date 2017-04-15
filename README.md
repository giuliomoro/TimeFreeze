FFT-based time freeze, runs on Bela (github.com/BelaPlatform/Bela).

CPU usage of the `fft-calculation` thread when using the following in the re-synthesis stage:
with cosf/sinf: 37.3%
with cosf_neon/sinf_neon: 21%
with sinfv_c: 29.5%
with sinfv_neon : 15%

