# NN-descent MMA benchmark: 20260810T020636Z-smc-522ga-0059-sm120

- Candidate: `4af65973922bbbe0e191ffa24e38c5ee9967b526`
- Main baseline: `be8ab314d044aee0f80fbe2c2277a893561288de` (`origin/main`)
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, compute capability `12.0`, build target `120-real`
- Shape: rows `1000000`; dimensions `16,64,128,256,512,786,1024,1536`; degree `64`; iterations `20`; repeats `5`

| Dim | Candidate TF32 ms | Main FP32 ms | TF32 speedup | Candidate FP16 ms | Main FP16 ms | FP16 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 6419.537 | 6804.919 | 1.060x | 7002.643 | 6416.536 | 0.916x |
| 64 | 7176.009 | 6295.142 | 0.877x | 7181.496 | 6798.112 | 0.947x |
| 128 | 6915.946 | 6649.891 | 0.962x | 6661.432 | 6632.577 | 0.996x |
| 256 | 6933.935 | 7207.889 | 1.040x | 7131.323 | 6578.641 | 0.922x |
| 512 | 7006.795 | 7935.303 | 1.133x | 7256.491 | 7196.382 | 0.992x |
| 786 | 7737.776 | 8472.215 | 1.095x | 7146.495 | 6996.668 | 0.979x |
| 1024 | 7894.968 | 8893.172 | 1.126x | 7398.169 | 6857.357 | 0.927x |
| 1536 | 9483.164 | 10756.561 | 1.134x | 7134.834 | 7388.689 | 1.036x |

## Source handoff

The benchmark was captured with the candidate based at `4af65973922bbbe0e191ffa24e38c5ee9967b526`
and the final source changes present in its working tree. That source tree was subsequently
materialized as these ordered commits on top of `4af65973`:

1. `93457390fd500ff47ff995e5ad4b42c6fcb749f9` — propagate the NN-descent distance dtype through
   the all-neighbors C API.
2. `1b5a7fcfb7c889d9cc341820c8ac92a775d26d0c` — preserve caller NN-descent settings in CAGRA and
   align both graph-degree fields with CAGRA's intermediate graph.
3. `65f8c71fad3bb4720a634493aaad14833a11271a` — select the TF32 backend and CTA size from the
   virtual architecture of the loaded CUDA image.
4. `90fba76d02fb4fd9115a08c3211e8a1392eebcf9` — join the NN-descent sampling worker during
   exception unwinding.
5. `159e904131111400974894ad46f425291838a9dd` — stage adjacent TF32 candidate values with one
   shared-memory `float2` store per lane.

## Changes that worked

- Virtual-image dispatch fixes the SM120 case where the physical device executes an SM100 image:
  the loaded image now determines whether the SM90, SM100, or portable TF32 backend and CTA size
  are selected. The physical architecture is retained as a TF32 availability check.
- Adjacent-pair shared-memory staging was the only portable-kernel experiment retained. Isolated
  local-join profiling measured `1.110x` at dimension 65 and `1.354x` at dimension 1024.
- An alternating same-commit, end-to-end A/B measured paired-median TF32 speedups of `1.016x` at
  dimension 65 and `1.034x` at dimension 1024. Its FP16 control showed no regression signal.
- Against `origin/main` FP32, candidate TF32 was strongest at larger dimensions: `1.133x` at 512,
  `1.095x` at 786, `1.126x` at 1024, and `1.134x` at 1536.
- The C API propagation, CAGRA parameter preservation/degree alignment, and thread joiner are
  correctness fixes rather than performance claims.

## Changes not retained

- Sparse epilogue handling was only about 2.2% faster at dimension 65 and 0.2% at dimension 1024,
  which did not justify duplicating the kernel path.
- Guarded global `float2` loads were slower.
- Tail rounding helped partial tiles but regressed aligned and common dimensions.
- Tile skipping was slower.
- A separate SM120 kernel specialization was not added because of its maintenance cost.
- Candidate TF32 remains slower than main FP32 at dimensions 64 (`0.877x`) and 128 (`0.962x`).
- Candidate-vs-main FP16 is not a patch-isolation control: the revisions differ by multiple
  commits and the protocol executes candidate before main. The alternating same-commit comparison
  did not reproduce an FP16 regression.

## Final validation

- Rebuilt `cuvs`, `cuvs_c`, `NEIGHBORS_ANN_NN_DESCENT_TEST`, and
  `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` from the committed tree.
- Passed 22/22 focused NN-descent dispatch and MMA-boundary tests.
- Passed the CAGRA internal-degree regression test.
- `git diff --check` passed before publication.
