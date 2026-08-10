# SM90 post-redesign benchmark results

These results measure the final pipelined SM90 WGMMA design committed as
`362379b2`. The comparison kernel is the preceding dual-warpgroup design from
`e063da6a`; FP32 SIMT is measured from the same tuned source tree.

The benchmark used one H200 NVL, CUDA 13.3, 1,000,000 rows, graph degree 64,
20 NN-descent iterations, and three timed repetitions per point. A speedup
greater than 1.0 means the tuned design is faster.

| Dimension | Tuned TF32 (ms) | Dual-WG TF32 (ms) | TF32 speedup | Tuned FP16 (ms) | Dual-WG FP16 (ms) | FP16 speedup | FP32 SIMT (ms) | TF32 vs FP32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 6580.001 | 6768.272 | 1.02861x | 6675.772 | 6655.985 | 0.99704x | 7080.749 | 1.07610x |
| 64 | 6686.010 | 6837.043 | 1.02259x | 6713.251 | 6711.712 | 0.99977x | 7063.341 | 1.05644x |
| 128 | 6761.062 | 6869.578 | 1.01605x | 6802.413 | 6761.847 | 0.99404x | 7282.270 | 1.07709x |
| 256 | 7066.412 | 7171.370 | 1.01485x | 6866.430 | 6979.093 | 1.01641x | 7496.543 | 1.06087x |
| 512 | 7296.783 | 7500.180 | 1.02787x | 7098.261 | 7214.038 | 1.01631x | 8318.856 | 1.14007x |
| 786 | 7615.376 | 7966.799 | 1.04615x | 7317.020 | 7563.281 | 1.03366x | 9173.904 | 1.20466x |
| 1024 | 7648.509 | 8178.669 | 1.06932x | 7412.773 | 7652.518 | 1.03234x | 10142.721 | 1.32610x |
| 1536 | 8107.263 | 8770.372 | 1.08179x | 7542.120 | 7922.975 | 1.05050x | 12255.611 | 1.51168x |
| Geometric mean | | | **1.03814x** | | | **1.01733x** | | **1.17268x** |

The final redesign improves TF32 over the preceding SM90 kernel at every
dimension, with a 3.81% geometric-mean gain. TF32 is 17.27% faster than FP32
SIMT geometrically and reaches 1.51x at dimension 1536. FP16 is essentially
flat at the three smallest dimensions, then improves by 1.6% to 5.1%.

Files:

- `tuned.csv`: raw tuned TF32 and FP16 samples.
- `dual-warpgroups-baseline.csv`: raw samples from the preceding SM90 design.
- `fp32-simt.csv`: raw FP32 SIMT comparison samples; the driver also emitted its FP16 mode.
- `summary.csv`: machine-readable medians and speedups.
- `tuned-resource-usage.txt`: resource records from the exact measured library.
