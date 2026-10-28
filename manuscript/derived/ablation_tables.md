<!-- generated from run 20261026T075410Z-e78b8cf6; do not edit by hand -->

### G. Gold sets (own-citation gold, one per question)

- Questions: 933; gold-set size mean 6.32, median 6, range 2–14
- Distinct chunks in any gold set: 51; distinct gold sets: 488

| Area | n | Distinct chunks | Gold size mean | Gold size median | Most-cited chunk in % of gold sets |
|---|---|---|---|---|---|
| Assessment | 234 | 18 | 6.69 | 7 | 89.32 |
| Investigations | 233 | 14 | 5.75 | 6 | 99.14 |
| Severity classification | 233 | 17 | 5.11 | 5 | 96.57 |
| Antibiotic course | 233 | 28 | 7.73 | 8 | 93.99 |

### A. MRR@10 by arm and BM25 weight w (mean, 95% CI), n = 933

| Arm | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Present only, raw | 0.365 (0.346 to 0.384) | 0.467 (0.452 to 0.484) | 0.703 (0.685 to 0.721) | 0.809 (0.792 to 0.826) | 0.870 (0.855 to 0.886) | 0.872 (0.857 to 0.888) |
| Present only, enriched | 0.377 (0.359 to 0.395) | 0.481 (0.465 to 0.496) | 0.696 (0.677 to 0.714) | 0.864 (0.848 to 0.880) | 0.865 (0.849 to 0.881) | 0.862 (0.846 to 0.878) |
| All assessed, raw | 0.368 (0.350 to 0.386) | 0.492 (0.477 to 0.507) | 0.697 (0.679 to 0.715) | 0.810 (0.792 to 0.827) | 0.868 (0.852 to 0.884) | 0.863 (0.848 to 0.879) |
| All assessed, enriched | 0.396 (0.378 to 0.414) | 0.496 (0.481 to 0.511) | 0.697 (0.679 to 0.716) | 0.856 (0.840 to 0.872) | 0.860 (0.844 to 0.876) | 0.859 (0.842 to 0.875) |

Pooled over the four arms (each question averaged over arms first):

| | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| All arms | 0.376 (0.358 to 0.395) | 0.484 (0.469 to 0.500) | 0.698 (0.683 to 0.714) | 0.835 (0.820 to 0.849) | 0.866 (0.851 to 0.881) | 0.864 (0.849 to 0.880) |

### B. Paired contrasts, MRR@10 (mean difference, 95% CI | p)

| Contrast | Pooled over | Difference (95% CI) | p |
|---|---|---|---|
| Level 1: present only − all assessed | L2 × w | -0.003 (-0.005 to +0.000) | 0.089 |
| Level 2: enriched − raw, present only | w | +0.010 (+0.005 to +0.014) | <0.001 |
| Level 2: enriched − raw, all assessed | w | +0.011 (+0.005 to +0.017) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, Present only, raw | — | +0.508 (+0.483 to +0.532) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, Present only, enriched | — | +0.485 (+0.461 to +0.509) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, All assessed, raw | — | +0.496 (+0.471 to +0.520) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, All assessed, enriched | — | +0.463 (+0.439 to +0.487) | <0.001 |
| Post hoc: best w (= 0.8, chosen on these data) − w = 1.0 | L1 × L2 | +0.002 (-0.001 to +0.004) | 0.190 |

### C. Arms vs the popularity baseline, MRR@10

Baseline (query-blind, leave-one-out): 0.803 (0.785 to 0.820)

| Arm | w | Arm − baseline (95% CI) | p |
|---|---|---|---|
| Present only, raw | 0.0 | -0.438 (-0.462 to -0.414) | <0.001 |
| Present only, raw | 0.2 | -0.335 (-0.355 to -0.316) | <0.001 |
| Present only, raw | 0.4 | -0.100 (-0.120 to -0.079) | <0.001 |
| Present only, raw | 0.6 | +0.007 (-0.013 to +0.027) | 0.512 |
| Present only, raw | 0.8 | +0.068 (+0.045 to +0.090) | <0.001 |
| Present only, raw | 1.0 | +0.070 (+0.048 to +0.092) | <0.001 |
| Present only, enriched | 0.0 | -0.426 (-0.447 to -0.404) | <0.001 |
| Present only, enriched | 0.2 | -0.322 (-0.340 to -0.303) | <0.001 |
| Present only, enriched | 0.4 | -0.107 (-0.130 to -0.085) | <0.001 |
| Present only, enriched | 0.6 | +0.062 (+0.039 to +0.085) | <0.001 |
| Present only, enriched | 0.8 | +0.062 (+0.039 to +0.085) | <0.001 |
| Present only, enriched | 1.0 | +0.059 (+0.036 to +0.083) | <0.001 |
| All assessed, raw | 0.0 | -0.435 (-0.458 to -0.412) | <0.001 |
| All assessed, raw | 0.2 | -0.311 (-0.328 to -0.294) | <0.001 |
| All assessed, raw | 0.4 | -0.105 (-0.125 to -0.085) | <0.001 |
| All assessed, raw | 0.6 | +0.007 (-0.013 to +0.027) | 0.507 |
| All assessed, raw | 0.8 | +0.066 (+0.043 to +0.089) | <0.001 |
| All assessed, raw | 1.0 | +0.061 (+0.038 to +0.084) | <0.001 |
| All assessed, enriched | 0.0 | -0.407 (-0.427 to -0.385) | <0.001 |
| All assessed, enriched | 0.2 | -0.307 (-0.324 to -0.289) | <0.001 |
| All assessed, enriched | 0.4 | -0.105 (-0.125 to -0.085) | <0.001 |
| All assessed, enriched | 0.6 | +0.053 (+0.032 to +0.075) | <0.001 |
| All assessed, enriched | 0.8 | +0.057 (+0.034 to +0.080) | <0.001 |
| All assessed, enriched | 1.0 | +0.056 (+0.033 to +0.080) | <0.001 |

### D. By query area, MRR@10 (exploratory)

Pooled over the four arms, by w (mean, 95% CI):

| Area | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Assessment | 0.733 (0.698 to 0.766) | 0.732 (0.699 to 0.766) | 0.732 (0.699 to 0.766) | 0.731 (0.697 to 0.764) | 0.725 (0.691 to 0.760) | 0.716 (0.681 to 0.751) |
| Investigations | 0.244 (0.230 to 0.259) | 0.437 (0.425 to 0.449) | 0.860 (0.836 to 0.882) | 0.933 (0.910 to 0.954) | 0.942 (0.919 to 0.963) | 0.945 (0.921 to 0.966) |
| Severity classification | 0.327 (0.303 to 0.353) | 0.502 (0.485 to 0.519) | 0.718 (0.690 to 0.745) | 0.937 (0.916 to 0.957) | 0.937 (0.917 to 0.957) | 0.931 (0.910 to 0.951) |
| Antibiotic course | 0.199 (0.183 to 0.215) | 0.264 (0.249 to 0.279) | 0.483 (0.460 to 0.506) | 0.738 (0.709 to 0.767) | 0.860 (0.826 to 0.892) | 0.866 (0.834 to 0.895) |

| Area | L1: present − all assessed | p | L2 enriched − raw (present only) | p | L2 enriched − raw (all assessed) | p |
|---|---|---|---|---|---|---|
| Assessment | +0.003 (+0.000 to +0.006) | 0.027 | -0.008 (-0.011 to -0.005) | <0.001 | -0.008 (-0.013 to -0.003) | 0.001 |
| Investigations | -0.015 (-0.024 to -0.007) | <0.001 | -0.023 (-0.033 to -0.014) | <0.001 | -0.009 (-0.021 to +0.003) | 0.145 |
| Severity classification | +0.002 (-0.005 to +0.009) | 0.564 | +0.013 (-0.001 to +0.027) | 0.073 | +0.004 (-0.012 to +0.019) | 0.624 |
| Antibiotic course | +0.000 (+0.000 to +0.000) | 1.000 | +0.057 (+0.050 to +0.064) | <0.001 | +0.057 (+0.050 to +0.064) | <0.001 |

### E. Sensitivity: PI-curated area reference sets as gold, MRR@10 (mean, 95% CI)

| Arm | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Present only, raw | 0.666 (0.644 to 0.688) | 0.714 (0.695 to 0.733) | 0.878 (0.864 to 0.892) | 0.919 (0.907 to 0.931) | 0.999 (0.998 to 1.000) | 0.999 (0.998 to 1.000) |
| Present only, enriched | 0.692 (0.672 to 0.713) | 0.762 (0.745 to 0.780) | 0.879 (0.865 to 0.893) | 0.995 (0.992 to 0.998) | 0.993 (0.989 to 0.997) | 0.991 (0.987 to 0.995) |
| All assessed, raw | 0.656 (0.634 to 0.679) | 0.727 (0.709 to 0.744) | 0.874 (0.860 to 0.888) | 0.920 (0.908 to 0.931) | 0.998 (0.996 to 0.999) | 0.994 (0.990 to 0.997) |
| All assessed, enriched | 0.699 (0.679 to 0.720) | 0.767 (0.749 to 0.784) | 0.889 (0.875 to 0.902) | 0.989 (0.984 to 0.993) | 0.988 (0.983 to 0.993) | 0.987 (0.982 to 0.992) |

### F. Sensitivity: MRR across K (from `arm_vs_popularity_baseline.csv`)

| K | Baseline MRR | Arm MRR range, w ≥ 0.6 | Best arm (MRR) | Arms above / below / not different from baseline (of 24) |
|---|---|---|---|---|
| 2 | 0.787 | 0.783–0.859 | Present only, raw, w = 1.0 (0.859) | 10 / 12 / 2 |
| 4 | 0.797 | 0.807–0.869 | Present only, raw, w = 1.0 (0.869) | 10 / 12 / 2 |
| 6 | 0.801 | 0.809–0.872 | Present only, raw, w = 1.0 (0.872) | 10 / 12 / 2 |
| 8 | 0.802 | 0.809–0.872 | Present only, raw, w = 1.0 (0.872) | 10 / 12 / 2 |
| 10 | 0.803 | 0.809–0.872 | Present only, raw, w = 1.0 (0.872) | 10 / 12 / 2 |
| 12 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 14 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 16 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 18 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 20 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |

### H. Secondary metric: Recall@10 by arm and w (mean, 95% CI)

| Arm | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Present only, raw | 0.280 (0.270 to 0.290) | 0.548 (0.536 to 0.559) | 0.602 (0.591 to 0.614) | 0.612 (0.601 to 0.623) | 0.594 (0.583 to 0.604) | 0.590 (0.579 to 0.601) |
| Present only, enriched | 0.320 (0.309 to 0.332) | 0.578 (0.566 to 0.590) | 0.619 (0.608 to 0.631) | 0.635 (0.624 to 0.646) | 0.626 (0.615 to 0.638) | 0.613 (0.601 to 0.624) |
| All assessed, raw | 0.293 (0.282 to 0.303) | 0.535 (0.524 to 0.545) | 0.625 (0.614 to 0.637) | 0.641 (0.631 to 0.652) | 0.632 (0.622 to 0.643) | 0.621 (0.610 to 0.633) |
| All assessed, enriched | 0.329 (0.317 to 0.341) | 0.566 (0.554 to 0.577) | 0.645 (0.633 to 0.657) | 0.663 (0.652 to 0.674) | 0.649 (0.637 to 0.660) | 0.638 (0.626 to 0.649) |
