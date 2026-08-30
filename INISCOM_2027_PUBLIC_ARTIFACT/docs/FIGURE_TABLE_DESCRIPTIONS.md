# Figure and table descriptions

Descriptions use stable LaTeX labels; the PDF supplies numbering. They accompany the anonymous manuscript and are not PDF/UA certification.

## Precision checkpoint heatmap (`fig:pr`)

Eight configurations are rows; six recall checkpoints (80, 90, 95, 97, 99, 99.5 percent) are columns. Cells give ten-seed mean attack precision. Bold marks the column maximum; yellow-to-red shows the deficit from that maximum, with a shared 0–6 percentage-point scale. At recall 90%, Cascade-A has about 96.98% precision versus 94.71% for the three-stage RF route. These are descriptive test-curve summaries, not selected deployment thresholds.

## Conditional score distributions (`fig:shift`)

Four line panels distinguish Normal and Attack. The left column shows cumulative distributions against attack score from 0 to 1: blue is RF-30, orange Cascade-A; dashed is validation and solid is test. The right column shows the signed validation-minus-test cumulative difference for the corresponding class, with a horizontal zero reference. Curves average ten seeds on 1,001 grid points. Within-Normal KS distances are approximately 0.284 for RF-30 and 0.235 for Cascade-A, versus 0.095 and 0.052 within Attack. KS uses full score support and does not establish a unique feature-level shift mechanism.

## RF threshold sensitivity (`fig:thresholdcurve`)

Side-by-side line panels show RF-30 and RF-300 mean F1-W on a fixed cutoff grid from 0.30 to 0.70. Dashed blue is validation, solid orange is test. A solid vertical line marks cutoff 0.5 and a dotted line the mean validation-selected cutoff (about 0.4286 and 0.4146). The selected policy is evaluated using each seed's own threshold, not that mean line. No test optimum is selected.

## Signed cost residuals by input size (`fig:residualsize`)

A twelve-row, two-column heatmap gives mean signed percentage errors at held-out input batches 32 and 1,000. Four component-score rows precede eight whole-policy rows. Positive red means overprediction; negative blue means underprediction; every cell has a signed numeric label. At batch 32, component RF-300 is overpredicted by about 64.6% and XGB-300 underpredicted by 72.3%. Colors share a symmetric scale. Each value averages five seed-specific errors of measured call means; holdout was not used for fitting.

## DRR residuals by actual Stage-3 size (`fig:drrresidual`)

Two bar panels keep input batches 32 and 1,000 separate. Categories are actual Stage-3 rows per call. Adjacent positive bins are merged using only call counts, with at least 10 calls per displayed bin. Batch 32 shows 0, 1 and 2-16 (53, 22 and 25 calls); batch 1,000 shows 0, 1-16, 17-64 and 65 or more (19, 21, 45 and 15 calls). Zero-stage calls remain separate. The vertical axis is whole-call signed percentage error, positive for overprediction. Within-seed errors are recomputed from call-weighted measured and predicted means, then averaged equally across represented seeds. Labels give seeds and calls. All 200 holdout calls are retained; some bins still represent only one seed. These are not stage-specific timings or a causal comparison.

## Quality-cost policies (`fig:qualitycost`)

A compact two-part heatmap compares eight configurations. The left matrix contains three cutoff-policy columns---fixed 0.5, global validation-F1 and oracle prior---and reports mean plus or minus the ten-seed sample SD of test F1-W in every cell. Bold text marks the largest mean in each policy column. The right matrix reports the mean plus or minus SD native-policy reference time once per configuration; its color is logarithmically normalized and bold marks the fastest configuration. Exact cell text remains readable without color. Timing is reused across a configuration's cutoff-only decisions and therefore does not claim policy-specific latency measurements. The figure is a descriptive quality--cost summary rather than a universal utility ranking.

## Information roles (`tab:roles`)

Columns identify the analysis, information used for selection, and interpretation. Archived primary, default, oracle-prior, constrained-transfer, score/category and timing diagnostics are distinguished. Target prevalence in the oracle policy comes from test labels.

## Default-cutoff comparison (`tab:defaultcost`)

Eight models have F1-W mean and sample SD, mean FAR/recall, and native-policy reference time. RF-30 has higher F1-W and lower reference time than Cascade-A, but lower recall. DT, DR, DRR, RF-30 and RF-300 are nondominated on the two displayed mean axes. The note distinguishes reference timing from an unperformed standalone-0.5 timing run.

## Validation-selected global cutoffs (`tab:threshold`)

Each of eight models has two rows: Global val-F1 cutoff and oracle-prior-weighted validation-F1, with cutoff and test F1-W mean ± sample SD plus mean FAR and recall over ten seeds. The eight fixed-0.5 rows are not repeated: they remain in the default-cutoff comparison (`tab:defaultcost`), so both tables together cover all three policies and all displayed quality metrics. RF-30's selected cutoffs are 0.4286 ± 0.0206 and about 0.5057 ± 0.0173. Global 0.5 reproduces native route decisions but does not equalize gates or calibration. No best route exceeds the best standalone mean F1-W within a policy; the oracle-prior best route is Cascade-A (89.510%), not DRR.

## Matched class-weight ablation (`tab:classweight`)

Six rows cross RF-30 balanced-subsample versus no class weights with three cutoff policies. Columns show cutoff and test weighted F1 mean ± sample SD, false-alarm rate and attack recall, across ten matched seeds. At fixed 0.5, removing weights reduces F1-W from 90.234% to 80.634% and raises FAR from 17.419% to 40.622%. The mean validation cutoff moves from 0.4286 to 0.5901; the mean oracle-prior cutoff moves from 0.5057 to 0.6929. The default-minus-validation-cutoff F1-W difference reverses from +3.486 to −5.148 percentage points. These are training-configuration and operating-point comparisons on one reused test split, not a new timing or route-refit experiment.

## Same-pool random-split control (`tab:randomsplit`)

Three rows report RF-30 at fixed 0.5, global validation-F1 and oracle-prior cutoffs across ten stratified 64/16/20 partitions of the official training pool. Columns give cutoff, validation/test F1-W and validation/test FAR. The global validation cutoff averages 0.4427; validation/test F1-W is 94.47%/94.47% and FAR is 9.64%/9.65%. Oracle and ordinary validation selections coincide because their class priors are nearly identical. Mean Normal and Attack KS distances are 0.011 and 0.008. This is a post-hoc same-pool control, not an external-deployment test.

## Constraint transfer (`tab:transfer`)

The compact main table has eight rows and three validation-constraint columns: FAR at most 10%, FAR at most 20%, recall at least 97%. Cells are achieved test FAR / recall percentages. No model under the first rule achieves mean test FAR at most 10%. The reproducibility CSV retains validation and test FAR/recall for all three rules. These do not prove no suitable deployment operating point exists.

## Probability diagnostics (`tab:calibrationfull`)

Rows distinguish raw validation, raw test and oracle-prior-adjusted test scores. Columns are Brier loss, natural-log loss and 15-bin expected calibration error. Brier and ECE are in [0,1]; log loss is unbounded; these are not percentages. The appendix table covers all eight models; the main text reports the Cascade-A contrast without duplicating a subset table.

## Category routing (`tab:categories`)

Ten categories have fixed counts, Stage-3 reach and native attack recall for Cascade-A and DRR. Reach averages all ten seeds including zero-stage cases. Normal recall is undefined and shown as a dash. Tiny positive reach is shown as less than 0.01%, not zero. Fuzzers has 6,062 rows; recall is 88.84% for Cascade-A and 80.46% for DRR. Stage reach is not category composition within a stage.

## Batch latency (`tab:batch`)

Eight configurations are compared at batches 1, 100, 10,000 and full test. Units are microseconds per flow, averaged from five seed-specific call means. At batch 100, XGB-300 has 7.678 versus Cascade-A 14.272. DRR at batch one has mean-summary 594.09 versus median-summary 56.75. Batch and threading change cost ordering.

## Retained paired inference and native quality (`tab:paired`, `tab:quality`)

All seed-paired intervals condition on the fixed official datasets, reflecting training/validation-split randomness rather than new deployments. Row-bootstrap sensitivity appears separately in `tab:rowbootstrap`.

The original four contrasts retain their means, paired-t intervals and Holm sensitivities. RF-30 minus Cascade-A remains primary; tight equivalence is not established. Two additional unadjusted post-hoc contrasts are separated from those families: RF-30@0.5 minus Cascade-A is 3.007 points, 95% CI [2.858, 3.157]; RF-300@0.5 minus RF-30@0.5 is 0.122, CI [-0.134, 0.378]. The complete original quality table gives F1-W, FAR, recall and average precision at native validation-selected policies, separate from the global cutoff comparison. For standalone models, validation selects the global classification cutoff; for routes it selects internal routing gates, while terminal experts retain cutoff 0.5.

## Exact threshold-band accounting (`tab:thresholdband`)

RF-30 and RF-300 have validation/test rows. Counts are mean Normal and Attack samples in [validation cutoff, 0.5), followed by changes in F1-W, FAR and recall. Raising RF-30's cutoff removes 3,371.6 test false positives and adds 687 false negatives on average, increasing test F1-W by 3.486 points while reducing validation F1-W by 0.307 points. Fractional counts arise only from averaging integer counts across seeds.

## Full-test and parallel timing (`tab:cost`, `tab:parallel`)

Full-test timing gives mean call time, mean seed median with sample SD, and a ratio to Cascade-A across ten seeds. Parallel timing compares one versus eight model workers using the same five sensitivity seeds, with mean paired speedup. BLAS stays single-threaded. The two DT-to-RF routes slow down under eight workers.

## Held-out cost adequacy (`tab:overhead`)

Four component and eight whole-policy rows give mean and maximum absolute percentage error across five seeds and two holdout sizes. All fail the fixed criteria of MAPE at most 20% and maximum error at most 50%. Cascade-A's values are 24.80% and 52.62%. This does not invalidate the measured benchmark or imply every cost model must fail.

## Secondary regime (`tab:cic`)

Eight models report attack F1 with SD, weighted F1, FAR and time on five random CIC-UNSW splits. Test rows vary across seeds. XGB-300 and Cascade-A both round to 95.92% attack F1. This UNSW-derived regime is not independent external deployment validation.

## Feature-unique policies (`tab:unique`)

Twenty-four rows cross eight models with fixed 0.5, global validation-F1 and oracle-prior policies. Columns give F1-W mean and SD, unique-minus-full F1-W difference, mean FAR and recall. The common subset has 73,791 rows: 35,528 Normal and 38,263 Attack. All within-policy mean rankings remain unchanged; all mean F1-W values decrease. Cutoffs, gates and the original full-test oracle prior are frozen. This is not retraining after deduplication; repeated rows within test remain.

## Separate conditional intervals (`tab:rowbootstrap`)

Four contrasts on full test and four on the feature-unique subset give mean differences and 95% row-bootstrap intervals. Full-test rows also show the original seed-paired t intervals. Every bootstrap draw resamples rows jointly across all ten fitted systems and models; no seed or cutoff is resampled. RF-30@0.5 minus Cascade-A has mean 3.007, seed CI [2.858, 3.157] and row CI [2.861, 3.150]. On unique-only rows its mean is 3.241 and row CI [3.078, 3.404]. For the primary contrast and RF-300-minus-RF-30 default contrast, row CIs exclude zero while seed CIs include zero: different conditioning, not replacement of the primary inference. The bootstrap assumes IID rows and does not capture session dependence or new deployments.
