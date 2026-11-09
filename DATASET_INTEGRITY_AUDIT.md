# Dataset integrity audit — 9 November 2026

**Scope:** Local copies of two road-feature image/label splits. Original data were not modified. The author-confirmed final YOLO26n training configuration refers to **split_dataset2**.

| Read-only check | split_dataset2 | split_dataset3 |
| --- | ---: | ---: |
| Train images | 1,996 | 1,996 |
| Validation images | 142 | 570 |
| Test images | 0 | 286 |
| Distinct image SHA-256 values | 1,797 | 2,274 |
| Exact-image duplicate groups across splits | 37 | 233 |
| Groups with conflicting annotation text for identical image bytes | 275 | 450 |
| Annotation rows flagged for geometry/format review | 98 | 134 |
| Validation labels without image | 21 | 0 |
| Validation images without label | 1 | 0 |

## Method

For every image with a corresponding TXT annotation in each local split, SHA-256 was calculated over raw image bytes. Annotation rows were checked for five fields, valid five-class IDs, finite normalized values, positive box dimensions and box bounds within the 0–1 image area. Duplicate image hashes spanning train, validation or test were flagged; repeated images with different sorted annotation rows were marked as conflicting. Files were only read, never altered. The complete report was saved locally to `D:/personal_projects/autonomous_vehicle/verification_2026-11-09/dataset_audit.json`.

These are **flags** for inspection, not assertions that every annotation is incorrectly labeled. The audit examines exact-byte duplicates, not perceptual duplicates, near-identical video frames, or independence by original scene/recording. A repeated image may have different but equally legitimate boxes; it needs human adjudication.

## Consequences

Neither supplied split is ready to be presented as a clean source-independent benchmark. The saved original YOLO26n results must be described as historical *validation* results only. No new corrected-dataset metrics or retraining results have been generated.

## Safe correction plan

1. Recover provenance for each source image/video and manually resolve uncertain left/right road-boundary definitions.
2. Review every flagged annotation and duplicated-image conflict; retain a provenance trail rather than silently overwriting originals.
3. Construct separate **new** train, validation and test directories, grouping duplicate/near-duplicate and source-sequence images together before assigning splits.
4. Run a new audit to ensure zero exact cross-split image duplicates, valid label geometry and no missing pairs.
5. Train a newly named YOLO26n experiment with `fliplr=0.0`, unless augmentation explicitly swaps left/right class labels.
6. Evaluate on untouched source-disjoint test data; publish recorded settings, checkpoints, aggregate and class-specific metrics.

The historical run and its original artifacts must remain unchanged.
