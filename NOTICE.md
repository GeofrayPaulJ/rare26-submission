# NOTICE

## GastroNet-5M weights are not redistributed here

This repository does not include, and this release does not redistribute,
any of the GastroNet-pretrained checkpoint files (`weights/*.pth` in the
source tree, e.g. the `RN50_GastroNet-*` and `RN50_Billion-Scale-SWSL+GastroNet-5M`
files). Per the GastroNet Data Use Agreement, section 5, redistribution of
these third-party weights (or fine-tuned derivatives of them) requires
authorisation from whoever holds the DUA; this repository does not attempt
to make that determination and ships no copy of the weights, fine-tuned or
otherwise, in any form.

What this release does carry instead:

- The code path that loads and consumes a GastroNet checkpoint (see
  `src/`), so the loading recipe is fully inspectable.

Note also: an internal review (`reports/member_budget.md`, section 5e)
flagged that the DUA §5 redistribution question for any *fine-tuned
derivative* baked into a shipped model was never resolved by the DUA
holder. That question does not affect this release (no such derivative
weights are included here either), but it is unresolved upstream and is
noted here for completeness.

## No RARE25/RARE26 patient imagery is included

No endoscopy images, image crops, or other pixel data from the RARE25 or
RARE26 datasets (or the EVC external-validation set) are included in this
release. Files under `manifests/` are limited to metadata (file paths,
cryptographic and perceptual hashes, fold assignments, aggregate
per-image statistics such as mean channel values, and QA/review flags) --
never raw pixel content. `reports/` retains only plots and a synthetic
augmentation-recipe listing; figures composed from real patient frames
were excluded (see the release staging report for the itemised list).
