---
status: accepted
---

# Rossmann as the reference system

BANA 7075 follows one end-to-end ML system all term. We chose a simplified course template derived from [rossmann-forecasting](https://github.com/bradleyboehmke/rossmann-forecasting) `main` instead of extending the course's own apple-demand ModelOps code, because Rossmann already has the data pipeline, validation, tests, CI, serving, and monitoring the course teaches, plus three maturity branches that show the same project at increasing sophistication. The apple-demand stack would have needed most of those parts written from scratch. The template is a separate repository so the original Rossmann repository and its branches stay unchanged for Week 1.

## Considered Options

- **Modernized apple-demand stack:** smaller, with a built-in drift generator, but no pipeline, validation, tests, or CI. Kept as Build It labs and a supplementary example.
- **A new purpose-built system:** fully controllable, but every component would be new code with no history, built under a 14-week deadline.

## Consequences

- The template must be simplified by deletion first: one LightGBM model, fewer features, down-sampled data, no DVC remote.
- Hands-on sections of book chapters 04 and 06–09, written against the old `ModelOps/` code, must be rewritten.
- The template ships down-sampled real Rossmann data; the Kaggle rules permit reuse.

Full evaluation: [planning/REFERENCE_SYSTEM_OPTIONS.md](../../planning/REFERENCE_SYSTEM_OPTIONS.md).
