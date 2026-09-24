# Submission README (Preprint 2: dl_forward)

This folder contains everything needed to post the second preprint. The
applicant performs the actual upload; nothing here submits on your behalf.

## Artifacts

- `MANUSCRIPT_learned_representations_admet.md` — canonical manuscript
  (single source of truth for prose and numbers).
- `PREPRINT_SUBMISSION.md` — generated, self-contained copy with the figures
  embedded inline. This is the upload-ready reading copy. Do not edit it by
  hand; regenerate it (see below) after any manuscript change.
- `figures/` — fig0 and fig_transfer_curves through fig_campaign_efficiency
  PNGs, referenced by both files.
- `SUBMISSION_CHECKLIST.md` — venue, category, title, abstract, author slot,
  citable repo commit, figure list, and the not-a-clinical-claim statement.

## Regenerate the submission copy

```bash
cd /home/snptx/snptx-core && source .venv/bin/activate
python pilot_phd/preprint/dl_forward/make_submission.py
```

## Make a PDF (optional)

`pandoc` and LaTeX are not installed in this environment, so no PDF is built
here. Do not install them without confirmation. Once a machine with pandoc
and a LaTeX engine is available:

```bash
cd pilot_phd/preprint/dl_forward
pandoc PREPRINT_SUBMISSION.md -o preprint.pdf \
  --resource-path=.:figures --pdf-engine=xelatex
```

bioRxiv also accepts a Markdown/PDF with embedded figures, or a LaTeX
source; `PREPRINT_SUBMISSION.md` is structured so either path works.

## Upload steps (performed by the applicant)

1. Confirm the citable repo commit is tagged and pushed
   (see `SUBMISSION_CHECKLIST.md`, tag `dl-forward-preprint-v1`).
2. Build the PDF from `PREPRINT_SUBMISSION.md` (step above) or upload the
   Markdown plus the `figures/` folder if the venue accepts Markdown.
3. Go to bioRxiv (q-bio, no-endorsement path) and follow its submission
   wizard.
4. Paste the title and abstract from `SUBMISSION_CHECKLIST.md`, add authors
   and affiliations, and attach the figures in order (fig0 first, as the
   system schematic).
5. Paste the Generative AI Disclosure into the venue's AI-use field if it
   asks for one separately; it is also included as a section in the
   manuscript.
6. Submit. **This final upload is the applicant's step, not the agent's.**

## Relationship to Preprint 1

This is Preprint 2, a DL-forward companion to
`../MANUSCRIPT_calibrated_sequential_discovery.md` (Preprint 1). It shares
the same decision-engine machinery (SPRT, split-conformal, selective
prediction) but swaps the oracle for a learned multi-task graph ensemble.
Post independently; each cites its own tag on the shared
`snptx-repro-discovery` repository.
