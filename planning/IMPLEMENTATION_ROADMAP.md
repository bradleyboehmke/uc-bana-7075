# Implementation Roadmap

**Plan:** six phases from 2026-11-19 to the first day of class on 2027-03-01 (D-38), about 14 weeks. The order puts the reference system and one fully built pilot module (Week 2 DataOps) ahead of everything else, so the module template is proven before it is copied six more times.

**Assumptions:**
- First day of class 2027-03-01 (confirmed, D-38).
- No development work over 2026-12-21 to 2027-01-01.
- The instructor leads development with Claude Code sessions; helpers are available for testing (D-40), with names and tasks still to assign (Q-07 in [DECISION_LOG.md](DECISION_LOG.md)).

Task IDs refer to [DEVELOPMENT_BACKLOG.md](DEVELOPMENT_BACKLOG.md).

## Phases

| Phase | Dates | Goal | Tasks | Exit gate |
|---|---|---|---|---|
| 0. Approve | Nov 19 (done) | Plan approved; data question answered | CID-01 (done), REF-01 (done) | G1 and G2 passed |
| 1. Foundations | Nov 30 – Dec 18 | Working, simplified reference system and environments; book restructured | CID-02, REPO-01, REPO-03, REF-02 – REF-06, REF-08, ENV-01, QA-01 (start) | Template runs end to end in Colab and Codespaces |
| 2. Pilot | Jan 4 – Jan 15 | Week 2 complete in every component and tested with a non-coder | REPO-02, REF-07, REF-09, ENV-02 (Week 2), CONT-02, TRY-02, CASE-02, CASE-03, AI-01, PROJ-01, CID-04, VID-01, VID-02 (Weeks 0, 2) | **G3** |
| 3. Early weeks and project | Jan 18 – Feb 5 | Weeks 0, 1, 3 complete; project materials ready for Week 0 | CONT-01, CONT-03, CONT-04, CONT-09, TRY-01, TRY-03, TRY-04, ENV-02 – ENV-04, PROJ-02 – PROJ-05, PROJ-07, CASE-01, BUILD-01 (start), VID-02 (Weeks 1, 3) | Weeks 0–3 publishable |
| 4. Late weeks and assessment | Feb 8 – Feb 19 | Weeks 4–7 required components; grading set up | CONT-05 – CONT-08, TRY-05 – TRY-08, BUILD-01, BUILD-02, AI-02, AI-03, PROJ-06, ASSESS-01 – ASSESS-04, CID-03, REPO-04, REPO-05, VID-02 (Weeks 4–7), VID-03 | **G4** |
| 5. Test and launch | Feb 22 – Mar 1 | Course runs on fresh accounts | QA-02 follow-ups, QA-03, QA-04 | **G5** |

Videos run across Phases 2–4 at about one every 1.5 weeks, each recorded once its week's Try It and reference system piece is stable.

## Why this order

1. **Reference system before content.** Every See It, Try It, and the hands-on rewrites of chapters 04 and 06–09 point at reference system files. Writing content first would mean rewriting it once the system settles.
2. **Simplify by deletion first.** Removing the ensemble, Optuna, extra features, and the DVC remote (REF-04, REF-06) gives a smaller system to modernize in REF-05 and REF-08.
3. **One pilot module before the rest.** Week 2 uses every module component, a Colab Try It, a failure scenario, Great Expectations Data Docs, a supplementary example, and a Build It. Problems with the template, notebooks, or lab check format show up once instead of seven times.
4. **Week 0 project materials by early February.** The team survey (PROJ-01) is due in Phase 2 because team formation happens before Week 1.
5. **Late weeks last.** Weeks 4–7 can be finished last because the in-term schedule gives slack: if needed, Week 6 and 7 See It videos can be finished during the term (D-31 sets the minimum).

## Decision gates

| Gate | When | Decision needed | Blocks |
|---|---|---|---|
| **G1** Plan approval | **Passed 2026-11-19** | Approve planning documents; accept or change P-01 – P-12 | Everything |
| **G2** Data source | **Passed 2026-11-19** | Real down-sampled Rossmann data; Kaggle reuse permitted (D-37) | — |
| **G3** Pilot validated | End of Phase 2 | Module template, Try It format, lab check format, and video style approved, or revised and re-tested | All other CONT, TRY, and VID tasks |
| **G4** Assessment approved | End of Phase 4 | Rubrics, weights in Canvas, peer evaluation procedure | Syllabus, Canvas publication |
| **G5** Go / no-go | 2027-02-22 | Fresh-account dry run passed; fallback plan for anything failing | Launch |

The template's location is settled: the instructor's personal GitHub account (D-39).

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Great Expectations 1.x port slower than estimated | Medium | Delays the pilot | Reuse the working 1.x pattern in [DataOps/dataops_utils.py](../DataOps/dataops_utils.py); limit suites to raw and features at launch |
| Colab or Codespaces behavior changes during term | Low | Try Its break mid-term | Scheduled fresh-install CI job (QA-01); fallback artifacts (ENV-04) |
| Chapter rewrites take longer than planned | Medium | Phase 4 slips | Concept sections are kept; only hands-on sections are rewritten. Weeks 6–7 rewrites can finish in term |
| Single lead developer | Medium | Any absence slips the plan | Helpers take testing (D-40); Phase 3 and 4 tasks are independent per week and can be reordered; P2 items absorb cuts |
| Teams of uneven size or no builder | Medium | Evidence minimums unmet | Assignment rules (PROJ-02); baseline-model fallback for the platform lead (D-26) |

## After launch

- **During the term:** record class sessions; log Try It problems as issues in the reference system repository; collect lab check statistics.
- **Summer 2027:** P2 tasks (REF-10, BUILD-03, BUILD-04, AI-04, VID-04, QA-05), plus revisions from student feedback, before the Fall 2027 offering.
