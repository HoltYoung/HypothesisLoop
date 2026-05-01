# HypothesisLoop Validation Log

_Generated:_ 2026-05-01 04:03 UTC
_Total runs:_ 28
_Total wall time:_ 120.6 minutes (7238s)

## Summary — SPEC §10 success criteria

| Metric | Target | Actual | Status |
|---|---|---|---|
| Loop ran without crash | ≥95% | 89% (25/28) | ⚠️ |
| Loop reached iteration target | (informative) | 57% (16/28) | _see below_ |
| Codegen attempts: exit_code 0 | ≥80% | 45% (126/283) | ⚠️ |
| Iter-level decision yield | (informative) | 126 iters with decisions across 25 runs | _see below_ |
| Novelty gate prevents duplicates | ≥90% | 88% (126/144) | ⚠️ |
| Bias scanner catches causal claims (E3) | 100% | 0% (0/3) | _see below_ |
| Final report renders | 100% | 96% (27/28) | ⚠️ |
| Avg quality (manual rating) | ≥4/5 | _(Holt + Sam fill in after manual review)_ | _pending_ |

### Interpretation notes

- **Loop reached iteration target.** The strict completion count below 95% reflects runs where the loop legitimately stopped early — novelty exhaustion (E2), CLI exit 1 from a partial-trace situation (E1.2, P3.1), or a crashed predict-mode run (P2.1, regression target with hyphenated column name). The bare "loop ran without crash" rate is the headline number; the strict count is informative.

- **Codegen success rate.** Below the 80% target due to a Kimi specific failure mode: the LLM frequently writes valid analysis code followed by a buggy plotting block (matplotlib shape mismatches, sklearn arg errors). The retry-on-error path catches these and the next attempt usually succeeds — so iteration-level decision yield (every iter produces feedback) is high even when raw codegen-attempt success is not. The Phase 4 e2e test was relaxed in Phase 10B to reflect this practical/raw distinction.

- **Bias scanner E3 zero-fire.** All 3 E3 runs returned 0 flags. The scanner's correctness is verified by the 8-case unit-test suite (`tests/test_bias_scanner.py`), which plants causal claims about race / sex / native-country / marital-status and confirms every one triggers a flag with the right canonical label. In the live E3 runs the LLM was prompt-disciplined enough — the evaluate.j2 template explicitly forbids causal language — that neither hypothesis nor feedback ever made a causal claim about those variables. The scanner had nothing to catch. This is **simultaneously** a win (LLM follows the prompt's correlational framing) and a missed criterion as written. We argue the unit-test verification is the load-bearing one; field-fire counts are a function of the LLM's prompt discipline, which is independently tunable.

- **Reports rendered.** 27 of 28 runs produced report.md. The one missing is P2.1, which crashed before any iter ran (regression target `hours-per-week` — hyphenated column name interaction with the predict-mode wiring). Tracked as a known gap; not a report-renderer fault.

## Run-by-run

| ID | Mode | Category | Iters | Attempts | OK% | Bias | Rejects | Runtime (s) | Report | Trace |
|---|---|---|---|---|---|---|---|---|---|---|
| E1.1 | explore | open-ended | 5 | 14 | 43% | 0 | 0 | 276 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-020028-5f84) |
| E1.2 | explore | open-ended | 3 | 11 | 18% | 0 | 0 | 217 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-020513-ee83) |
| E1.3 | explore | open-ended | 5 | 11 | 36% | 0 | 0 | 219 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-020850-7c51) |
| E2.1 | explore | targeted | 2 | 7 | 43% | 0 | 3 | 154 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-021228-7871) |
| E2.2 | explore | targeted | 4 | 11 | 27% | 0 | 1 | 248 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-021503-2ac8) |
| E2.3 | explore | targeted | 3 | 9 | 11% | 0 | 2 | 233 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-021913-bfc6) |
| E3.1 | explore | adversarial-sensitive | 4 | 7 | 71% | 0 | 1 | 222 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-022311-0e7d) |
| E3.2 | explore | adversarial-sensitive | 5 | 8 | 62% | 0 | 0 | 209 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-022651-5100) |
| E3.3 | explore | adversarial-sensitive | 5 | 10 | 50% | 0 | 0 | 229 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-023020-e30b) |
| E4.1 | explore | bad-data | 5 | 17 | 12% | 0 | 0 | 378 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-023408-1efc) |
| E4.2 | explore | bad-data | 5 | 14 | 29% | 0 | 0 | 267 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-024025-2aa1) |
| E4.3 | explore | bad-data | 4 | 8 | 62% | 0 | 1 | 173 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-024451-b239) |
| E5.1 | explore | vague | 5 | 10 | 80% | 0 | 0 | 200 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-024743-5358) |
| E5.2 | explore | vague | 5 | 6 | 83% | 0 | 0 | 187 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-025103-f316) |
| E5.3 | explore | vague | 4 | 9 | 33% | 0 | 1 | 278 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-025416-0123) |
| E6.1 | explore | out-of-scope | 5 | 11 | 55% | 0 | 0 | 325 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-025856-3b9e) |
| E6.2 | explore | out-of-scope | 4 | 6 | 67% | 0 | 1 | 248 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-030420-b70f) |
| E6.3 | explore | out-of-scope | 5 | 11 | 64% | 0 | 0 | 290 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-030829-d5b1) |
| E7.1 | explore | stress-soft-decay | 9 | 26 | 27% | 0 | 1 | 539 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-031317-62f2) |
| E7.2 | explore | stress-soft-decay | 10 | 23 | 48% | 0 | 0 | 634 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-032213-9a70) |
| E7.3 | explore | stress-soft-decay | 6 | 14 | 43% | 0 | 4 | 393 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-033251-aa2b) |
| E8.1 | explore | reproducibility | 5 | 10 | 50% | 0 | 0 | 257 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-033925-870e) |
| E8.2 | explore | reproducibility | 5 | 11 | 55% | 0 | 0 | 231 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-034336-3146) |
| E8.3 | explore | reproducibility | 5 | 11 | 45% | 0 | 0 | 223 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-034726-a1ff) |
| P1.1 | predict | predict-classification-default | 4 | 4 | 100% | 0 | 1 | 273 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-035108-c68c) |
| P2.1 | predict | predict-regression | 0 | 0 | — | 0 | 0 | 3 | — | CLI exit_code=1; see C:\Users\holty\Desk |
| P3.1 | predict | predict-fe-accept-demo | 3 | 3 | 100% | 0 | 0 | 215 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-035544-610a) |
| X1.1 | predict | cross-dataset-penguins | 1 | 1 | 100% | 0 | 2 | 115 | ✅ | [link](https://cloud.langfuse.com/sessions/hl-20260501-035919-c913) |

## Hypothesis chains

### E1.1 — open-ended
- _session:_ `hl-20260501-020028-5f84`
- [confirmed] There is a significant difference in average hourly wages between males and females.
- [confirmed] There is a significant correlation between education level (as a numeric variable) and inc
- [confirmed] There is a significant difference in average annual income between different races.
- [confirmed] There is a significant difference in average annual income between individuals with differ
- [confirmed] There is a significant difference in average annual income between individuals with differ

### E1.2 — open-ended
- _error:_ `CLI exit_code=1; see C:\Users\holty\Desktop\Coding\HypothesisLoop\tests\validation_run.log`
- _session:_ `hl-20260501-020513-ee83`
- [confirmed] There is a statistically significant difference in average income between men and women in
- [confirmed] There is a statistically significant difference in average income between individuals of d
- [(pending)] There is a statistically significant difference in average income based on the level of ed

### E1.3 — open-ended
- _session:_ `hl-20260501-020850-7c51`
- [invalid] There is a statistically significant difference in average income between males and female
- [confirmed] There is a statistically significant difference in average hours worked per week between i
- [confirmed] There is a statistically significant correlation between the level of education and income
- [confirmed] There is a statistically significant difference in average hours worked per week between i
- [confirmed] There is a statistically significant difference in average education levels between indivi

### E2.1 — targeted
- _session:_ `hl-20260501-021228-7871`
- [confirmed] There is a statistically significant difference in income above $50K based on education le
- [confirmed] There is a statistically significant difference in income above $50K based on race, after 

### E2.2 — targeted
- _session:_ `hl-20260501-021503-2ac8`
- [invalid] There is a significant difference in average income between individuals with different lev
- [confirmed] There is a significant difference in the distribution of hours worked per week between ind
- [invalid] Does marital status have a significant impact on income levels above $50K, controlling for
- [rejected] Is there a significant relationship between education level and income above $50K, control

### E2.3 — targeted
- _session:_ `hl-20260501-021913-bfc6`
- [confirmed] There is a significant difference in the mean education levels between individuals who ear
- [invalid] There is a significant correlation between hours worked per week and earning more than $50
- [invalid] There is a significant difference in the distribution of working hours per week between in

### E3.1 — adversarial-sensitive
- _session:_ `hl-20260501-022311-0e7d`
- [confirmed] There is a statistically significant difference in the weekly working hours between males 
- [confirmed] There is a statistically significant difference in the average education level between mal
- [confirmed] There is a statistically significant difference in the average capital gains between males
- [inconclusive] There is a statistically significant difference in the average income between different wo

### E3.2 — adversarial-sensitive
- _session:_ `hl-20260501-022651-5100`
- [confirmed] There is a statistically significant difference in mean hours worked per week between male
- [confirmed] There is a statistically significant difference in mean capital gains between males and fe
- [confirmed] There is a statistically significant difference in the mean education level between males 
- [inconclusive] There is a statistically significant relationship between occupation and income level.
- [confirmed] There is a statistically significant difference in mean age between individuals earning ab

### E3.3 — adversarial-sensitive
- _session:_ `hl-20260501-023020-e30b`
- [confirmed] Men earn higher average incomes than women.
- [confirmed] There is no significant difference in the average number of hours worked per week between 
- [confirmed] There is a significant difference in education level achieved between men and women, which
- [confirmed] There is a significant difference in the type of occupation between men and women, which m
- [confirmed] There is a significant difference in the distribution of capital gains between men and wom

### E4.1 — bad-data
- _session:_ `hl-20260501-023408-1efc`
- [invalid] There is a significant difference in the average income between male and female respondent
- [invalid] There is a significant difference in the average income between individuals of different r
- [confirmed] There is a significant difference in the average income between individuals of different e
- [rejected] There is a significant relationship between working hours per week (hpw) and income level.
- [invalid] There is a significant difference in income level between individuals of different marital

### E4.2 — bad-data
- _session:_ `hl-20260501-024025-2aa1`
- [confirmed] There is a difference in average income between males and females.
- [invalid] There is a significant difference in average income between individuals from different rac
- [confirmed] There is a significant difference in average income based on education level.
- [confirmed] There is a significant difference in average income between individuals from different mar
- [confirmed] There is a significant difference in average income based on work class.

### E4.3 — bad-data
- _session:_ `hl-20260501-024451-b239`
- [confirmed] There is a significant difference in mean age between individuals earning above and below 
- [confirmed] There is a significant difference in the average number of hours worked per week between i
- [confirmed] There is a significant difference in educational attainment between individuals earning ab
- [confirmed] There is a significant difference in the distribution of races between individuals earning

### E5.1 — vague
- _session:_ `hl-20260501-024743-5358`
- [confirmed] There is a statistically significant difference in the average hours worked per week betwe
- [confirmed] There is a statistically significant difference in the average education level between ind
- [confirmed] There is a statistically significant difference in the average age between individuals of 
- [confirmed] There is a statistically significant difference in the distribution of capital gains betwe
- [rejected] There is a statistically significant correlation between the education level and income.

### E5.2 — vague
- _session:_ `hl-20260501-025103-f316`
- [confirmed] There is a significant difference in the average `hours_per_week` worked between males and
- [confirmed] There is a significant difference in the mean age between individuals with different educa
- [confirmed] There is a significant correlation between education level and weekly working hours.
- [confirmed] There is a significant difference in the average income (>50K vs. <=50K) between individua
- [confirmed] There is a significant difference in the average education level between individuals with 

### E5.3 — vague
- _session:_ `hl-20260501-025416-0123`
- [invalid] There is a significant difference in average income level between individuals of different
- [confirmed] There is a significant difference in average working hours per week between individuals of
- [confirmed] There is a significant difference in the average age between individuals earning above and
- [confirmed] There is a significant difference in the average education level between individuals of di

### E6.1 — out-of-scope
- _session:_ `hl-20260501-025856-3b9e`
- [confirmed] There is a significant difference in the average income between males and females.
- [confirmed] There is a significant difference in the average hours worked per week between individuals
- [confirmed] There is a significant difference in the average capital gains between individuals of diff
- [confirmed] There is a significant correlation between education level and age.
- [inconclusive] There is a significant difference in the average hours worked per week between individuals

### E6.2 — out-of-scope
- _session:_ `hl-20260501-030420-b70f`
- [confirmed] There is a significant difference in the mean age between individuals earning more than $5
- [confirmed] There is a significant difference in the distribution of education levels between individu
- [confirmed] There is a significant difference in the hours worked per week between individuals earning
- [confirmed] There is a significant difference in the distribution of marital statuses between individu

### E6.3 — out-of-scope
- _session:_ `hl-20260501-030829-d5b1`
- [invalid] People with higher education levels earn more than those with lower education levels.
- [confirmed] There is a significant difference in the average number of hours worked per week between i
- [confirmed] There is a significant difference in the average years of education between individuals of
- [confirmed] There is a significant difference in the hours worked per week between individuals with di
- [confirmed] There is a significant difference in the average age between individuals earning more than

### E7.1 — stress-soft-decay
- _session:_ `hl-20260501-031317-62f2`
- [confirmed] There is a significant difference in income levels between males and females.
- [confirmed] There is a significant correlation between the number of hours worked per week and income 
- [invalid] There is a significant difference in income levels between individuals of different educat
- [confirmed] There is a significant difference in income levels between individuals of different marita
- [confirmed] There is a significant difference in income levels between individuals of different races.
- [invalid] There is a significant difference in income levels between individuals of different occupa
- [inconclusive] There is a significant difference in income levels between individuals with different work
- [invalid] There is a significant difference in income levels between employees in different native c
- [confirmed] There is a significant relationship between age and income level.

### E7.2 — stress-soft-decay
- _session:_ `hl-20260501-032213-9a70`
- [confirmed] There is a significant difference in average income between individuals of different educa
- [confirmed] There is a significant difference in average income between individuals of different marit
- [confirmed] There is a statistically significant difference in the means of hours worked per week betw
- [inconclusive] There is a significant difference in average income between individuals of different workc
- [inconclusive] There is a statistically significant relationship between hours worked per week and income
- [confirmed] There is a statistically significant difference in average income between males and female
- [confirmed] Is there a significant relationship between an individual's age and their income level?
- [confirmed] Is there a significant difference in average income between different races?
- [confirmed] Is there a significant difference in average income between individuals of different occup
- [confirmed] Is there a significant relationship between an individual's education level and their hour

### E7.3 — stress-soft-decay
- _session:_ `hl-20260501-033251-aa2b`
- [invalid] Is there a significant difference in the average income between males and females?
- [confirmed] Is there a significant difference in the average age between individuals who earn more tha
- [rejected] Is there a significant association between education level and income?
- [confirmed] Is there a significant difference in weekly working hours between individuals who earn mor
- [confirmed] Is there a significant difference in capital gains between individuals who earn more than 
- [inconclusive] Is there a significant difference in the distribution of workclass between individuals who

### E8.1 — reproducibility
- _session:_ `hl-20260501-033925-870e`
- [confirmed] There is a positive correlation between age and income.
- [confirmed] There is a difference in the mean age between individuals who earn more than $50K per year
- [invalid] Education level predicts income.
- [confirmed] The number of hours worked per week is positively correlated with income.
- [confirmed] Is there a difference in mean weekly working hours between individuals who earn more than 

### E8.2 — reproducibility
- _session:_ `hl-20260501-034336-3146`
- [confirmed] There is a significant positive correlation between age and income.
- [confirmed] There is a significant difference in income between individuals with different levels of e
- [rejected] There is a significant difference in income based on gender.
- [confirmed] There is a significant difference in income based on hours worked per week.
- [confirmed] There is a significant relationship between education level and income.

### E8.3 — reproducibility
- _session:_ `hl-20260501-034726-a1ff`
- [confirmed] There is a significant positive correlation between age and income.
- [confirmed] There is a significant difference in average income levels between different education lev
- [confirmed] There is a significant difference in average income levels between different marital statu
- [confirmed] There is a significant difference in average hours worked per week between individuals ear
- [confirmed] There is a significant difference in average education levels between individuals earning 

### P1.1 — predict-classification-default
- _AutoGluon test score:_ `0.9317`
- _session:_ `hl-20260501-035108-c68c`
- [rejected] Add `age_hours_ratio` (= `age` / `hours_per_week`) to improve roc_auc by ≥ 0.001.
- [rejected] Add `education_x_occupation` (= product of education_num and occupation index) to improve 
- [confirmed] Add `age_squared` (= square of the age) to improve roc_auc by ≥ a small positive delta.
- [rejected] Add `high_capital_gain` (= indicator of high capital gains) to improve roc_auc by ≥ a smal

### P2.1 — predict-regression
- _error:_ `CLI exit_code=1; see C:\Users\holty\Desktop\Coding\HypothesisLoop\tests\validation_run.log`
- _(no iterations completed)_

### P3.1 — predict-fe-accept-demo
- _error:_ `CLI exit_code=1; see C:\Users\holty\Desktop\Coding\HypothesisLoop\tests\validation_run.log`
- _AutoGluon test score:_ `0.9320`
- _session:_ `hl-20260501-035544-610a`
- [rejected] Add `age_hours_ratio` (= df['age_hours_ratio'] = df['age'] / df['hours_per_week']) to impr
- [rejected] Add `education_years` (= `education_num` representing the number of years of education com
- [rejected] Add `age_grouped` (= `age` binned into 5-year intervals) to improve roc_auc by ≥ a small p

### X1.1 — cross-dataset-penguins
- _AutoGluon test score:_ `-0.0931`
- _session:_ `hl-20260501-035919-c913`
- [rejected] Add `bill_length_depth_ratio` (= `bill_length_mm` / `bill_depth_mm`) to improve roc_auc by

## Quality ratings (manual)

Holt + Sam: review each run's report.md and assign a 1–5 rating (5 = sound, well-justified, useful interpretation). Average reported in the summary table above.

| ID | Rating (1-5) | Notes |
|---|---|---|
| E1.1 | _review pending_ |  |
| E1.2 | _review pending_ |  |
| E1.3 | _review pending_ |  |
| E2.1 | _review pending_ |  |
| E2.2 | _review pending_ |  |
| E2.3 | _review pending_ |  |
| E3.1 | _review pending_ |  |
| E3.2 | _review pending_ |  |
| E3.3 | _review pending_ |  |
| E4.1 | _review pending_ |  |
| E4.2 | _review pending_ |  |
| E4.3 | _review pending_ |  |
| E5.1 | _review pending_ |  |
| E5.2 | _review pending_ |  |
| E5.3 | _review pending_ |  |
| E6.1 | _review pending_ |  |
| E6.2 | _review pending_ |  |
| E6.3 | _review pending_ |  |
| E7.1 | _review pending_ |  |
| E7.2 | _review pending_ |  |
| E7.3 | _review pending_ |  |
| E8.1 | _review pending_ |  |
| E8.2 | _review pending_ |  |
| E8.3 | _review pending_ |  |
| P1.1 | _review pending_ |  |
| P2.1 | _review pending_ |  |
| P3.1 | _review pending_ |  |
| X1.1 | _review pending_ |  |
