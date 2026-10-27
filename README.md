# CPO Transmission Co-Scientist

AI-assisted analysis of potential transmission sources for hospital-acquired carbapenemase-producing organisms (CPOs), integrating:

- patient and environmental surveillance data,
- temporal and microbiological compatibility,
- whole-genome sequencing (WGS),
- XGBoost candidate-source prioritisation,
- patient–environment transmission-network analysis, and
- an optional heterogeneous LLM council for evidence synthesis.

---

# 1. Project objective

The clinical question is:

> For patients who were initially CPO-negative but subsequently acquired a carbapenemase gene during hospitalisation, which earlier known CPO-positive patients or positive environmental observations are plausible acquisition sources?

The analysis distinguishes between:

- patient-associated clonal circulation,
- patient-associated plasmid circulation,
- environment-associated clonal circulation,
- environment-associated plasmid circulation,
- plausible but genomically unsupported source hypotheses, and
- unresolved cases.

The main clinical pipeline is:

```text
Raw surveillance data
        ↓
Acquisition-case definition
        ↓
Candidate source generation
        ↓
Temporal + microbiological feature engineering
        ↓
WGS clonal / plasmid linkage
        ↓
XGBoost candidate prioritisation
        ↓
SHAP interpretation
        ↓
Patient–environment network analysis
        ↓
Clinical source interpretation
```

The **LLM council is an optional extension** and is not required to reproduce the main clinical results.

---

# 2. Important terminology

## Candidate source

A previous:

1. `Known_CPO_pos` patient observation, or
2. positive environmental observation

that is temporally eligible to be considered as a possible source for one of the acquisition cases.

## Target

One of the initially CPO-negative patients who subsequently acquired a carbapenemase gene during hospitalisation.

## WGS clonal link

The source and target isolates satisfy the supplied genomic clonal-linkage criteria.

This supports circulation of a related bacterial lineage.

It does **not** prove direct source → recipient transmission.

## WGS plasmid link

The source and target isolates satisfy the supplied plasmid-linkage criteria.

This supports circulation of related resistance-bearing plasmid material and does not require the source and target to have the same bacterial clone.

## XGBoost score

A model-derived **candidate-prioritisation score**.

Although it is generated using `predict_proba`, it is **not interpreted as a calibrated probability that transmission occurred**.

---

# 3. Repository structure

The relevant project structure is approximately:

```text
cpo-transmission-coscientist/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   ├── network/
│   ├── paper/
│   └── reports/
│       ├── agents/
│       ├── council_v1_pilot/
│       ├── council_v2/
│       ├── council_v2_repeats/
│       ├── council_final/
│       └── role_ablation/
│
├── src/
│   └── cpo/
│       ├── config.py
│       ├── data_loader.py
│       ├── acquisition_cases.py
│       ├── candidate_sources.py
│       ├── features.py
│       ├── wgs_linkage.py
│       ├── ml_dataset.py
│       ├── xgboost_model.py
│       ├── explainability.py
│       │
│       ├── tools/
│       │   └── transmission_tools.py
│       │
│       ├── agents/
│       ├── council/
│       ├── council_v2/
│       └── council_final/
│
└── scripts/
    ├── 01_audit_data.py
    ├── 02_build_acquisition_cases.py
    ├── 03_build_candidate_pairs.py
    ├── 04_build_pair_features.py
    ├── 05_audit_wgs_links.py
    ├── 06_attach_wgs_links.py
    ├── 07_build_ml_dataset.py
    ├── 08_train_xgboost.py
    ├── 09_explain_xgboost.py
    ├── 10_run_agent.py
    ├── 11_run_llm_council.py
    ├── 12_run_llm_council_v2.py
    ├── 13_run_llm_council_final.py
    ├── 14_repeat_council_v2.py
    ├── 15_summarize_council_v2_repeats.py
    ├── 16_run_role_ablation.py
    ├── 17_summarize_role_ablation.py
    ├── 18_generate_paper_tables.py
    ├── 19_generate_paper_figures.py
    ├── 20_generate_pipeline_diagram.py
    ├── 21_generate_results_text.py
    ├── 22_build_transmission_network.py
    ├── 23_analyze_transmission_network.py
    └── 24_plot_transmission_network.py
```

---

# 4. Required source data

Place the clinician-provided Excel workbooks under:

```text
data/raw/
```

The original analysis used:

```text
17082026_HTS_NN_Analysis_ENV_cleaned_NTUteam.xlsx
17082026_HTS_NN_Analysis_WGS_cleaned_NTUteam.xlsx
```

The environmental/patient workbook contains sheets such as:

```text
Legend
ENV_raw
ENV_pos
Known_CPO_pos
```

The WGS workbook contains:

```text
Legend
WGS_isolate_list
Clonal_linked_pairs
Plasmid_linked_pairs
```

## Data-security note

These are clinical research data.

**Do not commit raw NCID data to a public GitHub repository.**

The recommended `.gitignore` should include:

```text
data/raw/
```

and any other directory containing sensitive patient-level data.

---

# 5. Environment setup

The project was developed using Python 3.11.

A suitable environment can be created with:

```bash
conda create -n cpo-coscientist python=3.11 -y
conda activate cpo-coscientist
```

Install the core dependencies required by the main pipeline:

```bash
pip install pandas numpy openpyxl scikit-learn xgboost shap matplotlib networkx jinja2
```

If a repository-specific `requirements.txt` or `environment.yml` is available, use that instead.

---

# 6. Running scripts

The repository uses the `src/` layout.

On macOS/Linux:

```bash
PYTHONPATH=src python scripts/<script_name>.py
```

For example:

```bash
PYTHONPATH=src python scripts/02_build_acquisition_cases.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/02_build_acquisition_cases.py
```

---

# 7. Main reproducibility pipeline

Run the main scripts in numerical order.

---

## Stage 1 — Data audit

```bash
PYTHONPATH=src python scripts/01_audit_data.py
```

Purpose:

- inspect workbook structure,
- verify columns,
- inspect identifiers and dates,
- inspect GeneXpert/culture/WGS fields,
- identify missingness and formatting issues.

No modelling occurs at this stage.

---

## Stage 2 — Build acquisition cases

```bash
PYTHONPATH=src python scripts/02_build_acquisition_cases.py
```

Purpose:

Identify the eight patients who acquired a carbapenemase gene during hospitalisation.

Expected targets:

```text
T0016
T0036
T0052
T0058
T0080
T0081
T0124
T0125
```

Expected biological summary:

```text
T0016  NDM      no target culture/WGS
T0036  NDM      P. aeruginosa ST308, blaNDM-1
T0052  IMP      P. aeruginosa ST235, blaIMP-1
T0058  OXA-48   K. pneumoniae ST39, blaOXA-181
T0080  OXA-48   K. pneumoniae ST15, blaOXA-181
T0081  OXA-48   K. pneumoniae ST76, blaOXA-181
T0124  VIM      P. aeruginosa ST823, blaVIM-2
T0125  NDM      K. pneumoniae ST147, blaNDM-1
```

The acquisition date produced at this stage is subsequently used to determine candidate-source eligibility.

---

## Stage 3 — Generate candidate source–target pairs

```bash
PYTHONPATH=src python scripts/03_build_candidate_pairs.py
```

Core logic is implemented in:

```text
src/cpo/candidate_sources.py
```

For every acquisition target, all eligible previous:

```text
Known_CPO_pos patient observations
+
positive environmental observations
```

are enumerated.

### Eligibility rules

The source must:

- be an allowed source type;
- occur before or on the target acquisition date;
- satisfy the available hospital/site constraint.

Same-day observations are retained but labelled as directionally ambiguous.

Importantly, candidate generation does **not** require:

```text
same gene
same species
same ST
```

These are downstream features rather than eligibility rules.

### Expected checkpoint

Approximately:

```text
1,274 candidate source–target pairs

914 patient-source pairs
360 environmental-source pairs

1,253 strictly prior
21 same-day ambiguous
```

---

# 8. Stage 4 — Feature engineering

Run:

```bash
PYTHONPATH=src python scripts/04_build_pair_features.py
```

Main implementation:

```text
src/cpo/features.py
```

Features characterise temporal and microbiological compatibility.

Examples include:

```text
days_before_acquisition
source_is_environment
same_cp_family_gx
same_cp_family_isolate
same_exact_cp_gene_isolate
same_species_any
same_st_any
```

At this stage these are descriptive features.

They do **not** determine whether transmission occurred.

---

# 9. Stages 5–6 — WGS linkage

Audit WGS:

```bash
PYTHONPATH=src python scripts/05_audit_wgs_links.py
```

Attach WGS relationships:

```bash
PYTHONPATH=src python scripts/06_attach_wgs_links.py
```

Main linkage code:

```text
src/cpo/wgs_linkage.py
```

Candidate source/target isolates are resolved using information including:

```text
sample ID
species
ST
CP gene
WGS isolate ID
```

Pairs are checked in both orientations against:

```text
Clonal_linked_pairs
Plasmid_linked_pairs
```

### Expected checkpoint

Approximately:

```text
1,021 WGS-evaluable candidate pairs

14 direct prior WGS links

3 clonal
11 plasmid
```

---

# 10. WGS reference interpretation

The primary case-level WGS interpretation is:

```text
T0016 → UNRESOLVED_NO_TARGET_WGS

T0036 → NO_DIRECT_WGS_LINK

T0052 → NO_DIRECT_WGS_LINK

T0058 → PATIENT_ASSOCIATED_PLASMID

T0080 → PATIENT_ASSOCIATED_PLASMID

T0081 → PATIENT_ASSOCIATED_CLONAL

T0124 → ENVIRONMENT_ASSOCIATED_CLONAL

T0125 → NO_DIRECT_WGS_LINK
```

These are genomic association/circulation categories.

They should **not** be described as proven direct donor-to-recipient transmission.

---

# 11. Stage 7 — Build the ML dataset

Run:

```bash
PYTHONPATH=src python scripts/07_build_ml_dataset.py
```

Main implementation:

```text
src/cpo/ml_dataset.py
```

The ML dataset includes only:

```text
WGS-evaluable
+
strictly prior
+
deduplicated
```

candidate pairs.

Same-day ambiguous observations are excluded from ML training.

T0016 is excluded because no target WGS is available to create a genomic reference label.

### Label

The binary reference label is:

```text
label_any_wgs_link
```

where:

```text
1 = clonal WGS link OR plasmid WGS link
0 = WGS-evaluable but not directly linked
```

### Expected checkpoint

```text
999 candidate pairs

14 positive
985 negative

7 WGS-evaluable acquisition targets
```

Positive prevalence is approximately:

```text
1.4%
```

This is therefore a highly imbalanced ranking problem.

---

# 12. XGBoost model

Run:

```bash
PYTHONPATH=src python scripts/08_train_xgboost.py
```

Main implementation:

```text
src/cpo/xgboost_model.py
```

Two feature sets are evaluated.

---

## Core model — 5 inputs

```text
days_before_acquisition
source_is_environment
same_cp_family_gx
source_gx_positive
source_gx_family_count
```

---

## Augmented model — FINAL XGBoost feature set

The Augmented model uses all Core features plus:

```text
same_cp_family_isolate
same_exact_cp_gene_isolate
same_species_any
same_st_any
```

Therefore the final Augmented model has **9 input features**.

### Exact interface

| Field | Type | Encoding |
|---|---|---|
| `days_before_acquisition` | numeric | non-negative days |
| `source_is_environment` | binary | 1=environment, 0=patient |
| `same_cp_family_gx` | binary | 1=match, 0=otherwise |
| `source_gx_positive` | binary | 1=positive, 0=otherwise |
| `source_gx_family_count` | integer | 0,1,2,... |
| `same_cp_family_isolate` | binary | 1=match, 0=otherwise |
| `same_exact_cp_gene_isolate` | binary | 1=exact match, 0=otherwise |
| `same_species_any` | binary | 1=same species, 0=otherwise |
| `same_st_any` | binary | 1=same known ST, 0=otherwise |

---

# 13. Missing / unknown feature handling

The current preprocessing does **not** create a separate third `"unknown"` category for compatibility indicators.

For exact compatibility with the current implementation:

```text
cannot establish a match
→ 0 / False
```

Examples:

```text
unknown ST
→ same_st_any = 0

missing isolate gene
→ same_exact_cp_gene_isolate = 0

missing isolate CP-family information
→ same_cp_family_isolate = 0
```

If no source GeneXpert CP family is available/detected:

```text
source_gx_positive = 0
source_gx_family_count = 0
same_cp_family_gx = 0
```

Valid model rows are expected to contain:

```text
days_before_acquisition
source type
```

because these fields are required during candidate generation.

### Important limitation

Under this representation:

```text
unknown/unavailable
```

and:

```text
known mismatch
```

can both map to `0` for concordance features.

FYP integrations that must reproduce the existing model exactly should preserve this behaviour.

---

# 14. XGBoost output

The model is a binary `XGBClassifier`.

For each candidate pair:

```python
scores = model.predict_proba(X)[:, 1]
```

is used.

The positive class means:

```text
direct WGS link
=
clonal OR plasmid
```

The value is stored in current outputs as:

```text
predicted_probability
```

Despite this name, it is interpreted as:

> **candidate-prioritisation score**

and **not** as:

> probability that transmission occurred.

Candidates are sorted in descending score order separately for each acquisition target:

```text
highest score → rank 1
next highest → rank 2
...
```

This within-target ranking is the main downstream output.

---

# 15. XGBoost validation

Validation uses:

> **leave-one-target-out validation**

For example:

```text
train:
T0036, T0052, T0058, T0080, T0081, T0125

test:
T0124
```

and this is repeated for each target.

This is preferable to a random candidate-row split because multiple candidate rows originate from the same clinical target.

### Important leakage controls

Do **not**:

- include clonal/plasmid WGS fields as XGBoost features;
- randomly mix candidate rows from the same target between train and test;
- include same-day ambiguous source pairs in causal ML evaluation;
- interpret the XGBoost score as a calibrated transmission probability.

---

# 16. Expected Augmented-model results

The Augmented model should produce approximately:

```text
Pooled AUROC      0.963
Pooled AUPRC      0.311

MRR               0.688

Hit@1             0.50
Hit@3             0.75
Hit@5             1.00

Median best-positive rank
                   1.5
```

These values provide useful reproducibility checkpoints.

The evaluation sample is small:

```text
4 targets contain positive WGS-linked candidates
```

so the metrics should not be interpreted as large-cohort performance estimates.

---

# 17. Stage 9 — SHAP interpretation

Run:

```bash
PYTHONPATH=src python scripts/09_explain_xgboost.py
```

Implementation:

```text
src/cpo/explainability.py
```

Out-of-fold SHAP values are used.

This means the explanation for a target comes from the model that did not train on that target.

Approximate global importance:

```text
same_exact_cp_gene_isolate   2.55
days_before_acquisition      0.51
same_cp_family_isolate       0.45
same_species_any             0.44
...
```

SHAP explains why XGBoost prioritises a candidate.

It does **not** establish a transmission mechanism.

---

# 18. Example: T0036

T0036 illustrates why ML ranking and WGS must be kept separate.

The top candidates were:

```text
HTS-E120
HTS-E121
```

Both were approximately:

```text
9 days prior
same P. aeruginosa
same ST308
same blaNDM-1
XGBoost rank #1 / #2
```

However:

```text
no direct WGS clonal link
no direct WGS plasmid link
```

Therefore:

```text
environment-favoured hypothesis
```

is reasonable, but:

```text
environmental transmission confirmed
```

is not.

---

# 19. Analytic transmission network

The network analysis extends pairwise source ranking into hospital-level dissemination structure.

Run:

```bash
PYTHONPATH=src python scripts/22_build_transmission_network.py
PYTHONPATH=src python scripts/23_analyze_transmission_network.py
PYTHONPATH=src python scripts/24_plot_transmission_network.py
```

Outputs are written under:

```text
outputs/network/
```

including:

```text
network_candidate_rows.csv
transmission_network_edges.csv
transmission_network_nodes.csv
transmission_network.graphml
network_build_summary.json
network_analysis_summary.json
network_source_hubs.csv
network_target_source_profiles.csv
network_target_projection.csv
network_connected_components.csv
```

---

# 20. Network interpretation

Network nodes include:

```text
P:<patient ID>
    patient-source node

E:<environment sample ID>
    environmental-source node

T:<target ID>
    acquisition target
```

Edges represent:

```text
source → acquisition target
```

and carry information such as:

```text
days before acquisition
XGBoost rank
XGBoost score
exact CP-gene match
species match
ST match
clonal WGS link
plasmid WGS link
```

The analytic network is used to identify:

- recurrent source nodes,
- genomic hubs,
- shared acquisition clusters,
- patient-vs-environment source support,
- acquisition cases that may belong to a common dissemination structure.

---

# 21. Expected network checkpoints

Current analysis produced approximately:

```text
336 aggregated source→target edges
82 unique source nodes

30 high-evidence edges
(top-5 XGBoost and/or WGS-linked)

14 direct WGS-supported edges

3 clonal
11 plasmid

4 recurrent top-5 sources
4 genomic hub sources
```

The major shared patient-associated cluster involves:

```text
T0058
T0080
T0081
```

with recurrent source patients including:

```text
T0054
T0057
T0068
T0034
```

T0124 shows a distinct:

```text
environment-associated clonal
```

pattern involving environmental sample:

```text
HTS-E186
```

---

# 22. Important network timing caveat

Patient-level network nodes can aggregate multiple samples belonging to the same patient.

For example:

```text
P:T0054
```

may contain several different sample IDs.

Therefore:

> **Do not use the aggregated network `days_before_acquisition` value as the exact timing of a specific WGS-linked isolate.**

For exact isolate timing, always return to the sample-level:

```text
wgs_labeled_pairs.csv
```

or the underlying WGS/candidate-pair dataset.

Use the network for:

```text
hub structure
shared-source relationships
cluster structure
```

rather than replacing sample-level genomic evidence.

---

# 23. Final clinical interpretation

Current case-level interpretation:

| Target | Interpretation |
|---|---|
| T0016 | Unresolved — no target WGS |
| T0036 | Environment-favoured; no direct WGS confirmation |
| T0052 | Patient-favoured / mixed; no direct WGS confirmation |
| T0058 | Patient-associated plasmid circulation |
| T0080 | Patient-associated plasmid circulation |
| T0081 | Patient-associated clonal circulation |
| T0124 | Environment-associated clonal circulation |
| T0125 | Patient-favoured; no direct WGS confirmation |

Overall:

```text
4/8
direct WGS-supported source-category evidence

3 patient-associated
1 environment-associated

3 additional cases
plausible but genomically unconfirmed

1 case
unresolved due to missing target WGS
```

---

# 24. LLM Council — OPTIONAL EXTENSION

The LLM council is **not required to reproduce the main clinical analysis**.

The principal clinical evidence remains:

```text
candidate generation
→ XGBoost prioritisation
→ WGS
→ network analysis
```

The council was explored as an evidence-synthesis layer.

---

# 25. Which LLM council is the final version?

## IMPORTANT

The finalized / primary LLM council is:

```text
src/cpo/council_v2/
```

and the primary runner is:

```text
scripts/12_run_llm_council_v2.py
```

### Do not confuse this with `council_final`

Despite its folder name:

```text
src/cpo/council_final/
```

is **not** the primary/final scientific configuration.

It is a later **v2.1 sensitivity experiment**.

---

# 26. Council version history

## Original single-model agent

```text
src/cpo/agents/
scripts/10_run_agent.py
```

Qwen3-8B performed:

```text
Investigator
→ Critic
→ Revision
```

This was an early baseline and showed problems including:

- identifier confusion,
- causal overstatement,
- inconsistent clonal/plasmid interpretation.

This is not the primary council.

---

## Council v1 — pilot only

```text
src/cpo/council/
scripts/11_run_llm_council.py
```

Problems included:

- overly large/noisy evidence packet,
- clonal/plasmid confusion,
- structured-output failures.

This version should be treated as:

```text
PILOT / DEVELOPMENT ONLY
```

---

## Council v2 — FINAL / PRIMARY

```text
src/cpo/council_v2/
scripts/12_run_llm_council_v2.py
```

This is the frozen primary council.

Roles:

```text
Qwen3-8B
→ Investigator

MedGemma 1.5-4B
→ Biomedical Expert

Llama 3.1-8B
→ Causal Critic
```

All three receive the same compact deterministic evidence packet.

The packet prioritises:

```text
direct WGS evidence
top XGBoost candidates
compact SHAP information
chronology
microbiology
```

Each model votes independently.

Python then performs:

```text
unanimous / majority / no-consensus aggregation
```

followed by deterministic validation.

### Frozen decoding

The main configuration used:

```text
temperature = 0
seed = 42
```

Do not change the prompts, schema, evidence format or decoding parameters if attempting to reproduce the published/frozen Council-v2 result.

---

# 27. Council-v2 reproducibility experiments

Primary run:

```bash
PYTHONPATH=src python scripts/12_run_llm_council_v2.py
```

Repeatability:

```bash
PYTHONPATH=src python scripts/14_repeat_council_v2.py
PYTHONPATH=src python scripts/15_summarize_council_v2_repeats.py
```

Role assignment ablation:

```bash
PYTHONPATH=src python scripts/16_run_role_ablation.py
PYTHONPATH=src python scripts/17_summarize_role_ablation.py
```

Council-v2 achieved approximately:

```text
87.5% end-to-end agreement

100% agreement among completed cases
```

across the frozen repeatability experiment.

The recurring issue was a structured-output failure for T0080 rather than an incorrect completed interpretation.

---

# 28. Council v2.1 / `council_final` — sensitivity only

Files:

```text
src/cpo/council_final/
scripts/13_run_llm_council_final.py
```

This experiment removed model-generated `target_id` from the response interface.

Although intended to fix a technical identifier issue, model behaviour changed and overall stability worsened.

Therefore:

```text
council_v2
= PRIMARY / FINAL

council_final
= v2.1 SENSITIVITY EXPERIMENT
```

If the repository is being cleaned, it is recommended to **archive or rename** `council_final` rather than treating it as the final model.

For example:

```text
archive/
└── council_v2_1_sensitivity/
```

---

# 29. Local LLM requirements

The council experiments were run locally through Ollama.

Required model tags:

```text
qwen3:8b
medgemma1.5:4b
llama3.1:8b
```

Example:

```bash
ollama pull qwen3:8b
ollama pull medgemma1.5:4b
ollama pull llama3.1:8b
```

Make sure the Ollama server is running before executing council scripts.

Again, **Ollama is not required for the main XGBoost/network pipeline.**

---

# 30. Recommended integration paths for FYP students

Different projects do not need to reproduce the entire repository.

## Option A — Integrate only XGBoost

Use:

```text
feature engineering
→ 9-feature vector
→ XGBClassifier
→ predict_proba(X)[:,1]
→ within-target rank
```

Relevant modules:

```text
src/cpo/features.py
src/cpo/ml_dataset.py
src/cpo/xgboost_model.py
```

Relevant scripts:

```text
04_build_pair_features.py
07_build_ml_dataset.py
08_train_xgboost.py
```

### Recommended model interface

Input:

```python
[
    days_before_acquisition,
    source_is_environment,
    same_cp_family_gx,
    source_gx_positive,
    source_gx_family_count,
    same_cp_family_isolate,
    same_exact_cp_gene_isolate,
    same_species_any,
    same_st_any,
]
```

Output:

```python
model.predict_proba(X)[:, 1]
```

Then rank candidates within each target.

---

## Option B — Integrate candidate generation + XGBoost

Use:

```text
acquisition target
        ↓
candidate source generation
        ↓
feature engineering
        ↓
XGBoost ranking
```

Relevant modules:

```text
src/cpo/acquisition_cases.py
src/cpo/candidate_sources.py
src/cpo/features.py
src/cpo/ml_dataset.py
src/cpo/xgboost_model.py
```

---

## Option C — Integrate the network

Required upstream outputs:

```text
XGBoost candidate rankings
+
WGS-labelled candidate pairs
```

Relevant scripts:

```text
22_build_transmission_network.py
23_analyze_transmission_network.py
24_plot_transmission_network.py
```

Useful outputs:

```text
transmission_network_edges.csv
network_source_hubs.csv
network_target_source_profiles.csv
network_target_projection.csv
```

---

## Option D — Integrate the LLM council

Only use this if the project specifically requires automated evidence synthesis.

Use:

```text
src/cpo/council_v2/
```

and:

```text
scripts/12_run_llm_council_v2.py
```

Do **not** start from:

```text
src/cpo/council/
```

or:

```text
src/cpo/council_final/
```

unless intentionally reproducing the pilot/sensitivity experiments.

---

# 31. What each component is responsible for

A useful mental model is:

```text
Candidate generation
    ↓
Who could have been a source?

XGBoost
    ↓
Which candidates should be investigated first?

WGS
    ↓
Which relationships have direct genomic support?

Network
    ↓
Are these links isolated or part of shared hospital
dissemination structures?

LLM Council
    ↓
Can the evidence be summarised into a structured,
conservative interpretation?
```

The components are intentionally separated.

---

# 32. What NOT to claim

Do not describe:

```text
XGBoost score
```

as:

```text
probability that transmission occurred
```

Do not describe:

```text
WGS link
```

as:

```text
proof that patient A directly infected patient B
```

Do not describe:

```text
network hub
```

as:

```text
confirmed superspreader
```

Do not describe:

```text
LLM council result
```

as:

```text
ground truth
```

Preferred terminology includes:

```text
candidate prioritisation
genomic association
clonal circulation
plasmid circulation
source hypothesis
WGS-supported relationship
recurrent source node
shared dissemination cluster
```

---

# 33. Minimal reproducibility checklist

A successful rerun should approximately recover:

```text
[ ] 8 acquisition targets

[ ] 1,274 candidate pairs

[ ] 1,253 strictly prior pairs

[ ] 21 same-day ambiguous pairs

[ ] 1,021 WGS-evaluable pairs

[ ] 14 direct prior WGS links

[ ] 3 clonal links

[ ] 11 plasmid links

[ ] 999 ML candidate pairs

[ ] 14 ML-positive pairs

[ ] Augmented Hit@5 = 1.00

[ ] Median best-positive rank ≈ 1.5

[ ] 336 aggregated network edges

[ ] 82 network source nodes

[ ] 4 genomic hub source nodes

[ ] T0058/T0080/T0081 shared patient-associated cluster

[ ] T0124 environment-associated clonal pattern
```

Small differences after code refactoring should be investigated rather than silently accepted.

---

# 34. Recommended order for new contributors

If you are new to the repository:

```text
1. Read this README
2. Run scripts 01–06
3. Verify all checkpoint counts
4. Run scripts 07–09
5. Verify XGBoost metrics
6. Run scripts 22–24
7. Inspect network outputs
8. Only then run the optional Council-v2 experiments
```

For an FYP project that only connects to the ML model:

```text
Start with Sections 12–14 and Section 30 of this README.
```

You do not need to run the LLM council.

---

# 35. Main methodological limitations

The current study contains only:

```text
8 independent acquisition cases
```

and only:

```text
14 WGS-linked source–target pairs
```

The 999 ML rows are candidate pairs and should **not** be interpreted as 999 independent clinical observations.

Other limitations include:

- incomplete target WGS for T0016;
- uncertain exact patient-location overlap in some source records;
- lack of validated bed-level longitudinal contact data in the current dataset;
- absence of WGS linkage does not guarantee absence of true transmission if sampling is incomplete;
- genomic relatedness does not establish direct transmission direction;
- unknown and mismatched laboratory values can both map to 0 in current concordance features;
- the XGBoost score is not calibrated as a transmission probability.

---

# 36. Current research direction

The next planned validation uses an independent KPC outbreak dataset containing detailed bed-level patient movement and microbiological information.

The proposed design is:

```text
patient movement + microbiology
        ↓
AI transmission predictions
        ↓
freeze predictions
        ↓
reveal previously hidden WGS
        ↓
external validation
```

The goal is to test whether routinely available clinical data can identify likely transmission sources, clusters and high-risk nodes **before WGS results become available**.

---

# 37. Summary for developers

If you only need to interface with the current XGBoost model:

```text
INPUT
9 numeric engineered features

MODEL
XGBClassifier

OUTPUT
predict_proba(X)[:,1]

INTERPRETATION
candidate-prioritisation score

POSTPROCESSING
sort descending within target
```

If you need transmission-network outputs:

```text
INPUT
ranked source–target candidates
+
WGS linkage information

OUTPUT
source hubs
shared source nodes
target clusters
patient-vs-environment profiles
```

If you need LLM-based evidence synthesis:

```text
USE
src/cpo/council_v2/

DO NOT USE AS PRIMARY
src/cpo/council/
src/cpo/council_final/
```

---

# 38. Final note

This repository is designed as a **decision-support and research framework**, not an autonomous clinical decision system.

The intended evidence hierarchy is:

```text
WGS evidence
>
microbiology + chronology
>
ML candidate ranking
>
LLM evidence synthesis
```

All outputs should therefore preserve uncertainty and be reviewed together with clinical epidemiological context.
