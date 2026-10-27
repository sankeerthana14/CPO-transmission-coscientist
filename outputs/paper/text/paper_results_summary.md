# Paper-ready results summary

## Clinical acquisition-source analysis

Eight patients acquired a carbapenemase gene during hospitalisation. Direct WGS-supported circulation patterns were identified in 4/8 cases (50.0%). These comprised 2 patient-associated plasmid cases, 1 patient-associated clonal case, and 1 environment-associated clonal case. 3/8 targets had WGS available but no direct prior clonal or plasmid-linked candidate, while 1/8 lacked target WGS and remained unresolved.

## Machine-learning prioritisation

The augmented XGBoost candidate-ranking model achieved a pooled AUROC of 0.963 and AUPRC of 0.311. Mean reciprocal rank was 0.688; Hit@1, Hit@3 and Hit@5 were 0.50, 0.75, and 1.00, respectively. The median best-positive rank was 1.5. The most influential global SHAP feature was `same_exact_cp_gene_isolate`.

## LLM council reproducibility

Across five repeated executions of the frozen Council-v2 configuration, mean end-to-end agreement with the deterministic WGS reference was 0.875, while mean agreement among successfully completed cases was 1.000. Repeated executions were treated as inference-stability measurements rather than independent clinical observations.

## Model-role ablation

All six possible assignments of the three LLMs to the Investigator, Biomedical Expert and Causal Critic roles were evaluated. End-to-end agreement ranged from 0.625 to 0.875. The prespecified Qwen3-8B Investigator / MedGemma 1.5-4B Biomedical Expert / Llama 3.1-8B Causal Critic configuration achieved 0.875 end-to-end agreement and was tied for the best-performing assignment.
