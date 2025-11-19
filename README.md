# Smart Infrastructure Complaint Intelligence

This Streamlit platform triages civic complaints in real time by combining fine-tuned transformers, spaCy NER, and severity heuristics. It delivers analyst-ready insights for municipal control rooms, including issue categorisation, severity/urgency prioritisation, entity extraction, and trend dashboards that highlight hotspots and systemic risks.

## Key Capabilities
- **Issue classification** – BERT-based model augmented with civic keyword heuristics to surface descriptive categories and rationales.
- **Urgency assessment** – MiniLM transformer sentiment head that measures angry/urgent tone while falling back to rule-based severity escalation only when needed.
- **Entity intelligence** – spaCy NER for locations, assets, and hazard terms, rendered inline (`word{NER}`) for quick triage.
- **Operational analytics** – Severity SVM with keyword boosts plus dashboards for time-series, heatmaps, and word clouds to guide field deployments.

## Getting Started
```powershell
conda activate E:\Python\venv
streamlit run app.py
```

Ensure the `models/` directory contains the fine-tuned weights:

| Component | Expected Artefact |
|-----------|------------------|
| Classifier | `models/bert_classifier_fast (1).pt` + tokenizer files |
| NER | `models/spacy_model/en_core_web_sm/en_core_web_sm-3.8.0` |
| Urgency | `models/minilm_urgency_fast/model.pt` (MiniLM weights) |
| Severity | `models/severity_svm_fast.pkl` |

With all components available, the app presents live diagnostics explaining each decision and how the stack prioritises the city-wide complaint queue.
