# Vessel-Topology-Aware 3D Multi-Task Network for PE Detection, Localization and Quantification in CTPA

Notebook implementation of the proposed pipeline on the RSNA-STR PE dataset (`stanleyjzheng/rsna-str-tfrec`).

## Run order
| # | Notebook | Purpose |
|---|---|---|
| 00 | `00_common.ipynb` | Config, dataset, `PENet` model, losses, trainer (loaded by the others via `%run ./00_common.ipynb`) |
| 01 | `01_data_preprocessing.ipynb` | TFRecord schema check → volumes → HU/intensity normalisation, resampling, QC, splits |
| 02 | `02_vessel_topology.ipynb` | Vessel segmentation → centreline → vessel graph → candidate embolus ROIs |
| 03 | `03_train_proposed_model.ipynb` | Train full model, pick thresholds on validation |
| 04 | `04_baselines_and_ablation.ipynb` | Experiment A (baselines) and B (ablation) |
| 05 | `05_localization_quantification.ipynb` | Experiment C (localisation) and D (burden quantification) |
| 06 | `06_explainability_robustness_uncertainty.ipynb` | Experiment E (explainability), F (robustness), MC-dropout, final explainable PE map |
| 07 | `07_multiobjective_optimization.ipynb` | Multi-objective optimisation of hyper-parameters and thresholds |

Keep all notebooks in the same folder (they `%run` 00). On Kaggle, upload them together as a dataset or into the working directory.

## Architecture → code
| Block in the diagram | Where |
|---|---|
| 3D preprocessing + QC | 01 |
| Pulmonary vessel segmentation | 02 `segment_vessels` (classical pseudo-segmenter; replace for publication) |
| Vessel centreline + graph, candidate ROIs | 02 `vessel_graph_candidates` |
| Multi-scale 3D encoder (local + vessel + global) | 00 `PENet.e1..e4`, `sample_at` |
| Cross-scale fusion | 00 `PENet.forward` (global-vector-gated scale attention) |
| Vessel-topology attention | 00 `TopoBlock` (hop-distance bias on attention logits) |
| Multi-task heads | 00 detection / severity / burden / localisation decoder / vessel auxiliary / token heads |
| False-positive suppression | 00 `fp_head` + `gaussian_field` gate |
| Uncertainty | 00 `predict(mc=T)` (MC-dropout), 06 |
| Explainable output | 06 Grad-CAM, Grad-CAM++, vessel attention, combined map, `explain_case` |

## Limitations to state in the paper
1. RSNA-STR has no voxel masks, vessel masks or clot volumes. Localisation is trained with slice-level MIL; IoU/Dice are computed along z. Burden is a slice-extent proxy.
2. The vessel segmentation is heuristic. Report its quality or swap in a validated model.
3. Location wording is limited to side and central vs peripheral; lobar naming needs a lobe/artery-branch labeller.
4. Burden classes (low/moderate/high) are a research-derived index, not clinically validated.
