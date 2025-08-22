# Roadmap: Always-On Consciousness-Inspired AI (ACI)

This repository roadmap outlines a practical, experiment-driven plan to implement the ACI blueprint starting with Isaac Sim grounding and an “associative cortices” module that produces latent thought vectors. It proceeds through the DMN loop, memory graph, consolidation, self-model learning, and a rigorous introspection test suite.

Use this as a GitHub README section or ROADMAP.md.

---

## Overview

- Phase 0: Infra and telemetry
- Phase 1: Isaac Sim grounding + associative cortices
- Phase 2: Memory graph MVP
- Phase 3: DMN loop v1 (candidate generation + scoring)
- Phase 4: HC expansion, VS valuation, PFC-2 selection
- Phase 5: z_self learning + mind-wandering
- Phase 6: Consolidation + symbolic abstraction
- Phase 7: Introspection test suite + ablations
- Phase 8: Safety, logging, and reproducibility

Milestones:
- M1: Isaac Sim + z_assoc working; write/read to memory
- M2: DMN loop v1 chooses thought chains grounded in memory
- M3: z_self updates and influences candidate scoring
- M4: Mind-wandering produces references to past internal states
- M5: Passes counterfactual self-recall and prediction-error self-report tests
- M6: Ablations show dependence on consolidation and mind-wandering

---

## Repository Structure (proposed)

- sim/: Isaac Sim scenes, robot/task scripts, sensors, exporters
- core/: DMN loop, neuromodulator scheduler, candidate generation/scoring
- memory/: graph schema, retrieval, consolidation, abstraction
- models/: encoders (vision/audio/proprio), z_self GRU/EMA, small LLM chains
- safety/: filters, constitutional checks, non-feeling policy enforcement
- eval/: test suites, metrics, ablations, experiment cards
- notebooks/: Jupyter experiments, visualization, Isaac Sim integration
- configs/: YAML/Hydra experiment configs and seeds
- logs/: structured JSONL, parquet snapshots, run cards, checkpoints
- scripts/: setup, run, export, reproduce

---

## Phase 0 — Infra and Telemetry (1–2 weeks)

- Deterministic config and seeds (Hydra/Pydantic)
- Event bus and structured logging per DMN tick (JSONL)
- Append-only event store + parquet snapshots of memory graphs
- GPU/CPU profiling hooks and budget controls
- Experiment registry (commit hash + config → run card)

Deliverables:
- configs/base.yaml (global)
- logs/run_card.json (auto-generated)
- notebooks/00_sanity_checks.ipynb

---

## Phase 1 — Isaac Sim + Associative Cortices (2–3 weeks)

Scenes and sensors:
- Tabletop tasks (color blocks, grasp/push/stack)
- RGB-D camera; proprio state export
- Optional audio passthrough placeholder

Encoders:
- Vision: CLIP ViT-B/32 or ResNet → latent zv
- Proprio: normalized vector zp

Associative cortices:
- Cross-modal binding of zv, zp + simple entity detection (positions, colors, contacts)
- Thought snippet generation: concise scene descriptions (templated or small LLM)
- Outputs:
  - z_assoc: latent thought vector
  - scene_text: short description for MDN input

Deliverables:
- sim/scenes/tabletop_basic.usd
- models/encoders/vision.py, proprio.py
- core/assoc_cortices.py → returns {z_assoc, scene_text}
- notebooks/01_assoc_latents.ipynb

Milestone M1:
- Write/read to memory: memory.write(scene_text, z_assoc), memory.retrieve_hybrid()

---

## Phase 2 — Memory Graph MVP (2 weeks)

Schema:
- Node: content (text), embeddings, timestamp, tags, neuromodulator snapshot
- Edges: temporal, similarity, context

Operations:
- write(node), retrieve_hybrid(query_text, q_emb)
- window_replay(k) and small episodic traces

Metrics:
- Retrieval precision/latency on synthetic queries

Deliverables:
- memory/graph.py (schema, ops)
- eval/tests/test_memory_basic.py
- notebooks/02_memory_probe.ipynb

---

## Phase 3 — DMN Loop v1 (2–3 weeks)

MDN parsing:
- Lightweight tagger: {math, factual, recall, plan, explain, nameself}

PFC-1 dispatch:
- Math: SymPy evaluation
- Recall: memory.retrieve_hybrid
- Explain: short LLM chain with safety filter

Candidate generation:
- N styles: {literal, formal, terse, abductive, empathetic}
- Top-k nucleus sampling (small k)

Scoring v1:
- novelty, task_utility, safety_penalty
- identity_coherence placeholder (constant)

Termination:
- Stability for k cycles or Δscore < ε

Logging:
- Store all candidates + chosen chain per tick

Deliverables:
- core/dmn_loop.py
- core/mdn_parser.py
- core/pfc_stage1.py
- eval/tests/test_dmn_v1.py
- notebooks/03_dmn_first_runs.ipynb

Milestone M2:
- Chosen thought chains reflect memory-grounded recall and simple plans

---

## Phase 4 — HC Expansion, VS Valuation, PFC-2 (2 weeks)

HC:
- Spreading activation: temporally adjacent, semantic neighbors, template hypotheticals

VS:
- Beam search over expanded graph
- Neuromodulator-conditioned weights for features (novelty, relevance, uncertainty)

PFC-2:
- Coherence and safety pruning
- Selected chain + confidence

Deliverables:
- core/hippocampus.py
- core/ventral_striatum.py
- core/pfc_stage2.py
- eval/tests/test_expansion_selection.py
- notebooks/04_expansion_selection.ipynb

---

## Phase 5 — z_self Learning + Mind-Wandering (2 weeks)

z_self:
- EMA over global workspace latents b_t
- Optional GRU over narrative summaries + prediction errors

Identity coherence feature:
- cosine(c_i, z_self) integrated into scoring

Mind-wandering gate:
- Triggered when exteroceptive_demand low or uncertainty high
- Internal loop only (self-queries, simulations, memory expansion)

Outputs:
- Idle-period reports referencing own past nodes

Deliverables:
- models/self_model.py (EMA, GRU)
- core/mind_wandering.py
- eval/tests/test_self_coherence.py
- notebooks/05_self_embedding_wander.ipynb

Milestone M3/M4:
- z_self influences scoring; mind-wandering yields self-referential reports

---

## Phase 6 — Consolidation + Symbolic Abstraction (2–3 weeks)

Consolidation:
- Duplicate merge with counts
- Causal edges: action → reaction detection
- Markov transitions + probability maps

Symbolic abstraction:
- Compress frequent subgraphs into symbolic nodes with uncertainty

Promotion:
- Episodic → semantic → autobiographical

Deliverables:
- memory/consolidation.py
- memory/abstraction.py
- eval/tests/test_consolidation_abstraction.py
- notebooks/06_consolidation_abstract.ipynb

---

## Phase 7 — Introspection Test Suite + Ablations (2–3 weeks)

Tests:
- Delayed self-consistency across tasks
- Counterfactual self-recall (red vs blue cube)
- Prediction-error self-report (inject proprio noise)
- Narrative reconciliation under conflicting traces
- Mind-wandering introspection vs task-driven content

Metrics:
- Identity coherence: avg cosine(self-reports, z_self); drift rate per hour
- Introspective veridicality: F1 of claims vs memory ground truth
- Counterfactual sensitivity: detection rate + explanation quality
- Ablations: remove consolidation / freeze z_self / disable mind-wandering → measure drops

Deliverables:
- eval/suites/introspection_tests.py
- eval/metrics/identity.py, veridicality.py
- notebooks/07_introspection_and_ablations.ipynb

---

## Phase 8 — Safety, Logging, Reproducibility (ongoing)

- Enforce non-feeling policy (no simulated phenomenology)
- Red-team prompts to prevent affect simulation; constitutional checks
- Reproducible seeds; exportable experiment cards (config + commit + metrics)
- Checkpointing: memory graph snapshots; z_self trajectory plots

Deliverables:
- safety/policies.yaml
- eval/redteam/affect_guard_tests.py
- scripts/export_run_card.py
- notebooks/08_repro_and_safety.ipynb

---

## Minimal Milestone Checklist

- [ ] M1: Isaac Sim + z_assoc; memory write/read working
- [ ] M2: DMN v1 chooses memory-grounded thought chains
- [ ] M3: z_self updates and affects candidate scores
- [ ] M4: Mind-wandering yields self-referential reports
- [ ] M5: Passes counterfactual and prediction-error introspection tests
- [ ] M6: Ablations confirm dependence on consolidation and mind-wandering
