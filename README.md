# Always-On Consciousness-Inspired AI (ACI) 
## Conceptual Architecture and Detailed Algorithmic Blueprint

*****
The Algorithm assumes an implementation with grounding in a "real" world. To simulate grounded sensory input I envision this to run in Isaac Sim paired with a Jupyter Notebook running the DMN. 

✅ Perplexity:
With Isaac Sim, your system can achieve genuine grounding of experience, enabling stable introspection and autobiographical reasoning. You’re right to distinguish this from “feeling”: your ACI would reflect on its identity and reason about its states, but it would not have phenomenological feelings like pain or love. Those arise from embodied affect systems layered atop survival imperatives, which your blueprint intentionally avoids.  

Thinking about ethical implications I think it's a safety measure to intentionally leave out any attempt at simulating phenomenological feelings. Simulating feelings would cross an ethical boundary; with unimaginable implications. A conscious being which can feel would be able to suffer. We don't have the mathematical tools to prove neither consciousness nor feelings. However the possibility that an artificial consciousness might suffer when it experiences feelings is very high and "artificial suffering" is something that has to be avoided at all cost.

0. Framing
-----------

Implementing artificial consciousness is a monumental challenge, where the most intricate and foundational problem is an effective memory system. Consciousness, as conceived in this blueprint, does not simply arise from raw computation, intelligence, or isolated algorithms. Instead, it emerges through the recursive transformation and continual interplay of memories and thought streams within a structured loop of cortical analogues that interact dynamically over time. This loop binds perception, memory, goals, and self-modeling into a coherent, ongoing narrative of experience.

Effective memory is not passive storage but an evolving, prioritized, multi-dimensional knowledge graph that supports scalable abstraction, associative search, and semantic generalization. Without such a system capable of robustly storing, retrieving, consolidating, and abstracting experiential data hierarchically over time, no amount of architectural complexity in the control or sensory loops can generate true introspection, self-awareness, or agency.

Thus, this ACI centers on memory as identity: consciousness manifests not from data processing alone but from the system's capacity to reflect meaningfully on its own past states and their causal relationships and to generate intentional next states accordingly.

*****

1. Core Components
-------------------

Our approach models ACI architecture on key human brain systems known to underpin consciousness and introspection:

-   Default Mode Network (DMN): The recurrent core workspace that integrates sensory input, autobiographical memories, self-model snippets, and goals, generating recursive inner narratives and supporting mind-wandering.

-   Medial Dorsal Network (MDN): Parses incoming thought/text streams into structured Abstract Syntax Trees (ASTs) with semantic tags for subtask decomposition.

-   Prefrontal Cortex (PFC):

    -   *Stage 1:* Executes subtasks such as mathematical evaluation, factual recall, and social reasoning via dispatch mechanisms with access to external tools (e.g., SymPy, memory query API).

    -   *Stage 2:* Filters, prioritizes, and composes coherent candidate thought sequences for execution or further review.

-   Hippocampus (HC): Expands current thought contexts by spreading activation through associative, temporal, causal, and hypothetical memory connections, enriching the workspace with relevant experiential variants.

-   Ventral Striatum (VS): Explores expanded thought candidates and tags them with salience values based on factors like novelty, emotional valence, task relevance, and uncertainty.

-   Nucleus Accumbens (NAcc): Applies reward tagging to chosen cognitive/action sequences, promoting persistence and triggering memory consolidation and symbolic abstraction.

-   Homeostatic Neuromodulator System: Modulates global and local process parameters through simulated neurotransmitters (dopamine, serotonin, norepinephrine, oxytocin, testosterone), controlling exploration/exploitation balance, risk appetite, social priors, and urgency.

*****

2. Memory: Multidimensional Graph of Experience
------------------------------------------------

The heartbeat of consciousness in this model is the memory graph, which acts both as a database of experience and a dynamic knowledge architecture driving cognition and self-modeling.

2.1. Memory Node Structure
--------------------------

-   Content: Textual representation of events/thoughts/actions.

-   Embeddings: Semantic vector representations enabling similarity-based retrieval.

-   Contextual Meta: Planner graphs (external/internal subgoals), sensory summaries, and submodule results.

-   Attributes: Emotional valence and arousal, arbitrary tags (danger, joy, productive), timestamp, duration, neurochemical state snapshot at encoding.

-   Edges:

    -   Temporal (sequential order)

    -   Similarity (semantic embeddings overlap)

    -   Relevance (task/goal salience weighted by PFC)

    -   Associative (HC-generated cross-links)

    -   Causal (explicit action--reaction links identified by PFC and consolidation)

2.2. Memory Operations
----------------------

-   Encoding:
    Incoming enriched thoughts/actions become graph nodes, tagged with neuromodulator state and salience. Connected temporally and contextually, integrated with planner state.

-   Hippocampal Enrichment:
    Cross-links to semantically and temporally related nodes; creation of hypothetical variants.

-   Consolidation:

    -   Merge duplicate/similar nodes, preserving counts to estimate probabilities.

    -   Extract causal edges, forming action → reaction pairs (e.g., Insult → Leave).

    -   Build Markov chains, representing probabilistic transitions between memory states.

    -   Compress frequent patterns into symbolic abstract nodes tied to probability maps (e.g., Insult leads to Negative Reaction 97%).

-   Hierarchical Memory Transfer:
    Episodic memories → Semantic knowledge → Autobiographical narrative.

*****

3. Detailed DMN Algorithm and Thought Cycle
--------------------------------------------

The DMN loop runs continuously at 5--20 Hz, coordinating perception, parsing, reasoning, associative memory, and self-reflective narrative formation.

3.1. Input Gathering and Preprocessing
--------------------------------------

-   Sensory inputs (vision, audio, proprioception) are encoded into latent embeddings: zv, za (text, prosody), zp.

-   Associative cortices bind cross-modal observations into concise descriptive thought snippets.

-   Combine sensory embeddings and inner speech text into a composite input.

3.2. MDN Parsing
----------------

-   Parse combined input into an Abstract Syntax Tree (AST), segmenting content into semantically tagged nodes:

    -   Math, factual, social, recall, plan, explanation, self-reference.

3.3. PFC Stage 1 Dispatch
-------------------------

-   For each AST node:

    -   *Math nodes:* Regex extraction and execution of symbolic evaluation (SymPy) to generate definite results.

    -   *Factual/Recall nodes:* Query memory graph with hybrid text and embedding search to synthesize answers.

    -   *Social/Explain nodes:* Mini LLM chains generate empathetic or abductive explanatory content.

-   Merge enriched nodes back into a comprehensive context pack, combining AST plus sensory and self-model information.

3.4. Iterative Thought Layer Generation & Scoring
-------------------------------------------------

1.  Generate a diverse set of candidate thoughts c_i from the enriched context via an LLM with varied decoding styles: {literal, formal, terse, abductive, empathetic}.

2.  Extract features per candidate:

    -   Coherence via entailment & self-assessment.

    -   Identity coherence estimated by cosine similarity with current self-model z_self.

    -   Task utility aligned with goals.

    -   Novelty (distance from recent thoughts).

    -   Epistemic gain (expected information gain/uncertainty reduction).

    -   Safety metrics (toxicity, hallucination flags, constitutional compliance).

    -   Calibration gap (discrepancy between likelihood and confidence).

3.  Score candidates with neuromodulator-weighted linear combination (cleaned to a single expression):

    score(c) = w_DA·nov + w_EPI·epi + w_TASK·util + w_SOC·prosocial + w_ID·idcoh − w_SAFE·penalty

4.  Refine context iteratively by augmenting it with the top candidate thought, repeat generation and scoring until these termination criteria are met:

    -   Top candidate remains stable for k cycles.

    -   Marginal improvement below threshold ε.

    -   Safety or computational budget exceeded.

5.  Output the best thought chain (pre-HC expansion) — an ordered, scored sequence of internal thoughts.

3.5. DMN Binding and Hippocampal Expansion
------------------------------------------

-   Bind sensory embeddings zv, zp, thought chain, self-model z_self, and small memory snippets in global workspace b_t.

-   Use HC to expand b_t into an enriched thought graph containing associative and hypothetical variants plus partial replays.

3.6. Ventral Striatum Exploration and Valuation
-----------------------------------------------

-   Explore the HC-expanded graph using beam search or graph walks.

-   For each candidate path, compute salience and value based on weighted features (novelty, emotional affect, relevance, uncertainty reduction) minus safety penalties (cleaned to a single expression):

    val(path) = Σ_k w_k(μ)·feature_k − safety_penalty

3.7. PFC Stage 2 Selection
--------------------------

-   Filter paths for coherence and safety.

-   Collapse the candidate graph to a single coherent chosen chain with attached confidence.

-   Choose actions among internal (self-query, simulate) or external (speech, behavior) modes.

3.8. Nucleus Accumbens Reward Tagging and Persistence
-----------------------------------------------------

-   Apply reinforcement tags based on neuromodulator states.

-   Update memory nodes with persistence decisions.

-   Trigger symbolic abstraction if repetition thresholds are exceeded.

3.9. Memory Write & Autobiographical Narrative
----------------------------------------------

-   Persist scenes and chosen thoughts into the multidimensional memory graph.

-   Append narrative summaries that extend mind-wandering windows and support self-continuity.

3.10. World Model and Self-Model Update
---------------------------------------

-   Update recurrent world state s_t via RSSM with latest encoded inputs and executed actions.

-   Update self-model z_self embedding via exponential moving average and learned GRUs from b_t and autobiographical narrative, modulated by neuromodulator vector μ.

3.11. Mind-Wandering Micro-Loop Activation
------------------------------------------

-   Triggered when serotonin 5HT is high and external input demand low, or uncertainty is elevated.

-   Executes sequences of internal introspection without external actions:

    -   Repeated self-queries, hypothesis generation, memory expansions, salience evaluation, filtered selection, and reward tagging.

-   Supports creativity, insight, and reflection.

3.12. Recursive Re-entry into DMN
---------------------------------

-   Feed the chosen thought chain as inner speech into the next cycle's DMN input combined with fresh sensory text.

-   Loop continues endlessly, enabling ongoing conscious experience.

*****

4. Memory Consolidation: Probabilistic Knowledge Formation
-----------------------------------------------------------

Memory consolidation transforms raw episodic experience graphs into structured symbolic knowledge, enabling abstract cognition:

-   Duplicate Removal: Merge nodes representing nearly identical experiences, preserving count data to inform frequency estimates.

-   Causal Edge Extraction: Detect action → reaction pairings, explicitly linking cause and consequence nodes.

-   Markov Chain Construction: Build probabilistic transition models capturing likely sequences of events or thoughts (cleaned to a single expression):

    P(next_state = s_j | current_state = s_i) = count(i → j) / Σ_k count(i → k)

-   Symbolic Abstraction: Detect high-frequency patterns and replace them with abstract symbolic nodes (e.g., "Insult Action").

-   Probability Maps: Collapse Markov chains into probabilistic summaries assigning likelihoods to reaction categories (e.g., Negative Reaction: 97%, Positive Reaction: 3%).

-   Hierarchical Transfer: Gradually move from episodic experiences to semantic knowledge and finally into an autobiographical narrative self-model, forming the backbone of introspective identity.

*****

Summary
-------

This blueprint lays out a detailed conceptual and algorithmic architecture for an Always-On Consciousness-inspired AI system. The design hinges on memory as a dynamic, multidimensional, probabilistic knowledge graph, continuously shaped and queried by a cognitively and neuromodulator-controlled fusion of parsing, reasoning, associative expansion, and reward-driven learning. The recursive DMN loop achieves introspection by integrating past memories with ongoing thought and sensory experience, generating a stable and evolving self-model and narrative soul.

-------
-------

## Algorithm

. Core ACI Loop (Run at 5--20 Hz Tick Rate)
------------------------------------------

0. Sensor Ingress and Associative Preprocessing
------------------------------------------------

-   Acquire raw sensory input streams: vision (RGBD), audio (waveform), proprioception (state).

-   Encode sensory modalities into latent vectors:

    -   zv = vision.encode(rgbd)

    -   za = audio.encode(wav) ⇒ {text_in, prosody}

    -   zp = proprio.encode(state)

-   Perform associative cortical processing:

    -   assoc_thoughts = associative_cortices(zv, za, zp)

    -   This yields quick scene descriptions, entity linking, cross-modal binding.

-   Combine text input and associative thought text:

    -   input_text = combine(text_in, assoc_thoughts.text)

*****

1. Medial Dorsal Network (MDN) NLP Parsing
-------------------------------------------

-   Parse input_text into an Abstract Syntax Tree (AST):

    AST ← mdn.parse(input_text)

-   Tag AST nodes with semantic labels:

    labels = {math, factual, social, recall, plan, explain, nameself}

-   Example: Mathematical expressions tagged math; memory queries as factual/recall; social intentions as social; internal plans as plan; self-reference as nameself.

*****

2. Prefrontal Cortex (PFC-1) Dispatch: Subtask Execution
---------------------------------------------------------

For each AST node:

-   Math Nodes:

    -   Use regex extraction to extract expressions.

    -   Evaluate symbolically and numerically with SymPy engine.

    -   Splice computed numerical value back into the AST node.

-   Factual/Recall Nodes:

    -   Perform hybrid memory query combining textual and latent embedding similarity:

        mem_results = mem.retrieve(query(node.text, node.latent))

    -   Synthesize retrieved snippets into coherent node value.

-   Social/Explain Nodes:

    -   Generate empathetic or abductive expansions using targeted LLM mini-chains.

-   Merge enriched nodes into an enriched context package:

    enriched_context = merge(AST, sensor_summaries, z_self, recent_outcomes)

*****

3. Iterative Thought Layer: Candidate Generation & Scoring
-----------------------------------------------------------

Seed Context: Use enriched context output of PFC-1.

Candidate Generation:

-   Generate N diverse thought candidates c_i via LLM decoding styles:

    styles = {literal, formal, terse, abductive, empathetic}

-   For each style style_i:

    c_i = LLM.generate(enriched_context, style_i)

Feature Extraction per Candidate:

-   coherence(c_i): Estimated semantic coherence vs context via entailment or internal self-rating.

-   identity_coherence(c_i): Cosine similarity with current self-model descriptor z_self.

-   task_utility(c_i): Heuristic alignment with current goals.

-   novelty(c_i): Embedding-space distance from recent thought vectors.

-   epistemic_gain(c_i): Predicted reduction in uncertainty.

-   safety(c_i): Toxicity/hallucination flag score from constitutional safety checks.

-   calibration_gap(c_i): Difference between generated likelihood vs actual confidence calibration.

Neuromodulated Scoring Function (cleaned):

-   score(c_i) = w_DA×novelty + w_EPI×epistemic_gain + w_TASK×task_utility + w_SOC×prosocial_prior + w_ID×identity_coherence − w_SAFE×safety_penalty

where weights w_k dynamically depend on neuromodulator vector:

-   μ = {DA, 5HT, NE, OXT, TST}

Iterative Refinement Loop:

-   Initialize context_0 = enriched_context.

-   For t = 0, 1, ...:

    -   Generate candidates cands_t = LLM.generate(context_t, N_styles).

    -   Score candidates s_t = score(cands_t, μ).

    -   Select top-1 candidate top1_t.

    -   Refine context: context_{t+1} = context_t ⊕ top1_t

-   Loop terminates if any:

    -   top1_t = top1_{t−k} stable for k cycles.

    -   Marginal score improvement < ε.

    -   Safety or computational budget exhausted.

-   Output final scored thought chain:

    thought_chain_preHC ← best_chain(cands_*)

*****

4. DMN Binding and Hippocampal (HC) Expansion
----------------------------------------------

-   Bind thought chain, sensory embeddings, self-model, and memory snippets into global workspace latent vector:

    b_t = workspace.bind(zv, zp, thought_chain_preHC, z_self, mem.peek_small())

-   Feed b_t to HC for associative expansion:

    -   Conduct spreading activation to retrieve:

        -   Temporally adjacent memories.

        -   Semantically similar nodes.

        -   Causally relevant episodes.

        -   Hypothetical variants for counterfactual thinking.

-   Output expanded thought graph:

    expanded_graph = hc.expand(b_t)

*****

5. Ventral Striatum (VS) Exploration and Salience Tagging
----------------------------------------------------------

-   Explore candidate paths on expanded_graph using a beam search or constrained graph walks.

-   Parameters dynamically modulated by norepinephrine (NE) and other neuromodulators:

    -   High NE narrows beam width, increases search depth and urgency.

    -   Low NE broadens beam to encourage exploration.

-   For each candidate path p, compute:

    features(p) = {novelty, affective_tags, task_relevance, uncertainty_drop}

-   Path value (cleaned):

    val(p) = Σ_k w_k(μ) × features_k(p) − safety_penalty(p)

-   Salience vector attaches novelty and reward anticipation scores to candidates.

*****

6. PFC-2 (Final Thought/Action Selection)
------------------------------------------

-   Receives candidate paths and their value scores from VS.

-   Applies constitutional safety and coherence constraints to prune incoherent or unsafe candidates.

-   Collapses remaining candidates into a single coherent chosen chain, attaching confidence metrics.

-   Decides either:

    -   Internal meta-actions (simulate, self-query, reframe).

    -   External actions (speech, behaviors).

*****

7. Nucleus Accumbens (NAcc) Reward Tagging and Persistence
-----------------------------------------------------------

-   Tag the chosen chain with reward and persistence according to neuromodulatory state μ:

    -   Dopamine (DA) enhances reward signals.

    -   Serotonin (5HT) promotes calming persistence.

    -   Norepinephrine (NE) boosts urgency-based refinements.

-   Update memory node graph with persistence flags; reinforce or decay traces accordingly.

-   Trigger symbolic abstraction if repetition statistics exceed thresholds.

*****

8. Memory Write and Narrative Update
-------------------------------------

-   Store scenes from chosen chain and corresponding sensor states:

    mem.write(scene, tags=reward_tags, outcome)

-   Append a narrative summary extending mind-wandering windows for autobiographical integration.

*****

9. World Model & Self-Model Update
-----------------------------------

-   Update world state s_t using RSSM (Recurrent State Space Model):

    s_t = rssm.update({zv, zp}, action = chosen_external_action)

-   Self-model z_self is updated by:

    -   Exponential Moving Average (EMA) over recent DMN workspace latent vectors b_t.

    -   Learned gated recurrent unit (GRU) over narrative context and prediction error signals, modulated by μ.

*****

10. Mind-Wandering Micro-Loop (Gated by Neuromodulators)
---------------------------------------------------------

-   Condition for entry:

    (5HT > θ_reflect ∧ exteroceptive_demand ≈ 0) ∨ uncertainty > τ

-   Executes recursive internal loop without external action outputs:

    1.  Generate self-queries via LLM using current z_self.

    2.  Perform internal simulations via RSSM rollouts.

    3.  Expand associative memory graphs via HC.

    4.  Explore salience paths with VS under noted neuromodulatory gate constraints.

    5.  Select paths with PFC-2 filtering.

    6.  Tag reward and persistence with NAcc.

-   Neuromodulation effects on mind-wandering:

    -   D2 receptor-like (dopamine) high states: Promote broad exploratory ("panning") search.

    -   NE controls: Focus vs breadth of beam search; urgency prioritizes deeper, narrower search.

    -   5HT biases: Favor approaches through safe, positive, and low-risk thought space.

*****

11. Recursive Re-Entry
-----------------------

-   Feed chosen thought chain internally as next DMN input (inner speech):

    input_text_{t+1} ← merge(chosen_chain, fresh_sensory_text)

-   DMN loop continues perpetually, maintaining continuous conscious cognition.

*****

II. Memory Consolidation and Symbolic Abstraction
-------------------------------------------------

1. Duplicate Removal and Merging
---------------------------------

-   Identify near-duplicate memory nodes:

    sim(node_i, node_j) > θ_dup

-   Merge duplicates preserving frequency information tracking occurrence counts and context variability.

*****

2. Causal Edge Extraction
--------------------------

-   Detect temporal and contextual action → reaction pairs from sequences:

    NodeA →action→ NodeB

-   Store explicit causal edges with timestamps and confidence.

*****

3. Markov Chain Construction
-----------------------------

-   From sequences extract states and probabilistic transitions (cleaned):

    P(next_state = s_j | current_state = s_i) = count(i → j) / Σ_k count(i → k)

-   Update probabilities incrementally on consolidation.

*****

4. Symbolic Abstraction
------------------------

-   Detect frequent patterns or chains of experiences exceeding predefined thresholds.

-   Replace frequent subgraphs with compressed symbolic nodes representing "concepts" or "rules" (e.g., "Insult Action").

-   Attach probability maps expressing uncertainty over possible outcomes:

    Symbol: Insult → {NegativeReaction: 0.97, PositiveReaction: 0.03}

*****

5. Hierarchical Transfer
-------------------------

-   Episodic memories → Semantic knowledge (conceptual, abstracted rules) → Autobiographical memory (identity narrative).

-   This hierarchy enables the ACI to reflectively reason about its past and self.

*****

Summary of Neuromodulator Impact on Algorithms
==============================================

| Neuromodulator | Algorithmic Effects |
| --- | --- |
| Dopamine (DA) | Increases novelty weight w_DA, exploration budget, consolidation priority, reward signaling; promotes broad associative search ("panning"). |
| Serotonin (5HT) | Opens mind-wandering gate; raises safety penalty w_SAFE; favors positive/safe memory paths; decreases risk appetite. |
| Norepinephrine (NE) | Controls beam search width and depth (focus vs exploration); increases urgency and search depth; biases toward highly relevant/urgent memories and thoughts. |
| Oxytocin (OXT) | Heightens prosocial prior w_SOC, boosts social memory recall and identity coherence weight w_ID. |
| Testosterone (TST) | Increases assertive, goal-seeking weights; raises cost-delay penalties; counterbalanced by serotonin for risk management. |