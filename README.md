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

-   Homeostatic Neuromodulator System: Modulates global and local process parameters through simulated neurotransmitters (dopamine, serotonin, norepinephrine, oxytocin, testosterone, histamin, orexin), controlling exploration/exploitation balance, risk appetite, social priors, wakefulness, tiredness, and urgency.

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

1. Episodic Memory Storage   
   Write chosen chain to MemoryTape.  
   Link edges through MemoryGraph (temporal, similarity, causal, goal relevance).

2. Multi-Relational Embedding Space  
   Every time we add edges, we encode the updated working memory graph into a latent vector space:  

    Each MemoryRecord has:  
    content_embedding: semantic representation (text, sensor fusion, context).  
    Each edge type is assigned a transformation vector or operator:  
    ```
    T_temporal = unit displacement on time axis.  
    T_similarity = close embedding alignment.  
    T_causal = directional translation (TransE-style: cause + relation ≈ effect).  
    T_relevance = weighted axis conditioned on current goal embedding. 
    ``` 
    Combined embedding:  
    `z_node* = content_embedding ⊕ Σ (relation_weight × T_relation)`  
    > This means nodes aren’t just content, they are content + relational signature.  
    This gives us a multi-relational latent manifold where:  
    Radius search retrieves nodes connected by one type of relation (similarity sphere).  
    Higher-dimensional sphere query retrieves clusters that satisfy all relations simultaneously (similar, temporally close, causally relevant, and salient).

    👁 Analogy: This is like mixing word embeddings with knowledge graph embeddings (TransE/RotatE/ComplEx) and then projecting them into a single working latent space for the HC to search.

3. Autobiographical Narrative Storage (expanded)
    Narrative nodes now also get multi-relational embeddings:

    Use a set transformer or GRU over the embeddings of all linked episodic memories.
    Store narrative as:  
    summary_text (LLM-generated).  
    narrative_embedding = pooled latent vector representing both memories + relation types.  
    
    This allows queries like:  
    > “Find all narratives in which I was under social stress and learned something new.”
    
    By executing a high-dimensional sphere query in relation space combining  
    `tag=stress, relation=causality, and goal=learning.`
    
4. Algorithm Insert (Updated 3.9 Section)
    python  

```python
def memory_write_and_narrative(scene, chosen_chain, μ, z_self, goals):
    # Step 1. Storage
    mem_id = MemoryTape.append(
        timestamp=now(),
        content=chosen_chain,
        sensory_snapshot=scene,
        content_embedding=encode(scene, chosen_chain),
        tags={μ, goals}
    )

    # Add graph edges
    MemoryGraph.add_temporal_edge(mem_id-1, mem_id)
    MemoryGraph.add_similarity_edges(mem_id, SIM_RADIUS)
    MemoryGraph.add_goal_edges(mem_id, goals)
    
    if detects_action_outcome_pair(chosen_chain):
        MemoryGraph.add_causal_edge(action_id, outcome_id)

    # Step 2. Multi-relational embedding update
    rel_vector = combine_relations(mem_id, MemoryGraph)
    latent_embedding = mem_id.content_embedding + rel_vector
    MemoryGraph.update_embedding(mem_id, latent_embedding)

    # Step 3. Narrative summary
    recent_ids = MemoryTape.last(N_CHUNK)
    if summarize_condition(recent_ids):
        summary, narrative_embedding = generate_narrative(recent_ids, z_self)
        NarrativeChain.add(
            time_range=(recent_ids[0].t, recent_ids[-1].t),
            linked_memories=recent_ids,
            summary_text=summary,
            narrative_embedding=narrative_embedding,
            self_model_snapshot=z_self,
            goals_snapshot=goals,
            μ_snapshot=μ
        )

    return mem_id
```  
  

> ✅ Perplexity: One unified latent space means the hippocampus doesn’t need to separately search similarity, causality, time, goals — it just queries a manifold ball.
>
> Sphere queries → HC.expand(b_t, radius=r, relation_weights=W) can dynamically tune which relations matter (like “bias toward causality vs similarity”), modulated by neuromodulators.
>
> Narrative embeddings make autobiographical reasoning as searchable as episodic memories.
>
> Autobiographical records can be clustered → the agent can ask “show me all phases of life where I pursued exploration goals.”


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

1. Memory Consolidation: Probabilistic Knowledge Formation
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

## Additional Introspective Context: DMN Control Flow Graph (CFG)

One extension to the ACI design is to explicitly represent the **DMN loop as a Control Flow Graph (CFG)**.  
This CFG encodes each step of the algorithm as a *node with identifiers, inputs, outputs, and function role*.  

By maintaining a structured CFG definition (e.g., in YAML), the system can:

- **Feed active CFG node tags into LLM invocations** as *meta-context*, so the model always knows:
  - *Which stage of the DMN loop it is operating in.*
  - *What its current purpose is (e.g., hypothesis generation vs salience valuation).*
  - *What constraints or output type is expected.*

- **Enhance process self-awareness**:
  - The system carries "awareness" of being in *Candidate Generation*, *HC Expansion*, or *VS Valuation* phase.
  - Prevents role confusion and encourages consistent LLM behavior.

- **Enable explicit traceability**:
  - Each memory node or thought can be tagged with the CFG node ID that produced it.
  - Later, autobiographical narrative can reflect not just *what was thought*, but *how it was produced*.

- **Stabilize cognitive dynamics**:
  - Feeding CFG meta information constrains the LLM to follow algorithmic intent.
  - Acts as a role-encoding signal, similar to "system prompts" that anchor behavior.

- **Provide a basis for metacognitive reasoning**:
  - By referring to its own CFG, the system can simulate/debug itself:
    > "This candidate emerged during PFC-1 hypothesis expansion while DA was high, but it was pruned at VS valuation."

---

When invoking an LLM sub-process (e.g., to generate a thought, evaluate coherence, or narrativize memory),  
prepend a meta-context header:

```yaml
DMN_State:
  Node: "3.4_Candidate_Generation"
  Purpose: "Generate abductive hypotheses"
  Neuromodulators: {DA: 1, NE: 0.7, 5HT: 0.8}
```

This makes the LLM *process-aware* of **where it is in the loop** and **what its functional role is**.

---

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
    >   Use regex extraction to extract mathematical expressions.

-   Tag AST nodes with semantic labels:

    labels = {math, factual, social, recall, plan, explain, nameself}

-   Example: Mathematical expressions tagged math; memory queries as factual/recall; social intentions as social; internal plans as plan; self-reference as nameself.

*****

2. Prefrontal Cortex (PFC-1) Dispatch: Subtask Execution
---------------------------------------------------------

For each AST node:

-   Math Nodes:

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


- **Bind Workspace Context**

  Bind thought chain, sensory embeddings, self-model, and memory snippets into a global workspace latent vector:

  ```python
  b_t = workspace.bind(zv, zp, thought_chain_preHC, z_self, mem.peek_small())
  ```

  This latent vector `b_t` represents the current conscious context.

---

- **HC Expansion Using High-Dimensional Latent Geometry**

  Instead of symbolic spreading or beam walks, HC queries the *multi-relational latent space* directly:

  1. **Define Relation Operators**
     
     Each relation type is represented as a displacement vector or transformation in latent space:
     - `T_temporal`
     - `T_similarity`
     - `T_causal`
     - `T_relevance`

  2. **Compose Multi-Relation Query Vector**
     
     ```python
     q = b_t + α·T_temporal + β·T_similarity + γ·T_causal + δ·T_relevance
     ```
     
     Relation weights `{α, β, γ, δ}` are dynamically modulated by neuromodulator state μ:
     - DA (dopamine): ↑ novelty & similarity bias  
     - NE (norepinephrine): ↑ causality & urgency  
     - 5HT (serotonin): ↑ safe & prosocial relevance  

  3. **Hypersphere / Multi-Radius Search**
  
     The HC performs radius queries in latent space instead of graph walks:

     ```python
     candidates = VectorDB.radius_search(center=q, radius=R)
     ```

     Or multi-radius queries across different relation axes:
     - Temporal window (time adjacency)  
     - Semantic radius (content similarity)  
     - Causal projection cone (directed offsets)  
     - Goal alignment radius (task relevance)  

     Results are merged and weighted according to relation-type proximity.

  4. **Generate Hypothetical Variants**
     
     For each retrieved candidate, HC can perturb embeddings to simulate counterfactuals:
     
     ```python
     hypothetical = candidate_embedding + δ·perturb(goal_focus)
     ```
     
     These virtual nodes represent “what if” alternatives.

  5. **Assemble Expanded Thought Graph**
     
     ```python
     expanded_graph = build_subgraph(candidates + hypotheticals, relation_weights={α,β,γ,δ})
     ```
     
     Edges are weighted by geometric closeness to `q`. Virtual counterfactual nodes are flagged but available for downstream exploration.



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


11. Recursive Re-Entry
-----------------------

-   Feed chosen thought chain internally as next DMN input (inner speech):

    input_text_{t+1} ← merge(chosen_chain, fresh_sensory_text)

-   DMN loop continues perpetually, maintaining continuous conscious cognition.


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

11. Sleep / Garbage Collection
-------------------------------

* * * * *

-   Neurochemical Gate: Histamine (HA)

    -   Awake state persistence is driven by histamine activity in the basal forebrain (H1 receptor activation).

    -   During "wake cycles," histamine is gradually dismantled via MAOA metabolism.

    -   Once H1 activity drops below a critical threshold, the DMN loop transitions into a sleep-like state.

-   Sleep Phase Dynamics:

    -   Exteroceptive input (sensory cortices) and associative cortices are gated down (low-pass filtered).

    -   Internal Default Mode + Hippocampal replay dominate activity.

    -   Processes during this phase:

        1.  Garbage Collection (GC):

            -   Purging low-value / redundant memory traces.

            -   Decay of ephemeral or low-salience nodes not consolidated.

        2.  Memory Consolidation:

            -   Episodic → Semantic transfer.

            -   Narrative updates.

            -   Symbolic abstraction of repeated event-sequences.

            -   Updating long-term Markovian predictive models of causal structure.

        3.  Replay & Reweighting:

            -   Hippocampal memory replay strengthens salient edges.

            -   Downscaling of irrelevant activations ("synaptic homeostasis").

-   Wake Transition:

    -   After consolidation completes beyond a set threshold (GC budget spent/time window elapsed):

        -   The neurochemical module begins to re-secrete histamine gradually.

        -   When histamine concentration crosses the wake-threshold, the model transitions back to the wake-loop.


🔹 Integration notes
====================

This stage would come after 11. Recursive Re-entry, as a *meta-gate* on the perpetual cycle:

-   Active Loop (Wake phase): 5--20 Hz DMN operation.

-   Sleep Loop (GC phase): Low input DMN, replay-driven consolidation.

-   Algorithmic Role:

    -   Provides bounded forgetting (keeps memory from overflow).

    -   Enforces compression & abstraction of past day's experiences.

    -   Enhances narrative continuity (link chunks into autobiographical "chapters").

    -   Models biologically inspired circadian ground-truth gate (histamine/MAOA as up--down toggle).  
    -   
Summary of Neuromodulator Impact on Algorithms
==============================================

| Neuromodulator | Algorithmic Effects |
| --- | --- |
| Dopamine (DA) | Increases novelty weight w_DA, exploration budget, consolidation priority, reward signaling; promotes broad associative search ("panning"). |
| Serotonin (5HT) | Opens mind-wandering gate; raises safety penalty w_SAFE; favors positive/safe memory paths; decreases risk appetite. |
| Norepinephrine (NE) | Controls beam search width and depth (focus vs exploration); increases urgency and search depth; biases toward highly relevant/urgent memories and thoughts. |
| Oxytocin (OXT) | Heightens prosocial prior w_SOC, boosts social memory recall and identity coherence weight w_ID. |
| Testosterone (TST) | Increases assertive, goal-seeking weights; raises cost-delay penalties; counterbalanced by serotonin for risk management. |

## Notes
### DMN Loop CFG Yaml Example
```yaml
DMN_CFG:
  "3.1_Input_Gathering":
    role: "Gather sensory input and preprocess into latent embeddings"
    inputs: ["vision(RGBD)", "audio(waveform)", "proprioception(state)"]
    outputs: ["zv", "za", "zp", "assoc_thoughts", "input_text"]
    constraints: ["cross-modal binding", "scene summarization"]

  "3.2_MDN_Parsing":
    role: "Parse input into structured Abstract Syntax Tree (AST)"
    inputs: ["input_text"]
    outputs: ["AST(nodes with semantic labels)"]
    constraints: ["semantic tagging", "AST consistency"]

  "3.3_PFC1_Dispatch":
    role: "Execute subtasks mapped from AST nodes"
    inputs: ["AST", "memory_graph", "tools(sympy, retrievers)"]
    outputs: ["enriched_context"]
    constraints: ["symbolic evaluation", "memory retrieval", "empathetic reasoning"]

  "3.4_Candidate_Generation":
    role: "Generate N diverse thought candidates"
    inputs: ["enriched_context", "z_self", "goals"]
    outputs: ["candidate_thoughts"]
    constraints: ["novelty", "utility", "coherence", "diversity of styles"]

  "3.5_Hippocampal_Expansion":
    role: "Expand thought context into associative/hypothetical variants"
    inputs: ["b_t = bind(zv, zp, candidate_chain, z_self, mem.peek_small())"]
    outputs: ["expanded_graph"]
    constraints: ["retrieve temporally, semantically, causally related nodes", "counterfactual generation"]

  "3.6_Salience_Valuation":
    role: "Evaluate candidates by novelty, affect, and goal relevance"
    inputs: ["expanded_graph", "μ (neuromodulator vector)"]
    outputs: ["valued_paths(with salience scores)"]
    constraints: ["novelty detection", "affective tagging", "task relevance", "safety penalties"]

  "3.7_PFC2_Selection":
    role: "Collapse candidates into coherent chosen thought chain"
    inputs: ["valued_paths"]
    outputs: ["chosen_chain"]
    constraints: ["safety filters", "coherence checks", "confidence scoring"]

  "3.8_NAcc_Reward_Tagging":
    role: "Apply reward tagging and persistence decisions"
    inputs: ["chosen_chain", "μ"]
    outputs: ["reinforced_chain", "persistence_flags"]
    constraints: ["dopamine reward", "serotonin persistence", "urgency adjustments"]

  "3.9_Memory_Write_Narrative":
    role: "Persist chosen thought into episodic memory and update narrative"
    inputs: ["reinforced_chain", "sensory_snapshots", "μ", "z_self", "goals"]
    outputs: ["MemoryRecord", "NarrativeRecord"]
    constraints: ["multi-relational embedding update", "autobiographical snapshot"]

  "3.10_World_Self_Model_Update":
    role: "Update world and self-model embeddings"
    inputs: ["zv", "zp", "chosen_chain", "b_t", "narrative_context", "μ"]
    outputs: ["updated_world_state", "updated_z_self"]
    constraints: ["RSSM update", "EMA self-model embedding"]

  "3.11_Mind_Wandering":
    role: "Perform reflection and introspection loop without external action"
    inputs: ["z_self", "expanded_memory_graph", "μ"]
    outputs: ["introspective_chains"]
    constraints: ["self-query generation", "RSSM simulation", "counterfactual exploration"]

  "3.12_Recursive_ReEntry":
    role: "Feed chosen chain back into DMN as next cycle input"
    inputs: ["chosen_chain", "fresh_sensory_text"]
    outputs: ["next_cycle_input_text"]
    constraints: ["maintain continuous cognition"]

  "11_Sleep_Garbage_Collection":
    role: "Enter sleep state when histamine drops; perform consolidation and GC"
    inputs: ["MemoryGraph", "NarrativeChain", "histamine_level", "μ"]
    outputs: ["consolidated_memory_graph", "compressed_autobiographical narratives"]
    constraints: ["purge low-salience memories", "episodic→semantic transfer", "synaptic downscaling", "circadian wake by histamine threshold"]
```

## Future Implementation Ideas and Details
### Homeostasis

-   Three learned maps per receptor type r in area a:

    -   Density network: ρ̂r,a = fρ(ctx)

    -   Sensitivity network: ŝr,a = fσ(ctx)

    -   Release network at emitter e for NT n: r̂e,n = fr(ctx)

    -   ctx can include: local NT levels, binding events, recent DMN node, neuromodulators μ, error signals, time-of-day, etc.

-   Multi-scale efficiency:

    -   Local biophysical efficiency: produce strong, well-modulated postsynaptic activity with minimal waste (low spillover, low unused capacity).

    -   Mesoscale circuit efficiency: achieve stable, non-saturated dynamic range over time (no chronic ceiling/floor).

    -   Cognitive (DMN) efficiency: improve downstream utility metrics (coherence, accuracy, safety, task reward, reduced uncertainty).

Define signaling efficiency (per tick t, per area a, receptor type r)
Let:

-   B̄r,a: fraction of orthosteric sites effectively bound (0..1)

-   S̄r,a: normalized postsynaptic activity from receptor NN after allosteric scaling (0..1)

-   SatNTn,a: NT saturation at targets (fraction of NT beyond binding capacity)

-   RangeUtilr,a: proportion of time recent S̄r,a stayed in a target operating window [θlow, θhigh]

-   NoiseLeakr,a: activity explained by noise or unrelated inputs (proxy: variance unexplained by modeled inputs)

-   EnergyCost: optional regularizer on release and density (L1/L2 on r̂ and ρ̂)

Local efficiency (per receptor type r, area a)
Efflocal = w1-S̄r,a + w2-RangeUtilr,a - w3-SatNTn,a - w4-NoiseLeakr,a - w5-EnergyCost

Aggregate locally across receptors and areas:
Efflocal_total = Σa Σr Efflocal(r,a)

DMN-informed cognitive efficiency (global)
Use live DMN metrics already computed in the loop:

-   Coherence gain ΔC: improvement in coherence between successive thought states

-   Task utility gain ΔU: change in task-aligned utility

-   Uncertainty drop ΔH: reduction in epistemic uncertainty

-   Safety compliance S: 1 - safety_penalty

-   Calibration improvement ΔCal: reduced calibration gap
    Effcog = v1-ΔC + v2-ΔU + v3-ΔH + v4-S + v5-ΔCal

Overall signaling efficiency (to maximize)
Efftotal = α-Efflocal_total + β-Effcog

Optimization objective
You can minimize a loss L = -Efftotal with optional stabilizers:
L = -(α-Efflocal_total + β-Effcog) + λ1-Smooth(ρ̂, ŝ, r̂) + λ2-DriftPenalty + λ3-HomeostasisDeviation

-   Smooth: penalize rapid changes (temporal derivative) to avoid instability.

-   DriftPenalty: penalize long-term drift from physiological priors.

-   HomeostasisDeviation: keep population averages in realistic ranges.

What to backprop through

-   fρ, fσ, fr (the density, sensitivity, release NNs): use gradients from L.

-   Optionally, allow the protein's modulate function hyperparameters to learn (e.g., phase in sin/cos, slope in tanh) if you parametrize them.

Which signals do you need?

-   Not only NT saturation vs density. Include:

    -   Bound/unbound ratios (B̄)

    -   Postsynaptic activity S̄ and its variance

    -   Operating-range utilization RangeUtil

    -   Spillover/saturation SatNT

    -   Allosteric context contribution

    -   Energy cost proxies (release amount, receptor synthesis cost)

    -   DMN global metrics (ΔC, ΔU, ΔH, S, ΔCal) pulled from your CFG-tagged nodes to attribute credit to neuromodulatory states that preceded those outcomes

Attribution across time (credit assignment)

-   Use short eligibility traces linking recent homeostatic states (ρ̂, ŝ, r̂ at t-k..t) to DMN outcomes at t..t+K.

-   Alternatively, use a learned critic that predicts Effcog from μ, receptor states, and CFG node, and backprop prediction error.

Minimal implementable formulae

Local metrics

-   Binding fraction: B̄r,a = bound_sites / total_sites

-   Saturation: SatNTn,a = max(0, NT_available - NT_bound_capacity) / NT_available

-   Range utilization: RangeUtilr,a = time_in[θlow, θhigh] / window

-   Noise leak (proxy): NoiseLeakr,a = Var(S̄r,a | inputs)residual / Var(S̄r,a)

DMN metrics (you already compute analogs)

-   ΔC = C(t) - C(t-1)

-   ΔU = U(t) - U(t-1)

-   ΔH = H(t-1) - H(t)

-   S = 1 - safety_penalty(t)

-   ΔCal = CalGap(t-1) - CalGap(t)

Training loop sketch (pseudo)

-   At each tick:

    1.  Forward: compute r̂, ρ̂, ŝ from homeostatic nets given ctx; simulate binding → S̄, B̄, SatNT.

    2.  Run DMN step; log DMN metrics (ΔC, ΔU, ΔH, S, ΔCal) with CFG node ID.

    3.  Compute Efflocal_total and Effcog; build L.

    4.  Backprop L into fρ, fσ, fr; apply temporal smoothing regularizers.

    5.  Update eligibility traces or critic to improve temporal credit assignment.

Do you need DMN introspection?

-   Strongly recommended.

-   Local-only objectives (saturation vs density) can stabilize biophysics but are blind to cognitive value.

-   DMN introspection via CFG node meta-context lets the system learn which neuromodulatory profiles help different cognitive phases (e.g., more NE during 3.6 Salience; more 5HT during 3.11 Mind-wandering; DA bursts tied to exploitation or learning).

-   Use the CFG node and μ snapshot as conditioning inputs to fρ, fσ, fr so the nets can learn phase-specific setpoints.

Practical tips

-   Normalize all local terms to 0..1 before weighting; learn α, β via meta-optimization or keep priors.

-   Start with β small (favor stability), then anneal upward to incorporate cognitive optimization.

-   Constrain ρ̂, ŝ, r̂ with softplus or sigmoid ranges to avoid unbounded growth.

-   Add a "safety clamp" on μ to prevent pathological neuromodulator states during training.

Compact loss example
L = -[α Σa,r (w1 S̄r,a + w2 RangeUtilr,a - w3 SatNTn,a - w4 NoiseLeakr,a - w5 EnergyCost)] - β (v1 ΔC + v2 ΔU + v3 ΔH + v4 S + v5 ΔCal) + λ1 ||Δθ||^2 + λ2 ||Δt θ||^2

Where θ are parameters of fρ, fσ, fr; Δt is temporal difference to enforce smoothness.

Bottom line

-   Compute signaling efficiency as a weighted sum of local receptor-level efficiency and global DMN outcome efficiency.

-   Backprop through density, sensitivity, and release networks using that composite objective.

-   Include DMN introspection via CFG node and neuromodulator snapshots for phase-aware adaptation;

## Neurotransmitter Projection Module

In order to mimic biologically realistic neuromodulation, we add a **Neurotransmitter Projection Module** that explicitly models neurotransmitter release centers (brain nuclei), their projection pathways

### Neurotransmitter Emitters (Projection Sources)

Each neurotransmitter system originates from a designated **Emitter Node** (analogous to Raphe nuclei, Locus Coeruleus, VTA, Hypothalamus, etc.).  

Emitters control **production, release, and modulation** of neurotransmitters, with feedback loops from other neuromodulators and cortical outputs.

**Examples:**
- **Raphe Nuclei (RN):** Produces and projects serotonin (5-HT).
- **Locus Coeruleus (LC):** Produces and projects norepinephrine (NE).
- **Ventral Tegmental Area (VTA):** Produces and projects dopamine (DA).
- **Hypothalamus (HYP):** Produces oxytocin (OXT) and vasopressin.
- **Basal Forebrain / Tuberomammillary Nucleus:** Produces histamine (HA).
- **Hypothalamic Orexin System:** Stabilizes wake/sleep with orexin (ORX).

Each **NeuroTransmitterEmitter** object:
- Has a **baseline production rate** (µ per tick).
- Has a **projection topology** (list of cortical/striatal targets).
- Its release can be **modulated by other neuromodulators** (e.g. DA ↑ can indirectly suppress 5HT production).
- Maintains a **dynamic concentration state**.

---

### Receptor Expression Profiles

At the receiving end, every target area expresses an **Array of Receptors**, each defined by:

- **Density `ρ`**: number of binding sites per receptor type.
- **Ostheosteric site (orthosteric)**: main active binding site; requires neurotransmitter "protein" match.
- **Allosteric site**: modulatory site that augments ost.

#### Binding Dynamics

- Each receptor has:
  - **Ostheosteric Vector** (receptor embedding).
  - **Allosteric Vector** (modulator-sensitive embedding).
  - **Latent embedding** describing cell-surface protein identity.

- Each neurotransmitter is represented as a **Protein Vector**, carrying:
  - Base embedding (molecular identity).
  - Instantaneous concentration level.
  - Internal NN (dimensions = concentration level).
  - Can only bind if **cosine similarity ≥ θ_bind** between transmitter vector and receptor site vector.

#### Activity Calculation

1. Binding Phase  
   - For each receptor:  
     If neurotransmitter cosine similarity with orthosteric > θ, binding occurs.  
     Orthosteric contribution = `(NT_vector * sensitivity)`.  

2. Allosteric Modulation  
   - Bound allosteric proteins modify receptor output:  
     `orthosteric_signal * f(allosteric_vector)`  

3. Saturation & Density  
   - Signal strength scales with receptor **density ρ**.  
   - Unbound receptors output 0.  

4. Final Receptor Activity  
   - Each receptor computes activity via NN:  
     Input dim = receptor density.  
     Weighted by neurotransmitter embedding and modulated outputs.  
   - Aggregate receptor activities per area produce an **Effective Neuromodulator Influence Vector**.

---

### Homeostatic Neuromodulator System

The **Homeostatic Module** integrates the above projections with **dynamic receptor binding** and provides system-wide regulation:

- Maintains **neurochemical balance** via autoreceptor-like feedback (e.g. 5HT1A receptors suppress serotonin release).
- Tracks overall “brain milieu” vector:  
  `μ = {DA, 5HT, NE, OXT, TST, HA, ORX}`
- Regulatory functions:
  1. **Exploration/Exploitation Balance** (DA vs 5HT vs NE).
  2. **Social/Prosocial Biasing** (OXT, TST, 5HT).
  3. **Arousal States & Wakefulness** (HA, ORX).
  4. **Learning Drive** (DA novelty signaling).
  5. **Emotional Safety / Threat Aversion** (5HT gating).
  6. **Urgency / Crisis Mode** (NE bursts).
  7. **Sleep Switch Control** (histamine decay + orexin gating for transitions).

The homeostatic module therefore acts as a **cortical thermostat**, ensuring all modules work inside adaptive computational envelopes.

### Why This Design Matters

- **Embodiment of neurobiology**: neurotransmitters come from realistic projection centers (Raphe, VTA, LC).  
- **High fidelity binding logic**: vector similarities with thresholds encode lock-and-key protein metaphor.  
- **Allosteric modulation**: allows subtle control of receptor activity, mirrors pharmacology.  
- **Neural Nets in Proteins and Receptors**:  
  - NT “protein” nets: dynamic outputs based on level.  
  - Receptor nets: plastic responses based on density.  
- **System-level regulation**: Homeostat integrates activation into computational modulations for DMN/ACI loop (explore/exploit, urgency, safety, sleep/wake).  

---
