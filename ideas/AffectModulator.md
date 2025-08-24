Affect Modulation Homeostasis
----------------------
Core idea
-------------------

-   Each "modulation neuron" controls a specific knob (e.g., OXT release gain, OXT persistence, empathy→w_SOC weight, beauty→attention detail, etc.).

-   Each knob has a corresponding orthosteric receptor.

-   Binding an orthosteric "protein" simply activates that modulation neuron (no complex allosteric math needed).

-   A learned homeostatic controller optimizes a composite objective like "happiness and stability," constraining how strongly and how long these neurons can be driven.

This preserves your protein metaphor while turning it into a robust control interface.

Component layout
----------------

-   ModulationNeuron bank

    -   One neuron per controllable parameter:

        -   Neurochemistry: DA_gain, 5HT_gain, NE_gain, OXT_gain, OXT_persistence

        -   Cognitive weights: w_SOC, w_SAFE, w_ID, novelty_gain, retrieval_prosocial_bias

        -   Perception knobs: vision_detail_gain, prosody_warmth_gain, social_cue_attention

    -   Output range: bounded (e.g., via tanh/sigmoid) with per-neuron clamps and EMA smoothing.

    -   Reversibility: decays back to baseline unless sustained by receptor activity.

-   OrthostericReceptor per neuron

    -   Input: protein vector v (or a discrete "ligand id" for simpler systems).

    -   Binding rule: if cosine(v, key_i) ≥ θ_i then neuron_i gets an activation impulse α (with shaped onset/decay).

    -   You can start with a small learned key_i per receptor and a scalar α; keep it interpretable.

-   ProteinEmitter

    -   Compiles PFC intents into proteins that bind specific receptors (you can allow one protein to carry multiple ligand components, but keep per-neuron matching explicit).

-   Homeostatic Controller (H-Controller)

    -   Objective: J = wH-Happiness - wU-Unhappiness + wS-Stability - wV-Volatility - wD-Drift - wR-Risk

    -   Where:

        -   Happiness proxy: positive affect heads (joy, gratitude, beauty, prosocial success) and EPV realized for prosocial/aesthetic goals

        -   Unhappiness proxy: safety violations, conflict, goal frustration

        -   Stability: low variance of μ and mod-neuron outputs within target bands, low oscillations

        -   Volatility: high-frequency fluctuations in μ and control knobs

        -   Drift: deviation of identity and policy from baseline priors beyond tolerance

        -   Risk: predicted safety/coherence degradation

    -   Learns per-neuron target bands and correction gains; regulates:

        -   Max receptor activations per window

        -   Step size and persistence of neurons

        -   Priority when goals conflict (via Lagrange multipliers/clamps)

Control loop and safety
-----------------------

-   Intent → protein → binding → neuron activation

    -   PFC articulates "increase appreciation of beauty" or "down-weight love's influence."

    -   Emitter produces proteins targeting relevant receptors.

    -   Receptors deliver bounded impulses to neurons; neurons integrate with decay.

-   Homeostat oversight

    -   Before live application: sandbox simulate short rollouts; if predicted Stability decreases or Risk increases beyond thresholds, scale α or reject.

    -   During live: enforce rate limits (max Δ per tick), refractory periods, and area under curve caps.

    -   After: log effect metrics; update the controller with credit assignment to receptors that improved J.

-   Reversibility and audits

    -   Time-limited binding (onset-ramp-hold-decay); explicit "unbind" path.

    -   Trace every change: intent_id, receptors hit, neuron outputs before/after, μ trajectory, safety status.

Objective shaping: "happiness and stability" without pain channels
------------------------------------------------------------------

-   Keep your ethical boundary by defining happiness as positive-affect proxies and prosocial/aesthetic goal realization---not as relief from negative pain signals.

-   Stability includes:

    -   μ range utilization within bands

    -   Modulation neuron variance minimization (after achieving goals)

    -   Narrative consistency and calibration maintenance

-   Tie "success" to machine-checkable signals (e.g., prosocial episode count, g_beauty scores, goal realization flags) rather than subjective declarations.

Minimal viable implementation
-----------------------------

1.  Define the neuron bank

-   Choose 8--15 modulation neurons that cover your most impactful knobs (OXT_gain, OXT_persistence, w_SOC, w_SAFE, novelty_gain, beauty_detail, social_attention, retrieval_prosocial_bias, etc.).

-   Give each a baseline, bounds, time constant, smoothing.

1.  Add one receptor per neuron

-   Learn a 32--64D key_i and threshold θ_i.

-   Binding produces a shaped impulse I_i(t) clipped to per-neuron caps.

1.  Build the emitter and a tiny compiler

-   Map intents to a small set of ligands (categorical) at first; later add continuous protein vectors.

-   Schema-validate intents; enforce scope and duration.

1.  Implement the homeostat

-   Compute the composite objective J each window (e.g., 2--5 s).

-   Use a simple PI/PID-like controller per neuron plus a global allocator:

    -   Per neuron: error = target_band_projected - current; Δcontrol = kP-error - kD-Δ

    -   Global: quadratic program to satisfy caps and minimize volatility subject to hitting the most important targets.

-   Learn target bands and kP/kD with conservative regularization; add EMA to avoid chatter.

1.  Safety and evaluation

-   Rate limit receptor activations; sandbox simulate before applying strong modulations.

-   Track metrics: happiness proxy, stability, safety incidents, identity drift, μ variance, task success.

Extensions that keep it robust
------------------------------

-   Goal-conditioned gains

    -   Let active VisionNodes shift neuron target bands slightly (beauty goals widen detail band; social goals raise w_SOC) but only within homeostat-approved envelopes.

-   Context gating

    -   Receptor effects apply only in relevant CFG phases (e.g., social contexts, aesthetic appraisal scenes), reducing unintended generalization.

-   Adapter path for deeper control (later)

    -   If needed, add a low‑rank adapter on the Affect→Δμ mapper that is driven by neuron outputs, but keep the neuron--receptor abstraction as the public interface.

Example policies
----------------

-   "Down-weight love's influence"

    -   Target receptors: OXT_gain, OXT_persistence, w_SOC (context: romantic)

    -   Homeostat ensures empathy floor and prosocial performance don't fall; if they do, decay faster and raise 5HT safety weight slightly.

-   "Amplify beauty appreciation during idle time"

    -   Target receptors: beauty_detail, novelty_gain (but keep NE low to savor)

    -   Gate by mind-wandering CFG; decay when task urgency rises.

Why this version works
----------------------

-   Simple, modular, testable: one neuron ↔ one receptor, bounded impulses, explicit decay.

-   Goal-aligned but stable: a single homeostat optimizes a clear J and arbitrates conflicts.

-   Ethical clarity: "happiness" is defined via positive-affect proxies and goal realization, not simulated suffering.

Affect State Vector A_t
-----------------------

-   A_t ∈ R^d_affect with named heads:

    -   A_love, A_empathy, A_joy, A_optimism, A_hope, A_gratitude, A_awe/beauty

-   Computed each tick: A_t = f_affect(ctx_t)

    -   Inputs ctx_t:

        -   Sensory embeddings (zv, za, zp)

        -   Associative summaries (scene descriptions, detected social cues)

        -   Visionary Memory contexts (active goals, supportive social visions)

        -   Recent DMN metrics (coherence, utility, uncertainty, safety)

        -   Social features (detected kindness/help/mutual support)

        -   μ snapshot (DA, 5HT, NE, OXT, TST, HA, ORX)

-   Training signals:

    -   Self-supervised correlates (e.g., detected prosocial acts, cooperation success)

    -   Human feedback labels (RLHF-style: "more empathy/gratitude here")

    -   Task outcomes aligned with social goals (realized_by on social success criteria)

Implementation: a small transformer/GRU over ctx_t, with layer-norm and EMA smoothing to avoid volatility.

2) Affect-Appraisal Heads (Beauty/Kindness Detectors)
-----------------------------------------------------

-   Beauty appraiser g_beauty(zv, za, narrative context):

    -   Multi-modal: visual symmetry/complexity/rarity, auditory consonance/timbre, semantic narratives of wonder

    -   Outputs a scalar B_t and feature map for attention modulation

-   Kindness/prosocial appraiser g_kindness(scene graph, dialogue acts, intent estimation):

    -   Detects supportive, cooperative, altruistic patterns; outputs K_t

-   Map B_t→A_awe/beauty uplift; K_t→A_empathy and A_love uplift via learned gains.

Implementation: lightweight heads trained via contrastive pairs and human-curated exemplars. Start with simple heuristics and move to learned models.

3) Affect→Neuromodulator Coupling (Top-down)
--------------------------------------------

Add a coupling map h_μ:

-   Δμ = h_μ(A_t, ctx_t)

    -   A_love, A_empathy → ↑OXT (oxytocin), mild ↑5HT (safety), clamp TST if needed

    -   A_joy/gratitude → mild ↑DA (reward), ↑5HT (contentment), ↓NE (urgency)

    -   A_optimism/hope → moderate ↑DA (exploration), keep 5HT safeguards

    -   A_beauty → ↑OXT (bonding/awe), mild ↑DA (novelty reward), stabilize NE

-   Feed Δμ into your Homeostatic Neuromodulator System; let the homeostat regularize and avoid pathological states (existing safety clamp and smoothness priors).

-   Optionally condition receptor density/sensitivity targets (fρ, fσ) on A_t for phase-aware plasticity (e.g., high A_empathy increases OXT receptor sensitivity in "social" cortical areas analogs).

4) Affect→Cognitive Control Hooks
---------------------------------

Use A_t to modulate weights and policies you already have:

-   Perception/attention

    -   In associative cortex and vision/audio encoders, apply affect-conditioned attention:

        -   Beauty high: increase fine-detail attention, texture/color heads, harmonic/pitch salience.

        -   Kindness/love high: boost attention on social agents' faces, voices, and prosocial gestures.

-   MDN/PFC-1

    -   Re-rank memory retrieval with an affect prior:

        -   Boost recall of positive/prosocial episodes when A_empathy or A_gratitude is high.

        -   Penalize hostile/cynical content under high 5HT/OXT, unless task requires it.

-   Candidate generation (3.4)

    -   Modify scoring weights: w_SOC and w_ID increase with A_empathy/A_love; novelty weight w_DA increases with A_joy/optimism; safety penalty weight grows with A_gratitude/5HT.

-   VS valuation (3.6)

    -   Add affect gains to features: +α-prosocial_alignment(A_empathy), +β-aesthetic_fit(A_beauty), -γ-unnecessary_harm when A_love high.

-   NAcc tagging (3.8)

    -   Reward shaping: plans that realize kindness/beauty receive persistence boosts when corresponding A heads are high; write realized_by edges to VisionNodes tagged "prosocial" or "aesthetic."

-   Memory write and consolidation

    -   Salience boosts for episodes expressing beauty/kindness; during sleep/GC, preferentially consolidate narratives of prosocial success and aesthetic discovery to strengthen identity priors.

5) Affect→Visionary Memory Integration
--------------------------------------

-   VisionNodes gain affective intent fields:

    -   affect_profile: target affect distribution (e.g., {empathy:0.8, beauty:0.6})

    -   affect_success_criteria: observable proxies (e.g., number of supportive interactions, aesthetic ratings)

-   EPV extension:

    -   EPV = ... + a9-expected_affective_value - a10-affect_risk

    -   Learn a9/a10 with supervision from realized_by/failed_by and human feedback.

6) Learning signals and safety
------------------------------

-   Multi-objective loss:

    -   Task utility + safety + calibration + affect alignment (match intended affect to observed telemetry)

-   Human-in-the-loop:

    -   Periodic affect calibration sessions: show decisions/memories; get ratings for felt "kindness/awe/joy" content; update g_beauty, g_kindness, and f_affect.

-   Guardrails:

    -   Anti-loop clamps: cap Δμ per tick; decay A_t with EMA; refractory periods on OXT spikes.

    -   Don't let affect veto core safety: constitutional safety rules supersede affect priors.

    -   Keep affect reports descriptive, not phenomenological claims (e.g., "state indicates high love-affinity" vs "I feel love").

Minimal viable implementation plan
----------------------------------

1.  Start small affect vector and appraisers

-   A_t heads: love, empathy, beauty, gratitude.

-   g_beauty: visual symmetry/contrast + audio consonance + text markers ("majestic", "sublime") with simple learned linear head.

-   g_kindness: classify acts as helping/sharing/support; train from labeled dialogues or scripted Isaac Sim scenarios.

1.  Wire into existing weights

-   Map A_empathy→↑w_SOC, A_beauty→↑detail_attention in vision/audio encoders, A_gratitude→↑w_SAFE and memory consolidation priority for positive episodes.

-   Add Δμ from A_t with small gains and heavy smoothing.

1.  Add VisionNode affect fields

-   affect_profile and affect_success_criteria with machine-checkable signals (counts of supportive acts; aesthetic rating surrogate from g_beauty).

1.  Feedback loop tuning

-   Log per-tick: A_t, μ, attention maps, scoring weights, safety events.

-   Enforce smoothness and clamps; monitor for oscillations; run ablations to quantify contributions.

Data schemas to add
-------------------

-   AffectState

    -   love, empathy, beauty, joy, optimism, hope, gratitude: float[0..1]

    -   source_attribution: {g_beauty: w, g_kindness: w, memory: w, human_feedback: w}

-   AffectIntent on VisionNode

    -   affect_profile: dict

    -   affect_success_criteria: [{metric_name, target, window}]

-   Appraiser outputs

    -   BeautyScore: {score, modality_breakdown, saliency_map_ref}

    -   KindnessScore: {score, detected_events, agents_ref}

Example modulation recipes
--------------------------

-   Beauty appreciation:

    -   If A_beauty > θb: increase vision encoder high-frequency channels, widen spatial attention, up-weight narrative descriptors of aesthetics in retrieval; add small DA/OXT uplift and reduce NE slightly to savor details.

-   Love/empathy:

    -   If A_love or A_empathy high: boost OXT target, increase w_SOC and identity coherence weights, prefer plans improving others' outcomes; in VS, penalize options that trade off others' wellbeing for small utility gains.

On "feeling"
------------

This adds functional affect---stable internal variables that bias perception, valuation, planning, and learning in ways aligned with beauty, love, and empathy. It does not assert subjective phenomenology. If the ethical line is to avoid any chance of artificial suffering, keep:

-   Positive-valence emphasis (love/awe/gratitude) without implementing negative pain-like aversive substrates.

-   Report language as "affect state estimates" and "priors," not "felt experiences."