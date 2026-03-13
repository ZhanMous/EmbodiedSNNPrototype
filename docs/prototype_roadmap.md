# Prototype Roadmap

## Goal

Build a staged embodied SNN system that starts as a controllable LIF closed loop and can later absorb connectome priors, plasticity, and richer body simulation.

## Design principle

Do not jump directly to a full fly. First build a system where every approximation is inspectable.

## Phase 0: Minimal closed loop

Current scaffold in this repository:

- 2D arena with a food source
- left and right sensory sampling
- dust accumulation as a grooming trigger
- structured recurrent LIF circuit
- low-dimensional motor readout for turn, forward, groom, and eat

Success criterion:

- the agent reliably exhibits approach, pause, grooming, and local feeding behavior

## Phase 1: Better SNN science

Add:

- synaptic traces and spike monitors
- raster plots and state logging
- parameter sweeps over time constants and recurrent sparsity

Questions to answer:

- which dynamics are robust versus hand-tuned
- whether recurrent inhibition stabilizes behavior better than feedforward gating alone

## Phase 2: Plasticity and adaptation

Add one of the following first, not all at once:

1. STDP on selected recurrent synapses
2. reward-modulated STDP on readout synapses
3. homeostatic firing-rate control

Success criterion:

- the agent improves food approach efficiency or grooming timing without destroying stability

## Phase 3: Connectome-inspired structure

Replace the hand-written adjacency with:

- sparse typed populations
- excitatory or inhibitory signs from known neurotransmitter classes
- sensory and descending subnetworks that reflect known circuit motifs

Success criterion:

- behavior still works after moving from arbitrary toy wiring to structured circuit priors

## Phase 4: Richer embodiment

Upgrade the body from the 2D arena to a physics-based morphology.

Options:

1. differential-drive agent in MuJoCo
2. legged point-mass body with contact dynamics
3. eventually NeuroMechFly or another insect body model

Success criterion:

- spike dynamics remain interpretable while body control becomes physically meaningful

## Phase 5: Research-grade experiments

Potential experiments:

1. compare fixed readout, RL-trained readout, and reward-modulated STDP readout
2. compare random sparse wiring against connectome-constrained wiring
3. test whether embodiment reduces the amount of learning needed for useful behavior
4. measure energy, sparsity, and latency versus task competence

## Recommended implementation sequence

1. get the current demo running and inspect trajectories
2. add spike logging and a simple raster visualization
3. run parameter sweeps on dt, tau, inhibition, and action thresholds
4. add one plasticity mechanism
5. only then swap in a richer body or richer wiring

## Failure modes to watch

- unstable recurrent excitation that looks impressive but is not task grounded
- behavior carried entirely by hand-coded action rules rather than by the circuit
- adding a complex body before the neural interface is understood
- claiming biological relevance before validating which approximations dominate results