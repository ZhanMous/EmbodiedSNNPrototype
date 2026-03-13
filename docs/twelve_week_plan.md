# Twelve-Week Plan

This is the operational version of the one-year plan.

The point of these 12 weeks is not to finish a field. The point is to build momentum, standards, and a working internal map.

Each week has five parts:

1. primary theme
2. concepts to understand
3. concrete work
4. written output
5. taste checkpoint

## Week 1: Functions, variables, and modeling

### Primary theme

Move from school math to modeling language.

### Concepts to understand

- variables and units
- functions as relationships
- graphs as behavior
- exponential growth and decay

### Concrete work

- implement three tiny scripts: linear growth, exponential decay, threshold switch
- explain what each variable means physically

### Written output

- one page: "A model is a compressed causal story"

### Taste checkpoint

- can you tell whether an equation describes mechanism or merely curve fitting?

## Week 2: Derivatives, accumulation, and state

### Primary theme

Understand change locally and globally.

### Concepts to understand

- derivative as local rate of change
- integral as accumulated quantity
- state update versus closed-form solution

### Concrete work

- implement Euler integration for one decaying state variable
- compare different step sizes and observe numerical error

### Written output

- one page: "Why time discretization already teaches engineering taste"

### Taste checkpoint

- do you notice when a model result is physics versus simulation artifact?

## Week 3: Vectors, matrices, and noisy observations

### Primary theme

Learn how structure and uncertainty coexist.

### Concepts to understand

- vectors and matrices
- matrix multiplication as connectivity
- mean, variance, conditional uncertainty

### Concrete work

- implement a tiny matrix-based recurrent update
- add noise to a sensory signal and compute running averages

### Written output

- one page: "Why real systems are structured and noisy"

### Taste checkpoint

- are you beginning to see why idealized clean models can mislead?

## Week 4: LIF from first principles

### Primary theme

Build the simplest spiking neuron that still teaches something real.

### Concepts to understand

- membrane leak
- threshold crossing
- reset and refractory period
- current injection

### Concrete work

- implement LIF from scratch in a notebook or script
- sweep membrane time constant and threshold
- plot membrane trace and spikes

### Written output

- one page: "What LIF preserves and what it throws away"

### Taste checkpoint

- can you defend LIF as a good model for some questions without overselling it?

## Week 5: Synapses, spike trains, and coding

### Primary theme

Move from a neuron to communication and representation.

### Concepts to understand

- synaptic filtering
- spike trains
- rate coding and temporal coding
- excitation and inhibition

### Concrete work

- add synaptic current traces to your LIF code
- compare fast and slow synaptic decay
- build a two-neuron excitatory or inhibitory motif

### Written output

- one page: "Not all spikes mean the same thing"

### Taste checkpoint

- do you see how timing changes the computation, not just the visualization?

## Week 6: Recurrent circuits and competition

### Primary theme

Understand how network structure creates behavior motifs.

### Concepts to understand

- recurrence
- winner-take-most competition
- stabilization by inhibition
- persistent activity versus runaway excitation

### Concrete work

- create a small recurrent SNN with left-right competition
- observe when it stabilizes and when it explodes

### Written output

- one page: "Why recurrence is power and danger at once"

### Taste checkpoint

- can you tell whether a dramatic result is genuine dynamics or parameter abuse?

## Week 7: Embodiment and closed loops

### Primary theme

See why intelligence changes when it must act.

### Concepts to understand

- sensing, actuation, feedback
- delay and instability
- body-environment coupling
- morphology as computational bias

### Concrete work

- run the embodied prototype
- inspect sensory history, action history, and trajectory
- alter one body parameter and observe behavioral changes

### Written output

- one page: "Bodies compute too"

### Taste checkpoint

- have you started distrusting intelligence claims that ignore real-time embodiment?

## Week 8: Connectomes and structure-constrained models

### Primary theme

Learn what anatomical structure buys you and what it does not.

### Concepts to understand

- graph structure in brains
- typed neurons and synapse sign
- structural priors versus functional sufficiency

### Concrete work

- redraw the prototype network as sensory, interneuron, descending, and motor groups
- justify every connection in words

### Written output

- one page: "A wiring diagram is not yet a mind"

### Taste checkpoint

- can you respect anatomical structure without mystifying it?

## Week 9: Plasticity and local learning

### Primary theme

Ask how change can happen without standard end-to-end backprop.

### Concepts to understand

- Hebbian ideas
- STDP
- reward modulation
- local information versus global objective

### Concrete work

- implement one toy plasticity rule in a tiny circuit
- log how synaptic weights change over time

### Written output

- one page: "What information does a learning rule need?"

### Taste checkpoint

- can you analyze learning rules by information flow instead of ideology?

## Week 10: Interfaces and hidden arbitrariness

### Primary theme

Study where systems are secretly hand-held by their designers.

### Concepts to understand

- sensory encoding interfaces
- descending control interfaces
- calibration and tuning
- approximation layers

### Concrete work

- inspect the embodied prototype's sensory mapping and action decoder
- write down which parts are principled and which are hand-tuned

### Written output

- one page: "Interfaces are where truth leaks"

### Taste checkpoint

- can you spot the difference between a research platform and an over-claimed proof?

## Week 11: Trust, privacy, and human stakes

### Primary theme

Move beyond pure technical fascination into responsibility.

### Concepts to understand

- trust boundaries
- privacy by architecture
- inclusion and accessibility
- institutional misuse risk

### Concrete work

- write a toy design for a privacy-first memory system for a personal AI companion
- identify who can abuse it and how

### Written output

- one page: "Trust is a technical property before it is a policy statement"

### Taste checkpoint

- do you reject architectures that require blind faith from users?

## Week 12: Synthesis and founder-level clarity

### Primary theme

Turn fragments into a worldview.

### Concepts to understand

- reusable stack thinking
- architecture memos
- strategic refusal
- invariants across fields

### Concrete work

- write a one-page architecture memo each for Alive, Realize, and Innocence
- write one page on the reusable shared stack

### Written output

- one page: "What I now believe is worth building"

### Taste checkpoint

- are you clearer, stricter, and less distractible than 12 weeks ago?

## Non-negotiable rules for all 12 weeks

1. Never let reading replace implementation.
2. Never let implementation replace writing.
3. Never let ambition excuse vagueness.
4. Record confusions explicitly instead of hiding them.
5. Refuse at least one seductive but shallow idea every two weeks.

## Minimum weekly scorecard

At the end of each week, score yourself from 1 to 5 on:

1. understanding
2. implementation quality
3. writing clarity
4. judgment improvement
5. discipline

Do not use the score to perform. Use it to notice drift early.