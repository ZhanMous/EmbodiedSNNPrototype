# Reading List

This list is ordered from foundation to embodiment. The aim is not to read everything at once. It is to build a ladder from neuron dynamics to closed-loop behavior.

## 1. Spiking and computational neuroscience foundations

### Gerstner, Kistler, Naud, Paninski. Neuronal Dynamics.

Why read it:

- best compact bridge from differential equations to spike-based computation
- gives you the right mental model for membrane time constants, refractory effects, and population coding

What to extract:

- LIF as a dynamical system
- firing-rate versus spike-time viewpoints
- synaptic filtering and recurrent dynamics

### Dayan and Abbott. Theoretical Neuroscience.

Why read it:

- still one of the cleanest books for connecting neuroscience intuition with mathematical modeling

What to extract:

- coding, inference, attractors, and sensory-motor transformations
- how simplified neuron models support system-level reasoning

### Maass, 1997. Networks of spiking neurons: the third generation of neural network models.

Why read it:

- useful for thinking about what spike timing adds beyond standard rate networks

What to extract:

- temporal coding
- computational arguments for spike-based models

## 2. Learning rules for SNNs

### Bi and Poo, 1998. Synaptic modifications in cultured hippocampal neurons.

Why read it:

- canonical STDP paper

What to extract:

- local learning without backpropagation
- time asymmetry in spike-dependent plasticity

### Florian, 2007. Reinforcement learning through modulation of STDP.

Why read it:

- a clean starting point for reward-modulated plasticity in SNNs

What to extract:

- three-factor learning rules
- how to bridge local plasticity and task reward

### Bellec et al., 2020. A solution to the learning dilemma for recurrent networks of spiking neurons.

Why read it:

- one of the most important modern papers if you want to compare biological plausibility against gradient-based training

What to extract:

- e-prop
- surrogate gradients versus online local approximations

## 3. Connectomics and structure-constrained models

### Dorkenwald et al., 2024. Neuronal wiring diagram of an adult brain.

Why read it:

- this is the adult fruit-fly whole-brain connectome result behind much of the recent excitement

What to extract:

- what data a connectome actually gives you
- where structural data is rich and where it is still incomplete functionally

### Schlegel et al., 2024. Whole-brain annotation and multi-connectome cell typing of Drosophila.

Why read it:

- turns the raw wiring diagram into something more usable computationally

What to extract:

- cell types, annotations, and why labeling matters for model building

### Shiu et al., 2024. A connectome-derived model of the Drosophila central brain.

Why read it:

- this is the key bridge from connectome data to a large LIF brain model

What to extract:

- how they construct a large recurrent LIF network from anatomy
- what “structure alone recovers sensorimotor motifs” really means
- which behaviors they can and cannot reproduce

## 4. Sensory pathways and embodied interfaces

### Lappalainen et al., 2024. Connectome-constrained networks predict neural activity across the fly visual system.

Why read it:

- shows how anatomical constraints plus task constraints can produce functionally useful visual models

What to extract:

- how to combine wiring priors with learned parameters
- how sensory pathway models can feed larger embodied systems

### Wang-Chen et al., 2024. NeuroMechFly v2.

Why read it:

- this is the most relevant body model if you want to move from abstract SNNs to embodied insect control

What to extract:

- neuromechanical interfaces
- sensory simulation
- why the body itself contributes computation

### Braun et al., 2024. Descending networks transform command signals into population motor control.

Why read it:

- essential for understanding why brain-to-body interfaces should not be treated as one-neuron one-behavior switches

What to extract:

- descending neurons as low-dimensional but distributed control handles
- population control versus symbolic command labels

## 5. Embodied control and simulation practice

### Todorov, Erez, Tassa, 2012. MuJoCo.

Why read it:

- practical foundation if you want physically grounded closed-loop behavior rather than toy kinematics

What to extract:

- contact dynamics, control loops, and simulation tradeoffs

### Sutton and Barto. Reinforcement Learning: An Introduction.

Why read it:

- not an SNN book, but necessary if you eventually want to learn descending interfaces, sensor mappings, or behavioral policies

What to extract:

- credit assignment in closed-loop systems
- exploration, delayed reward, and policy shaping

## 6. How to read this efficiently

Suggested order:

1. Neuronal Dynamics
2. Shiu et al.
3. NeuroMechFly v2
4. Lappalainen et al.
5. Braun et al.
6. Florian or Bellec depending on whether you want biologically local learning or trainable SNN engineering

## 7. What to focus on for your use case

Since your three projects already share LIF and SNN interests, the highest-value questions are probably these:

- when should LIF remain the core state model versus being replaced by richer neuron dynamics
- how much structure can be injected before learning starts to dominate
- what low-dimensional readout layer best matches a real descending-motor interface
- whether plasticity should live in sensory pathways, recurrent core, or readout layer