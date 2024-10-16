# Dynamics of Learning in Spiking Neural Networks
Enhancing Learning Processes in Spiking Neural Networks through Neuronal Architecture and 

## Introduction
This project investigates neuron structures within a single layer of a spiking neural network (SNN) and their impact on learning processes. It aims to explore how different structures influence learning outcomes.

### Inhibition Mechanisms
Inhibition mechanisms in neural networks and biological systems are crucial for regulating neural activity, shaping information processing, and ensuring balanced dynamics. Here are the primary types of inhibition mechanisms:

#### 1. **Lateral Inhibition**
- **Description**: This mechanism allows activated neurons to inhibit their neighboring, less active neurons. It enhances contrast and sharpens the representation of stimuli.
- **Function**: Promotes feature detection and enhances spatial resolution in sensory systems, such as vision.

#### 2. **Feedback Inhibition**
- **Description**: In this type, the output of a neuron or group of neurons feeds back to inhibit their own activity or the activity of upstream neurons.
- **Function**: Maintains stability in neural circuits and prevents over-excitation. It is crucial in maintaining homeostasis in neural networks.

#### 3. **Feedforward Inhibition**
- **Description**: This occurs when an excitatory neuron activates an inhibitory neuron that then inhibits another neuron. This type of inhibition acts before the target neuron has a chance to activate.
- **Function**: Provides precise control over the timing and amplitude of the excitatory input, improving signal processing and temporal dynamics.

#### 4. **K-Winners-Take-All (k-WTA)**
- **Description**: As previously discussed, this mechanism allows the top k most active neurons to fire while inhibiting all others.
- **Function**: Enhances competition among neurons, encouraging sparsity in neural activations, which improves efficiency and noise reduction.

#### 5. **Global Inhibition**
- **Description**: Involves the inhibition of multiple neurons across different populations or layers in a neural network, often through the release of inhibitory neurotransmitters.
- **Function**: Helps in synchronizing activity across different neural populations and regulating overall network excitability.

#### 6. **GABAergic Inhibition**
- **Description**: Inhibition mediated by gamma-aminobutyric acid (GABA), the primary inhibitory neurotransmitter in the brain. GABA receptors, when activated, allow chloride ions to enter the neuron, making it more negative and less likely to fire.
- **Function**: Essential for balancing excitation in the brain and preventing excessive neuronal firing, contributing to functions like relaxation, anxiety reduction, and sleep regulation.

#### 7. **Disinhibition**
- **Description**: This mechanism involves the inhibition of inhibitory neurons, which leads to increased activity in the target neuron or population of neurons.
- **Function**: Can enhance specific pathways or responses in the presence of competing inhibition, allowing for flexible responses in neural circuits.

#### 8. **Synaptic Inhibition**
- **Description**: Occurs when the activation of an inhibitory synapse results in the hyperpolarization of the postsynaptic neuron, preventing it from firing.
- **Function**: Reduces the likelihood of action potentials in target neurons, playing a critical role in regulating neural circuits.

#### 9. **Homeostatic Inhibition**
- **Description**: This mechanism helps maintain stable neuronal activity over time. When a neuron becomes too active, homeostatic inhibition mechanisms kick in to reduce its excitability.
- **Function**: Essential for the long-term stability of neural circuits, preventing runaway excitation or inhibition.
--- 

## Project Objectives
1. Understand the neuron structures within a single layer.
2. Assess the effects of these structures on learning processes.

## Activities

### Part 1: Basic Spiking Neural Network Implementation
1. **Network Structure**:
   - Implement a spiking neural network with one input layer and one output layer containing two neurons.
   - Ensure that all input neurons are connected to all output neurons.

2. **Learning Mechanism**:
   - Utilize Spike-Timing-Dependent Plasticity (STDP) for weight training in the experiments below.

3. **Experiments**:
   - **Lateral Inhibition**:
     - Introduce lateral inhibition to the second layer.
     - Investigate whether the two neurons in the second layer become sensitive to different inputs.
     - Repeat the previous experiment with varying degrees of input overlap and report the results.

   - **k-Winners-Take-All Mechanism**:
     - Add the k-Winners-Take-All mechanism to the second layer.
     - Similar to the previous experiment, assess the response using different input patterns and report the findings.

### Part 2: Enhanced Learning Mechanisms
1. **Combining Mechanisms**:
   - Create a network similar to the first part, incorporating both lateral inhibition and the k-Winners-Take-All mechanisms.
   - Add a homeostasis mechanism to the second layer to stabilize the learning process.

2. **Network Configuration**:
   - The second layer should consist of at least five neurons, with various input patterns.
   - Train the network and report the results.

3. **Parameter Analysis**:
   - Analyze the previous experiments using different parameters.
   - Report the influence of each mechanism on the learning process.

