# Deep Learning — Phase 1, Class 02

# Biological Neurons vs Artificial Neurons

**Purpose of this chapter:** Build a precise understanding of the biological inspiration behind artificial neurons, then understand the artificial neuron's components, mathematical operation, and limitations of the analogy.

---

## 1. Learning outcomes

After studying these notes, you should be able to:

1. Define a biological neuron and identify its major structures.
2. Describe, at a high level, how a biological neuron receives and transmits information.
3. Define an artificial neuron and explain its computation.
4. Distinguish inputs, weights, bias, pre-activation, activation function, and output.
5. Calculate a neuron's output for a given set of inputs, weights, bias, and activation function.
6. Express a single neuron and a layer of neurons mathematically.
7. Explain which parts of the biological analogy are useful and where the analogy breaks down.

## 2. Prerequisite concepts

- **Scalar:** A single numerical value, such as 3 or -0.2.
- **Vector:** An ordered collection of numbers, such as \(\mathbf{x}=[2,3]^T\).
- **Function:** A rule that maps an input to an output. For example, ReLU maps a real number \(z\) to \(\max(0,z)\).
- **Parameter:** A value inside a model that is adjusted during training. Weights and biases are parameters in an ordinary artificial neuron.

These are brief contextual reminders; a full mathematics review is not required here.

## 3. Biological neuron: definition and role

A **biological neuron** is a specialized cell in the nervous system that receives information, integrates incoming influences, and communicates with other cells using electrical and chemical signalling.

Neurons communicate through networks. A neuron's activity is influenced by signals arriving from other neurons and, when conditions are met, it generates an electrical impulse called an **action potential**. Communication with a downstream cell commonly occurs at a **synapse**.

### 3.1 Main structures

| Structure                          | Definition and function                                                                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dendrites**                      | Branched extensions that receive signals from other cells. They carry incoming influences toward the cell body.                                                                       |
| **Cell body (soma)**               | The main part of the neuron containing the nucleus and cellular machinery. It helps integrate incoming signals.                                                                       |
| **Nucleus**                        | A structure inside the soma that contains most of the cell's genetic material and helps regulate cell activity.                                                                       |
| **Axon hillock / initial segment** | A region near where the axon begins. It is an important site for initiating an action potential when the neuron's membrane potential reaches threshold.                               |
| **Axon**                           | A long projection that conducts action potentials away from the cell body toward target cells.                                                                                        |
| **Myelin sheath**                  | A fatty insulating covering around some axons. It enables faster, more efficient conduction in myelinated axons.                                                                      |
| **Axon terminals**                 | Endings of the axon that communicate with target cells, commonly by releasing neurotransmitters.                                                                                      |
| **Synapse**                        | The junction or communication site between a neuron and another neuron or target cell. A chemical synapse uses neurotransmitters; electrical synapses use direct electrical coupling. |

### 3.2 High-level signal flow

```text
Signals from other cells
          |
          v
       Dendrites
          |
          v
    Cell body (soma)
          |
          v
Integration of incoming influences
          |
          v
Axon hillock / initial segment
          |
          v
Action potential, if threshold is reached
          |
          v
         Axon
          |
          v
    Axon terminals
          |
          v
 Synapse / target cell
```

This is a simplified flow. Real neurons have complex branching structures, time-dependent electrical dynamics, many kinds of ion channels, and diverse synaptic mechanisms.

### 3.3 Action potential and synaptic communication

- A neuron's membrane has an electrical potential across it.
- Incoming synaptic signals can make the membrane potential more or less likely to reach the firing threshold.
- If the relevant region reaches threshold, the neuron generates an **action potential**: a brief, regenerative electrical event that travels along the axon.
- At many chemical synapses, the arriving action potential causes neurotransmitter release. Neurotransmitters cross the synaptic cleft and bind to receptors on the target cell, changing its activity.

**Important distinction:** The synapse is not the axon. The axon carries the action potential; the synapse is the communication junction with the next cell.

## 4. Artificial neuron: definition

An **artificial neuron** is a simplified mathematical computational unit that combines numerical inputs using learnable weights and a bias, then transforms the result using an activation function to produce an output.

A standard artificial neuron performs two stages:

1. **Affine / pre-activation calculation:** weighted sum plus bias.
2. **Activation:** apply a function to the pre-activation value.

```text
Input values                 x1, x2, ..., xn
     |                             |
     v                             v
Multiply by corresponding weights w1, w2, ..., wn
     |
     v
Sum weighted inputs and add bias b
     |
     v
Pre-activation value z
     |
     v
Apply activation function f
     |
     v
Output a
```

## 5. Components of an artificial neuron

### 5.1 Inputs — \(x_i\)

An **input** is a numerical feature supplied to the neuron. A neuron may receive one input or many inputs.

For example, a student-performance model might receive features such as study hours, attendance rate, and a previous assessment score. The values must be represented numerically before they can be processed by the neuron.

### 5.2 Weights — \(w_i\)

A **weight** is a learnable numerical parameter associated with an input. It scales that input's contribution to the neuron's weighted sum.

- A positive weight means increasing that input increases the weighted sum, with other values held fixed.
- A negative weight means increasing that input decreases the weighted sum, with other values held fixed.
- A weight close to zero means that input has a small direct contribution to the weighted sum.

Weights are generally learned by an optimization procedure during training; they are not usually hand-selected and kept fixed.

### 5.3 Bias — \(b\)

A **bias** is a learnable scalar parameter added to the weighted sum before the activation function is applied.

The bias allows the neuron to shift its pre-activation value independently of the current input values. If all inputs are zero, the pre-activation value is still \(b\), not necessarily zero.

Bias is not the same thing as an input feature. It is a separate model parameter.

### 5.4 Weighted sum / pre-activation — \(z\)

The **pre-activation value** is the result of multiplying each input by its corresponding weight, adding all those products, and adding the bias.

For \(n\) inputs:

\[
z = \sum\_{i=1}^{n} w_i x_i + b
\]

For two inputs:

\[
z = w_1x_1+w_2x_2+b
\]

The term _pre-activation_ emphasizes that the activation function has not yet been applied.

### 5.5 Activation function — \(f\)

An **activation function** is a mathematical function applied to the pre-activation value to produce the neuron's output:

\[
a=f(z)
\]

A nonlinear activation function enables a multilayer network to represent nonlinear relationships. If a network consists only of compositions of ordinary affine/linear transformations, the overall mapping remains affine; depth alone does not make that mapping nonlinear.

Examples introduced for later study:

- **ReLU:** \(f(z)=\max(0,z)\)
- **Sigmoid:** \(f(z)=\frac{1}{1+e^{-z}}\)
- **Tanh:** \(f(z)=\tanh(z)\)

These functions and their gradients will be studied in the activation-functions class. The current class uses ReLU for a worked example.

### 5.6 Output — \(a\)

The **output** is the value produced after applying the activation function. It may become an input to a later layer or, depending on the network architecture, contribute to the model's final prediction.

## 6. Complete mathematical description

For a neuron with \(n\) inputs:

**Pre-activation:**

\[
z = \sum\_{i=1}^{n}w_ix_i+b
\]

**Activation / output:**

\[
a=f(z)
\]

Where:

| Symbol  | Meaning                   | Type in this single-neuron expression |
| ------- | ------------------------- | ------------------------------------- |
| \(n\)   | Number of input features  | Positive integer                      |
| \(x_i\) | The i-th input feature    | Scalar                                |
| \(w_i\) | Weight for the i-th input | Scalar parameter                      |
| \(b\)   | Bias                      | Scalar parameter                      |
| \(z\)   | Pre-activation value      | Scalar                                |
| \(f\)   | Activation function       | Function                              |
| \(a\)   | Neuron output             | Scalar                                |

### 6.1 Vector form

Represent the inputs and weights as column vectors:

\[
\mathbf{x}=\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix},\qquad
\mathbf{w}=\begin{bmatrix}w_1\\w_2\\\vdots\\w_n\end{bmatrix}
\]

Then:

\[
z=\mathbf{w}^{T}\mathbf{x}+b
\]

Here \(\mathbf{w}^{T}\) is a row vector with shape \((1,n)\), and \(\mathbf{x}\) is a column vector with shape \((n,1)\). Their product is a scalar with shape \((1,1)\), to which the scalar bias is added.

## 7. Worked numerical example

Given:

\[
x_1=2,\quad x_2=3
\]

\[
w_1=0.5,\quad w_2=0.4,\quad b=0.2
\]

Activation function: ReLU, where \(f(z)=\max(0,z)\).

**Step 1 — Multiply each input by its weight.**

\[
w_1x_1=0.5\times2=1.0
\]

\[
w_2x_2=0.4\times3=1.2
\]

**Step 2 — Add the weighted inputs and bias.**

\[
z=1.0+1.2+0.2=2.4
\]

Therefore, the pre-activation value is:

\[
\boxed{z=2.4}
\]

**Step 3 — Apply ReLU.**

\[
a=\max(0,2.4)=2.4
\]

Therefore, the final neuron output is:

\[
\boxed{a=2.4}
\]

**Interpretation:** The weighted-sum calculation produces 2.4. ReLU leaves this positive value unchanged. The pre-activation and output happen to be equal in this example; they are not always equal.

### 7.1 Negative pre-activation example

If \(z=-3\) and the activation is ReLU:

\[
a=\max(0,-3)=0
\]

The pre-activation is -3, but the output is 0. This illustrates why it is important not to confuse \(z\) with \(a\).

## 8. From one neuron to a layer of neurons

A layer contains multiple artificial neurons. If each of \(m\) neurons receives the same \(n\)-dimensional input vector, their computations can be written together using a matrix.

Using column-vector convention:

\[
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}
\]

\[
\mathbf{a}=f(\mathbf{z})
\]

For these expressions:

| Quantity                             | Shape     |
| ------------------------------------ | --------- |
| Input \(\mathbf{x}\)                 | \((n,1)\) |
| Weight matrix \(\mathbf{W}\)         | \((m,n)\) |
| Bias vector \(\mathbf{b}\)           | \((m,1)\) |
| Pre-activation vector \(\mathbf{z}\) | \((m,1)\) |
| Activation vector \(\mathbf{a}\)     | \((m,1)\) |

The activation function is applied elementwise in this common layer formulation. These shapes use a single example represented as a column vector; batched implementations often use a different convention, which will be covered when tensor shapes are taught.

## 9. Biological and artificial neurons: comparison

| Aspect           | Biological neuron                                  | Artificial neuron                                |
| ---------------- | -------------------------------------------------- | ------------------------------------------------ |
| Nature           | Living cell                                        | Mathematical/computational unit                  |
| Input            | Electrochemical signals from other cells           | Numerical feature values                         |
| Input influence  | Complex synaptic and membrane dynamics             | Numerical weights scale input values             |
| Integration      | Distributed electrical and biochemical processes   | Weighted sum plus bias                           |
| Output mechanism | Action potential and synaptic communication        | Activation-function output                       |
| Adaptation       | Biological plasticity and other cellular processes | Parameter updates through a training algorithm   |
| Time behaviour   | Rich, continuous and event-based dynamics          | Often a static computation for each forward pass |

### 9.1 Useful analogy — and its limits

| Biological concept      | Rough artificial-network analogy | Important limitation                                                          |
| ----------------------- | -------------------------------- | ----------------------------------------------------------------------------- |
| Dendritic inputs        | Input values                     | Artificial inputs are numbers, not electrochemical signals.                   |
| Synaptic influence      | Weights                          | A weight is not a literal synapse; biological synapses have complex dynamics. |
| Integration in the cell | Weighted-sum operation           | A soma does not simply calculate a dot product.                               |
| Firing behaviour        | Activation function              | Most activation functions are not realistic models of action potentials.      |
| Synaptic plasticity     | Learning / parameter updates     | Biological learning is much richer than standard gradient-based optimization. |

**Core point:** Artificial neural networks were inspired by biological nervous systems, but an artificial neuron is a deliberately simplified mathematical model, not a digital copy of a biological cell.

## 10. Why weights and bias matter

- Weights determine how strongly each feature contributes to the pre-activation value.
- Bias shifts the pre-activation value independently of the input vector.
- Together, weights and bias determine the affine transformation performed before activation.
- During training, an optimizer updates these parameters based on gradients of a loss function. The details of loss, gradients, backpropagation, and optimization are covered in later classes.

## 11. Common misconceptions and corrections

| Misconception                                                    | Correct understanding                                                                                         |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| “The synapse carries the signal along the whole neuron.”         | The axon conducts the action potential; the synapse is the communication junction with another cell.          |
| “The soma alone decides whether a neuron fires.”                 | Integration occurs across the neuron; the axon initial segment is a key site for action-potential initiation. |
| “A weight is an input.”                                          | An input is data; a weight is a learnable parameter that scales that data.                                    |
| “Bias must be zero when inputs are zero.”                        | The bias is independent of the input values; if all inputs are zero, \(z=b\).                                 |
| “The weighted sum is the final output.”                          | The weighted sum is the pre-activation value. The activation function is applied afterward.                   |
| “ReLU returns the absolute value.”                               | ReLU returns \(\max(0,z)\); negative inputs become zero.                                                      |
| “An artificial neuron behaves exactly like a biological neuron.” | It is a highly simplified mathematical abstraction inspired by selected biological ideas.                     |
| “More layers automatically create nonlinear behaviour.”          | Without nonlinear activations, stacked affine layers remain an affine transformation.                         |

## 12. Applications and engineering relevance

Artificial neurons are combined into layers and networks to perform tasks such as:

- **Classification:** map input features to class scores or probabilities.
- **Regression:** predict a continuous numerical value.
- **Computer vision:** learn representations from image pixels.
- **Natural language processing:** process numerical representations of text.
- **Time-series modelling:** learn patterns from sequential numerical data.

In practical systems, engineers do not usually train a single isolated neuron for a complex task. They assemble many neurons into architectures, choose suitable losses and optimizers, prepare data, evaluate generalization, and monitor training.

## 13. Key terminology glossary

| Term                | Definition                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| Biological neuron   | A nervous-system cell that receives, integrates, and transmits information.                        |
| Dendrite            | A branched structure that receives incoming signals.                                               |
| Soma                | The cell body of a neuron.                                                                         |
| Axon                | A projection that conducts action potentials away from the cell body.                              |
| Action potential    | A brief regenerative electrical signal that travels along a neuron.                                |
| Synapse             | A junction or site of communication between cells.                                                 |
| Artificial neuron   | A mathematical unit that combines inputs with weights and bias and applies an activation function. |
| Input feature       | A numerical value supplied to a model.                                                             |
| Weight              | A learnable parameter that scales an input's contribution.                                         |
| Bias                | A learnable parameter added to the weighted sum.                                                   |
| Pre-activation      | The weighted sum plus bias, before applying the activation function.                               |
| Activation function | A function applied to the pre-activation value.                                                    |
| Neuron output       | The value produced after activation.                                                               |
| Layer               | A group of computational units operating at the same stage of a network.                           |
| Parameter           | A model value learned from data during training.                                                   |

## 14. Interview revision points

Be prepared to explain:

1. What are the main parts of a biological neuron, and what does each do?
2. What is the difference between an axon and a synapse?
3. What is an artificial neuron?
4. What is the difference between an input, a weight, and a bias?
5. What is the difference between pre-activation and activation output?
6. Why are nonlinear activation functions important in multilayer networks?
7. How would you calculate the output of a neuron for given inputs, weights, bias, and activation?
8. In what ways is the biological-neuron analogy useful, and where does it fail?

A strong answer should define the term precisely, explain its role, and use the relevant equation or example where appropriate.

## 15. One-page quick revision

**Biological neuron:** A specialized cell that receives, integrates, and transmits information.

- Dendrites: receive incoming signals.
- Soma: cell body; integrates incoming influences.
- Axon initial segment: key action-potential initiation region.
- Axon: conducts action potentials away from the soma.
- Axon terminals: communicate with target cells.
- Synapse: communication junction.

**Artificial neuron:** A mathematical unit that applies an affine transformation followed by an activation function.

\[
\boxed{z=\sum\_{i=1}^{n}w_ix_i+b}
\]

\[
\boxed{a=f(z)}
\]

- \(x_i\): input feature.
- \(w_i\): learnable weight.
- \(b\): learnable bias.
- \(z\): pre-activation.
- \(f\): activation function.
- \(a\): output.

**ReLU example:**

\[
f(z)=\max(0,z)
\]

If \(z=2.4\), output = 2.4. If \(z=-3\), output = 0.

**Layer form:**

\[
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b},\qquad
\mathbf{a}=f(\mathbf{z})
\]

**Remember:** Data are inputs; weights and bias are parameters; weighted sum is not yet the activated output. Biological neurons are much more complex than artificial neurons.

## 16. Five-minute self-recall checklist

Without looking at the notes, try to:

- [ ] Define a biological neuron.
- [ ] Name and explain dendrites, soma, axon, axon terminals, and synapse.
- [ ] Explain the difference between an axon and a synapse.
- [ ] Define an artificial neuron.
- [ ] State the order of operations inside an artificial neuron.
- [ ] Write the pre-activation equation and define every symbol.
- [ ] Explain why weights and bias are separate parameters.
- [ ] Calculate the worked example from memory.
- [ ] Explain why the biological analogy is only approximate.
- [ ] Distinguish pre-activation \(z\) from output \(a\).

---

## Scope note

This class introduces the biological inspiration and the basic artificial-neuron computation. Detailed study of perceptrons, activation functions, forward propagation through multiple layers, loss functions, backpropagation, and parameter optimization belongs to later classes. Do not treat those topics as mastered based only on this introductory chapter.
