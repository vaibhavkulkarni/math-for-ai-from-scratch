# Chapter 2: Machines That Reshape Space (Matrices)

> 🧭 **The question.** In Chapter 1 *we* chose the numbers for every movie. A neural network can't wait for someone to hand-score everything. It has to *compute* useful new numbers from raw ones: turn 784 pixel values into "this is probably a 7", or turn a word's vector into something more useful for predicting the next word.
>
> What kind of machine turns one list of numbers into another? And what exactly is a "layer" in a neural network?

*Where this fits in the book:* every word of *"The cat sat on the …"* passes through dozens of these machines before the model makes its guess.

**The big idea:** a matrix is a machine that takes a vector in and gives a new vector out. A neural network layer is exactly this, plus one small twist.

## A matrix is a grid, and also a verb

$$
\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}
$$

Feed it the vector `[1, 1]`. Out comes `[2, 3]`: this matrix **stretches** the x-direction by 2 and the y-direction by 3. Other matrices rotate, flip or squash space. Same grid-of-numbers shape, different behaviour.

A machine needs to know what fits into it. A 2 × 2 matrix takes 2 numbers and gives back 2. How can you tell what *any* matrix takes and gives?

## Shapes and notation

A matrix with `m` rows and `n` columns is called an **m × n matrix** ("m by n"). The entry in row `i`, column `j` is written `Wᵢⱼ`. Here is a 2 × 3 matrix:

$$
W = \begin{bmatrix} 1 & 0 & 2 \\ 3 & 1 & 0 \end{bmatrix}
$$

Its **shape** is (2, 3). A list of 3 numbers is treated as a column of height 3, a 3 × 1 matrix. The rule for multiplying is:

> The inner numbers must match, and the outer numbers give the answer's shape. (2 × **3**) times (**3** × 1) gives (2 × 1).

```mermaid
flowchart LR
    A["W<br/>2 × 3"] -->|"inner 3 = 3 ✓"| C["result<br/>2 × 1"]
    B["x<br/>3 × 1"] --> C
```

So this matrix takes **3 numbers in and gives 2 numbers out**. Try it on `[1, 2, 3]`:

- Row 1: `1×1 + 0×2 + 2×3 = 7`
- Row 2: `3×1 + 1×2 + 0×3 = 5`

The output is `[7, 5]`. Notice that the matrix changed the *size* of the data, from 3 numbers to 2. That is how networks compress or expand information: a layer might turn 784 pixel values into 128 features.

A practical tip: **shape mistakes are the most common bug in AI code.** When something breaks, print `.shape` on everything.

We have been multiplying without saying exactly how. Here is the recipe, and what it means.

## How the multiplication works

To multiply a matrix by a vector, take the **dot product of each row with the vector**:

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 5 \\ 6 \end{bmatrix}
=
\begin{bmatrix} 1\times5 + 2\times6 \\ 3\times5 + 4\times6 \end{bmatrix}
=
\begin{bmatrix} 17 \\ 39 \end{bmatrix}
$$

> 🤔 **Pause and predict.** Work out `[[2, 0], [1, 1]]` times `[3, 4]`.
>
> <details markdown="1"><summary>Answer</summary>
>
> Row 1: `2×3 + 0×4 = 6`. Row 2: `1×3 + 1×4 = 7`. The answer is `[6, 7]`.
>
> </details>

That is all it is. Each output number answers one question: "how much does the input align with this row?" So **each row is a detector**. A row that is large where the input is large is detecting that pattern.

There is a second, equally useful way to read the same product: **the output is a mix of the matrix's columns**, using the input numbers as the amounts. Here `5 × [1, 3] + 6 × [2, 4] = [5 + 12, 15 + 24] = [17, 39]`. Rows-as-detectors and columns-as-ingredients are two views of one calculation. Use whichever makes the picture clearer.

Rows act as detectors. But in two dimensions you can do something even better: watch a matrix move the whole plane.

## Reshaping space: a few 2D examples

In two dimensions you can *see* what a matrix does. Apply each of these to the point `[1, 2]`:

| Matrix | What it does | Result on `[1, 2]` |
| --- | --- | --- |
| `[[2, 0], [0, 3]]` | stretches x by 2 and y by 3 | `[2, 6]` |
| `[[-1, 0], [0, 1]]` | mirrors left-right | `[-1, 2]` |
| `[[0, -1], [1, 0]]` | rotates 90° anticlockwise | `[-2, 1]` |

There is a neat way to read any matrix: **its first column is where the arrow `[1, 0]` lands, and its second column is where `[0, 1]` lands.** In the rotation matrix, the first column is `[0, 1]`, so the arrow pointing east is sent to point north, exactly what a quarter turn does. A matrix is fully described by where it sends the basic arrows.

![Three matrices applied to a small triangle: a stretch, a left-right mirror, and a 90 degree rotation, with the images of the basic arrows shown](figures/ch2_transforms.svg)

Watch what each matrix does to a small triangle. The dashed outline is the triangle before and the shaded one is where it lands. Whatever the matrix does to the two basic arrows, it does to every point in between.

A network applies one layer, then another, then another. So what happens when you run one matrix machine and then a second?

## Matrix times matrix

What if you apply one transformation after another? Say we double everything (matrix `A`) and then rotate (matrix `B`):

$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \qquad
B = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \qquad
BA = \begin{bmatrix} 0 & -2 \\ 2 & 0 \end{bmatrix}
$$

To get `B×A`, take the dot product of each row of `B` with each column of `A`. Check it on `[1, 2]`: doubling gives `[2, 4]`, rotating gives `[-4, 2]`. And `B×A` applied to `[1, 2]` gives `[0×1 + (-2)×2, 2×1 + 0×2] = [-4, 2]`. The same.

Note the order: `B×A` means "do `A` first, then `B`", read right to left like function notation `B(A(x))`. Order usually matters. For most pairs of matrices `B×A` and `A×B` are different (try putting on socks and shoes in the other order).

The lesson: **a chain of matrices collapses into a single matrix.** Ten matrix layers in a row would be no more powerful than one. So stacking plain matrices gets us nowhere new. Something has to break the straightness.

## Why we need the bend

A matrix can only do straight-line things: stretch, rotate, mirror, shear. It can never make a V-shape. But a little ReLU bend can. Take one input `x`. Use a matrix to make `[x, -x]`, apply ReLU to both entries, then add them up:

- `x = 2`: `[2, -2]` becomes `[2, 0]`, and the sum is `2`.
- `x = -3`: `[-3, 3]` becomes `[0, 3]`, and the sum is `3`.

That is the absolute value function, `|x|`, a V shape that no matrix alone can produce. Stack many such bends and you can approximate nearly any shape. This is the intuition behind the claim that neural networks can learn almost anything.

![ReLU of x, ReLU of minus x, and their sum, which is the V-shaped absolute value function](figures/ch2_relu_abs.svg)

<details markdown="1">
<summary>💡 <b>Going deeper:</b> can a network really learn <i>any</i> shape?</summary>

In 1989 George Cybenko proved (and Kurt Hornik and colleagues extended) a precise version of this intuition. A network with just one hidden layer of bends can approximate any continuous function on a bounded range as closely as you like, given enough hidden units. Note what it does *not* promise: it says nothing about how many units you need, or whether training will actually find the right weights. Those practical questions are why depth and good optimisers matter.

</details>

Each bend alone is a simple hinge. Added together they make a shape with a corner, which no stack of plain matrices can produce.

So a bend is the missing ingredient. Put a matrix and a bend together and you have a neural network layer.

## From matrix to neural network layer

A layer does three things:

1. **Multiply** the input by a weight matrix `W` (the detectors).
2. **Add** a bias vector `b` (a built-in nudge for each detector).
3. **Squash** the result with a simple non-linear function. The most popular, *ReLU*, just replaces negatives with 0.

That is `output = ReLU(W · input + b)`.

![One layer drawn as a network: 3 input circles connected by 12 lines to 4 detector circles, each applying ReLU to a row of W times x plus a bias](figures/ch2_layer.svg)

Here is the same layer as a picture. Each of the 12 lines carries one weight from `W`, each detector adds its own bias, and the ReLU bend is applied last.

Why the squash? Without it, stacking many layers collapses into a single matrix, however deep you go. The bend in ReLU is what lets networks learn curvy, complicated patterns.

**The numbers inside `W` and `b` are the "parameters". When people say a model has 70 billion parameters, they mean 70 billion of these numbers.** Learning (Chapters 3 to 5) is finding good values for them.

<details markdown="1">
<summary>📜 <b>History:</b> the first artificial neuron (1943)</summary>

The first mathematical neuron was Warren McCulloch and Walter Pitts' 1943 model: add up weighted inputs, and "fire" if the total passes a threshold. Frank Rosenblatt's **perceptron** (1958) added a rule for *learning* the weights, and the US Navy demonstrated it on a room-sized computer. ReLU, the simple bend used here, became the default much later, around 2010 to 2012, after researchers found that it made deep networks far easier to train than the older S-shaped squashes.

</details>

The numbers inside `W` and `b` are what training will tune. How many of them are there?

## Counting parameters

A layer that maps 3 numbers to 4 numbers has a weight matrix of shape 4 × 3 (that is 12 weights) plus a bias of 4 numbers. That is **16 parameters**.

> 🤔 **Pause and predict.** How many parameters does a layer from 4 inputs to 3 outputs have?
>
> <details markdown="1"><summary>Answer</summary>
>
> `4 × 3 = 12` weights plus 3 biases: 15 parameters.
>
> </details>

Scale it up. A network for 28×28-pixel images (784 numbers) with a hidden layer of 128 detectors and 10 outputs has:

- Layer 1: `784 × 128 + 128 = 100,480`
- Layer 2: `128 × 10 + 10 = 1,290`
- Total: **101,770 parameters**

A large language model has the same ingredients, just wider, deeper and with attention (Chapter 10), reaching tens or hundreds of billions of parameters. There is no new kind of math at that scale, only more of it.

Enough counting. Let's build one.

## Build it: one layer from scratch

```python
import numpy as np

rng = np.random.default_rng(0)

def relu(x):
    return np.maximum(0, x)

class Layer:
    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, 1, (n_out, n_in))  # n_out detectors
        self.b = np.zeros(n_out)

    def __call__(self, x):
        return relu(self.W @ x + self.b)

layer = Layer(3, 4)          # 3 numbers in, 4 detectors
movie = np.array([4, 8, 6])  # Scream from Chapter 1
print(layer(movie))
```

You just built the basic unit of every deep network. Stack a few and you have a deep network; you will make one learn in Chapter 9.

**Try this:** print `layer.W` next to the output. Pick one output number and verify by hand that it equals a row of `W` dotted with the movie, plus the bias, with negatives clipped to 0.

Our layer handles one input at a time. Real code processes thousands at once and shuffles shapes around constantly. Three practical tools make that possible.

## Transpose, batches and tensors

Three practical ideas that every piece of AI code relies on, and that the later chapters use without stopping to explain.

### Transpose: flip a matrix over

The **transpose** of a matrix swaps its rows and columns: row 1 becomes column 1, row 2 becomes column 2, and so on. It is written `Wᵀ` on paper and `W.T` in numpy. A 2 × 3 matrix becomes 3 × 2.

![A 2 by 3 matrix with coloured columns flipped into a 3 by 2 matrix whose rows carry the same colours; next to it, a 3D tensor drawn as a stack of three matrices labelled batch, tokens and features](figures/ch2_transpose_tensor.svg)

You will meet it in three places:

- **Dot products as matrix products.** If vectors are treated as columns, the dot product is "row times column", written `aᵀb`. That is why papers write `qᵀk` where this book writes `q · k`.
- **Going backwards through a layer.** A layer with `W` of shape 4 × 3 sends 3 numbers forward to 4 detectors. To send blame backwards from the 4 detectors to the 3 inputs, you use the same weights flipped: `Wᵀ`, shape 3 × 4. Each input collects blame from every detector it fed, weighted by the connection between them. That is the `.T` you will see in Chapter 9's backward pass.
- **Reordering products.** `(AB)ᵀ = BᵀAᵀ`. The order flips, like taking off your shoes before your socks.

### Batches: many inputs at once

Our layer took one input vector at a time. Graphics chips are fastest when given lots of work at once, so real code stacks many inputs as the **rows** of one matrix `X`, with shape (batch size, inputs). A single multiplication then processes the whole batch:

$$
\underbrace{X}_{(\text{batch} \times 3)} \; \underbrace{W^{\top}}_{(3 \times 4)} + b \;=\; \underbrace{\text{outputs}}_{(\text{batch} \times 4)}
$$

Each row of the result is one example's output. The bias `b` (4 numbers) is added to every row automatically, which numpy calls **broadcasting**. Many libraries store the weights already flipped, with shape (inputs, outputs), so the line becomes plain `X @ W + b`. That is the convention Chapter 9 uses. Neither is more correct: when in doubt, print the shapes.

### Tensors: beyond rows and columns

A single number has 0 dimensions, a vector 1, a matrix 2. Keep stacking and you get a **tensor**, the deep-learning word for an array with any number of dimensions.

| Data | Typical shape | Read it as |
| --- | --- | --- |
| a loss | `()` | one number |
| one token's embedding | `(768,)` | 768 features |
| a batch of embeddings | `(32, 768)` | 32 examples × 768 features |
| a batch of sentences | `(32, 128, 768)` | batch × tokens × features |
| a batch of colour images | `(32, 3, 224, 224)` | batch × colour channels × height × width |

Matrix multiplication extends naturally: the leading dimensions are treated as "do this for each one". Multiplying a `(32, 128, 768)` tensor by a `(768, 3072)` weight matrix applies the same layer to every token of every sentence and gives `(32, 128, 3072)`. This is why the shape-matching rule from earlier in this chapter is the most useful debugging tool in AI.

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 1, (4, 3))        # 4 detectors, 3 inputs
print(W.shape, W.T.shape)           # (4, 3) (3, 4)

X = rng.normal(0, 1, (5, 3))        # a batch of 5 inputs, one per row
b = np.zeros(4)
out = X @ W.T + b                   # (5, 3) @ (3, 4) -> (5, 4); b is added to every row
print(out.shape)

T = rng.normal(0, 1, (2, 6, 3))     # 2 sentences, 6 tokens each, 3 features per token
print((T @ W.T).shape)              # (2, 6, 4): the same layer applied to every token
```

## Back to the question

We asked what kind of machine turns one list of numbers into another. The answer is a **matrix**: each row is a detector that scores the input with a dot product, and together the rows produce a new vector, possibly of a different size. And a neural network **layer** is a matrix, plus a bias, plus a bend. The bend matters: without it, any number of layers would collapse into a single matrix.

In the code above, our layer turned *Scream* into 4 detector outputs. But its numbers are random, so those outputs mean nothing yet.

## Check yourself

Try each question on paper before opening the answer.

1. A 4 × 3 matrix multiplies a vector of 3 numbers. How many numbers come out?
    <details markdown="1"><summary>Answer</summary>4.</details>
2. Compute `[[1, 2], [3, 4]]` times `[1, -1]`.
    <details markdown="1"><summary>Answer</summary>`[1 - 2, 3 - 4] = [-1, -1]`.</details>
3. What is ReLU applied to `[-2, 0.5, 3]`?
    <details markdown="1"><summary>Answer</summary>`[0, 0.5, 3]`.</details>
4. How many parameters (weights plus biases) in a layer from 10 inputs to 5 outputs?
    <details markdown="1"><summary>Answer</summary>`10 × 5 + 5 = 55`.</details>
5. `X` has shape (32, 10) and `W` has shape (10, 4). What is the shape of `X @ W`? And of `X.T`?
    <details markdown="1"><summary>Answer</summary>(32, 4), and (10, 32).</details>
6. A batch of sentences has shape (8, 50, 512) and goes through a layer with weights of shape (512, 256). What comes out?
    <details markdown="1"><summary>Answer</summary>(8, 50, 256): the same layer applied to each of the 50 tokens in each of the 8 sentences.</details>

## Chapter 2 in one breath

A matrix is a machine that turns vectors into vectors, with each row acting as a pattern detector. A neural layer is a matrix, a bias and a bend.

---

**Next:** [Chapter 3: How Steep Is It?](03-derivatives.md). Our layer is full of random numbers, so it produces nonsense. If you nudge one of those numbers, does the output get better or worse, and by how much? That question is called a derivative, and it is much friendlier than its reputation.

*Curious about determinants, eigenvectors, SVD or convolutions? Take the optional side trip to [Chapter 2½: More Ways to See a Matrix](02b-matrix-views.md) first.*
