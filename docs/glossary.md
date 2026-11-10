# Glossary

| Term | Plain meaning | Chapter |
| --- | --- | --- |
| Vector | a list of numbers, read as an arrow or a point | 1 |
| Norm (length) | the size of a vector: square the entries, add, take the root | 1 |
| Dot product | multiply matching entries and add; measures alignment | 1 |
| Cosine similarity | the dot product with lengths removed; 1 means same direction | 1 |
| Embedding | a learned vector where closeness means similar meaning | 1 |
| Matrix | a grid of numbers that turns vectors into vectors | 2 |
| Parameter (weight, bias) | a number inside a model that training adjusts | 2 |
| Activation function | the bend after a layer (ReLU, tanh, sigmoid) | 2, 9 |
| Determinant | how much a matrix scales area or volume; 0 means information is lost | 2 |
| Rank | how many independent directions a matrix's output can point in | 2 |
| Eigenvector, eigenvalue | a direction a matrix only stretches, and the stretch factor | 2 |
| Transpose | a matrix flipped so rows become columns; written Wᵀ | 2 |
| Batch, tensor | many examples stacked into one array; an array with any number of dimensions | 2 |
| SVD, singular value | any matrix as rotate, stretch, rotate; the stretch factors | 2 |
| Convolution, kernel | sliding one small detector (the kernel) across an image or signal | 2 |
| Derivative | the slope: how much the output moves per tiny nudge of the input | 3 |
| Gradient | the list of partial derivatives; points uphill | 3 |
| Limit | the value a calculation heads towards as a gap shrinks to zero | 3 |
| Integral | adding up infinitely many thin slices; the area under a curve | 3 |
| Loss | a number measuring how wrong the model is | 3, 5 |
| Chain rule | multiply the rates along a chain of steps | 4 |
| Backpropagation | the chain rule applied from the loss back through the model | 4 |
| Vanishing / exploding gradients | gradients shrinking to nothing or blowing up across many layers | 4 |
| Learning rate | the size of each downhill step | 5 |
| Local minimum, saddle point | a valley that is not the lowest; a flat spot that is not a valley | 5 |
| Convex | bowl-shaped everywhere, so there is only one valley | 5 |
| Gradient descent | repeatedly stepping parameters against the gradient | 5 |
| Epoch, mini-batch | one pass over the data; a small random chunk of it | 5 |
| Conditional probability | the chance of A once you know B happened | 6 |
| Prior, posterior | belief before and after seeing evidence | 6 |
| Expected value | the probability-weighted average outcome | 6 |
| Markov chain | a random process whose next state depends only on the current one | 6 |
| Confusion matrix | the table of true and false positives and negatives | 6 |
| Precision, recall | how often a "yes" is right; how many real positives are found | 6 |
| Mean, median, mode | three ways to measure the centre of data | 7 |
| Variance, standard deviation | how spread out data is | 7 |
| Central limit theorem | sums of many small random effects tend towards a Gaussian | 7 |
| Standardising, z-score | rescaling to mean 0 and standard deviation 1; distance from the mean in standard deviations | 7 |
| Covariance, correlation | whether two quantities move together; the same scaled to between -1 and 1 | 7 |
| Gaussian | the bell-shaped distribution that arises from adding many small effects | 7 |
| Entropy | average surprise of a source | 8 |
| Cross-entropy | average surprise of a model about what really happens | 8 |
| KL divergence | the extra cost of using a wrong model: cross-entropy minus entropy | 8 |
| Logits, softmax | raw scores, and the function that turns them into probabilities | 8 |
| Perplexity | e to the average loss; the effective number of equally likely choices | 8 |
| Temperature | a divisor on the logits; low = confident, high = adventurous | 8 |
| Maximum likelihood | choosing parameters that make the observed data most probable | 8 |
| Overfitting | memorising training data instead of learning the pattern | 9 |
| Generalisation | working well on data the model has never seen | 9 |
| Regularisation | techniques that fight overfitting, such as weight decay, dropout and early stopping | 9 |
| Attention | each word takes a weighted mix of the other words' values | 10 |
| Residual connection | adding a block's input back to its output | 10 |
| Token | the unit a language model reads and predicts: a word or piece of a word | 10 |
| Autoregressive generation | writing text one token at a time, each predicted from all before it | 10 |
| Greedy, top-k, top-p | ways to choose the next token from the model's probabilities | 10 |
| Transformer | a stack of attention and ordinary layers; the basis of modern language models | 10 |

---

[Acknowledgements](acknowledgements.md) · [Licence](license.md)
