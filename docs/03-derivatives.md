# Chapter 3: How Steep Is It? (Derivatives)

**The big idea:** a derivative answers one question: *if I nudge the input a tiny bit, how much does the output move?*

## Start with a straight line

Before curves, recall the slope of a straight road: **rise over run.** If a road climbs 3 metres over 100 metres of ground, its slope is `3 / 100 = 0.03`.

For the line `y = 2x + 1`, going from `x = 0` to `x = 1` moves `y` from 1 to 3. Rise 2, run 1, slope 2. And it is the same slope everywhere, which is what makes a line a line.

A curve is trickier because its slope keeps changing. The trick is to **zoom in until the curve looks straight.** Take `f(x) = x²` and measure the slope between `x = 1` and a point a little further along:

| Second point | Rise | Run | Slope |
| --- | --- | --- | --- |
| x = 2 | 4 - 1 = 3 | 1 | 3 |
| x = 1.1 | 1.21 - 1 = 0.21 | 0.1 | 2.1 |
| x = 1.01 | 1.0201 - 1 = 0.0201 | 0.01 | 2.01 |
| x = 1.001 | 1.002001 - 1 = 0.002001 | 0.001 | 2.001 |

As the gap shrinks, the slope settles on **2**. That limiting value is the derivative of `x²` at `x = 1`. (Mathematicians call this process taking a *limit*; you can think of it as "keep shrinking the gap and see where the answer is heading.")

![Secant lines from P = (1, 1) to points at x = 2, 1.5 and 1.2 have slopes 3, 2.5 and 2.2, closing in on the tangent line of slope 2](figures/ch3_secants.svg)

The line through two nearby points is called a secant. As the second point slides towards `P`, the secant tilts less and less, and its slope settles on the derivative.

### Limits and continuity, in one paragraph each

A **limit** asks what value a function is *heading towards* as the input approaches some point, even if you are not allowed to land exactly on it. In the table above we never set the gap to zero (that would be `0/0`, which is meaningless). We only watched where the slopes were heading as the gap shrank, and they headed to 2.

A function is **continuous** if you can draw its graph without lifting your pencil: no jumps or holes. Derivatives need even more, a graph with no sharp corners. ReLU has one corner at 0, where the slope jumps from 0 to 1. In practice code simply picks one of the two values there, and nothing breaks.

> 📜 **History.** Isaac Newton (in the 1660s, calling derivatives "fluxions") and Gottfried Wilhelm Leibniz (publishing in 1684) invented calculus independently, and spent years in a bitter dispute over who came first. We use Leibniz's notation `dy/dx` because it is so practical: it looks like a fraction and, as Chapter 4 shows, often behaves like one. The prime notation `f'(x)` is due to Joseph-Louis Lagrange (1797). Newton and Leibniz reasoned with "infinitely small" quantities that nobody could quite define. It took until the 19th century for Augustin-Louis Cauchy and Karl Weierstrass to make limits rigorous.

## The hill

You are standing on a hillside in thick fog. You cannot see the valley, but you can feel the ground under your feet: it slopes down to your left. That feeling is a derivative. It is the **slope** at the point where you stand.

![Tangent lines on x squared at x = -2, -1, 0, 1 and 2 with slopes -4, -2, 0, +2 and +4](figures/ch3_tangents.svg)

For a function `f(x)`, the derivative `f'(x)` is the slope at `x`:

`slope ≈ (f(x + h) - f(x)) / h`, for a very small `h`.

That is the entire definition. Nudge, measure the change, divide by the nudge. Written properly as a limit:

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

## A worked example

Take `f(x) = x²`, at `x = 3`. Nudge by `h = 0.001`:

`(3.001² - 3²) / 0.001 = (9.006001 - 9) / 0.001 ≈ 6.001`

The exact answer is 6. The rule you can memorise: the derivative of `x²` is `2x`, and `2 × 3 = 6`. A few more rules cover almost everything you will meet:

- `x²` becomes `2x`
- `3x` becomes `3` (a straight line has constant slope)
- a constant becomes `0` (flat ground)

## The rules you actually need

You will almost never compute derivatives from the definition. A short table of rules does nearly everything. The derivative is written `f'(x)` ("f prime") or `df/dx`.

| Function | Derivative | Example |
| --- | --- | --- |
| constant `c` | `0` | flat ground has no slope |
| `x` | `1` | the line y = x rises 1 per step |
| `xⁿ` | `n × xⁿ⁻¹` | `x³` becomes `3x²` |
| `c × f(x)` | `c × f'(x)` | `5x²` becomes `10x` |
| `f(x) + g(x)` | `f'(x) + g'(x)` | `x³ + 2x` becomes `3x² + 2` |
| `eˣ` | `eˣ` | the only function equal to its own slope |
| `ln(x)` | `1/x` | steep near 0, flatter further out |

Check one numerically. For `x³` at `x = 2` the rule says `3 × 2² = 12`. The nudge method gives `(2.001³ - 8) / 0.001 = (8.012006 - 8) / 0.001 ≈ 12.006`. Close enough.

ReLU, from Chapter 2, has an easy derivative too: its slope is 0 for negative inputs (the line is flat) and 1 for positive inputs.

## Flat spots: where learning stops

At the very bottom of a valley the ground is flat, so the slope is **zero**. That gives a way to find the lowest point: solve `f'(x) = 0`. (Pierre de Fermat used this trick in the 1630s, before calculus even had a name.) For `f(x) = (x - 5)²` the derivative is `2(x - 5)`, which is zero exactly when `x = 5`. And indeed `(5 - 5)² = 0` is the smallest possible value. Training a model is a search for such flat spots in a landscape far too big to solve by hand, which is why we creep towards them step by step in Chapter 5.

## Many inputs: partial derivatives and the gradient

Real models have millions of parameters, so the loss depends on many inputs at once. A **partial derivative** asks the same question as before, but nudges one input and freezes all the others. It is written with a curly d: `∂f/∂x`.

Take `f(x, y) = x² + 3y`.

- Freeze `y` and nudge `x`: the `3y` part does not move, so `∂f/∂x = 2x`.
- Freeze `x` and nudge `y`: the `x²` part does not move, so `∂f/∂y = 3`.

Stack the partial derivatives into a list and you get the **gradient**:

$$
\nabla f = \left[\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right] = [\,2x,\ 3\,]
$$

At the point `(2, 5)` the gradient is `[4, 3]`. Verify with nudges: `f(2, 5) = 4 + 15 = 19`. Nudging `x` by 0.1 gives `f(2.1, 5) = 19.41`, which is up by about 0.4 (that is `4 × 0.1`). Nudging `y` by 0.1 gives `f(2, 5.1) = 19.3`, which is up by 0.3 (that is `3 × 0.1`).

The key insight: **the gradient is an arrow pointing in the direction of steepest uphill.** To go downhill, step the opposite way. That single sentence is gradient descent, which we meet in Chapter 5.

Picture a bowl-shaped function of two inputs, seen from directly above like a contour map. Each ring joins points of equal height. At every point the gradient arrow crosses the rings at a right angle, pointing straight uphill, and its length says how steep the ground is there.

![Contour rings of the bowl x squared plus 2 y squared, with gradient arrows pointing uphill and negative gradient arrows pointing downhill, all crossing the rings at right angles](figures/ch3_gradient_field.svg)

## Why AI cares

A model's **loss** is a number saying how wrong it is. The loss depends on the parameters. If the derivative of the loss with respect to a parameter is positive, increasing that parameter makes things worse, so you should decrease it. If it is negative, do the opposite. **The derivative tells each parameter which way to move.**

With many parameters, you take one such slope per parameter. That bundle of slopes is called the **gradient**.

## Build it: measure slopes numerically

```python
def f(x):
    return x ** 2

def slope(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

for x in [-2, 0, 1, 3]:
    print(x, slope(f, x), 2 * x)   # numerical vs exact
```

The last two columns should agree. Now swap in `f(x) = x**3` (exact derivative `3x²`) and check again.

Look closely and the agreement is not perfect: at `x = 0` the code prints `1e-05` instead of `0`. The one-sided nudge `(f(x+h) - f(x)) / h` measures the slope of a secant that sits slightly to the right, so it is off by a little. A better recipe nudges both ways and divides by the total gap:

$$
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
$$

This **central difference** is centred on `x`, so the errors on the two sides cancel. For `x²` it gives the exact answer. Libraries use it for *gradient checking*: they compare the gradients from backpropagation (Chapter 4) against this slow-but-simple estimate to catch bugs.

> 💡 **Two ways to get an answer.** The rule "derivative of `x²` is `2x`" is an **analytical** answer: exact, found with algebra. Nudging with a small `h` is a **numerical** answer: approximate, found by computing. Real-world engineering, and AI especially, leans heavily on numerical methods, because the functions are far too big to solve on paper. A rule of thumb: derive by hand to understand, and compute to get things done.

**Try this:** find where `f(x) = (x - 2)²` has slope 0. Is that the top or the bottom of the curve? (That flat spot is exactly what learning is hunting for.)

## The other half of calculus: integrals

Derivatives cut a curve into tiny pieces to find its slope. **Integrals** do the opposite: they add up tiny pieces to find a total, most often the **area under a curve**. The recipe is to fill the area with thin rectangles, add up their areas, and use thinner rectangles until the total stops changing. It is another limit.

![The area under sin(x) + 1.5 from 0 to pi approximated by 4, 8 and 32 rectangles, approaching the exact area 6.712](figures/ch3_integral.svg)

| Rectangles | 2 | 4 | 8 | 16 | 32 | exact |
| --- | --- | --- | --- | --- | --- | --- |
| Area estimate | 6.934 | 6.765 | 6.725 | 6.716 | 6.713 | 6.712 |

The symbol for it is a stretched S, for "sum":

$$
\int_0^{\pi} \big(\sin x + 1.5\big)\,dx \approx 6.712
$$

Read it as: "add up height × tiny width, for every sliver from 0 to π." In numpy, `np.trapezoid(y, x)` does the adding for you.

The **fundamental theorem of calculus**, discovered by Newton, Leibniz and Newton's teacher Isaac Barrow, says that integrating and differentiating undo each other, just as `exp` and `ln` do. In AI, integrals mostly appear in probability: the chance of a range of outcomes is an area under a curve (Chapter 7), and an average over infinitely many possibilities is an integral.

## Check yourself

1. What is the derivative of `5x²`? *Answer: `10x`.*
2. What is the slope of `x²` at `x = -1.5`? *Answer: `2 × (-1.5) = -3`. Negative means the curve is going downhill as x increases.*
3. For `f(x, y) = 3x + y²`, what is the gradient at `(1, 2)`? *Answer: `[3, 2y] = [3, 4]`.*
4. If the derivative of the loss with respect to a parameter `w` is `+6`, should `w` go up or down? *Answer: down, because increasing `w` would increase the loss.*
5. What is the derivative of `x³ + 2x`? *Answer: `3x² + 2`.*
6. Estimate the area under `f(x) = x` from 0 to 2 using 2 rectangles with heights taken at the midpoints. *Answer: midpoints 0.5 and 1.5, each width 1, so `0.5 + 1.5 = 2`. It is exact here, because the area is a triangle with base 2 and height 2.*

## Chapter 3 in one breath

The derivative is the slope: nudge the input, measure the change in output. It tells you which way to move a parameter to reduce the loss.

---

**Next:** [Chapter 4: Nudges Through a Chain](04-chain-rule.md): how a nudge travels through many steps, which is the whole secret of backpropagation.
