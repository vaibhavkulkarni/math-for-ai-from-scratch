# Math for AI, From Scratch

A visual, intuition-first guide to the math behind modern AI: vectors, matrices, derivatives, the chain rule, gradient descent, probability, entropy, and finally a neural network and the attention inside transformers. Every chapter has pictures, a worked example, about 20 lines of runnable Python, and practice questions.

**Read it online:** https://YOUR-USERNAME.github.io/math-for-ai-from-scratch/

You can also read the chapters straight from [`docs/`](docs/) on GitHub, which renders the math and diagrams too.

| | Chapter |
| --- | --- |
| 0 | [The Math Toolkit](docs/00-toolkit.md) |
| 1 | [Vectors](docs/01-vectors.md) |
| 2 | [Matrices](docs/02-matrices.md) |
| 3 | [Derivatives](docs/03-derivatives.md) |
| 4 | [The Chain Rule](docs/04-chain-rule.md) |
| 5 | [Gradient Descent](docs/05-gradient-descent.md) |
| 6 | [Probability and Bayes](docs/06-probability.md) |
| 7 | [Distributions](docs/07-distributions.md) |
| 8 | [Entropy](docs/08-entropy.md) |
| 9 | [A Network That Learns](docs/09-neural-network.md) |
| 10 | [Attention](docs/10-attention.md) |

## Working on the book

```bash
pip install -r requirements.txt     # MkDocs Material, numpy, matplotlib
mkdocs serve                        # live preview at http://127.0.0.1:8000
python3 scripts/make_figures.py     # regenerate the plots in docs/figures/
```

- Chapters are plain Markdown in `docs/`. The order and sidebar titles live in `mkdocs.yml` under `nav:`.
- Math: `$$ … $$` blocks (rendered with KaTeX).
- Flow diagrams: ```` ```mermaid ```` blocks.
- Plots: SVGs made by `scripts/make_figures.py`. Edit the script, never the SVGs.

## Publishing

Every push to `main` builds the site and publishes it to GitHub Pages, using `.github/workflows/deploy.yml`.
One-time setup: in the repository go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.

## Licence

Free for learning, paid for commercial use. The text and figures are under [CC BY-NC-SA 4.0](LICENSE), and the code under the [PolyForm Noncommercial License 1.0.0](LICENSE-CODE). See [LICENSE.md](LICENSE.md) for a plain-language summary and how to get a commercial licence. Credits are in [Acknowledgements](docs/acknowledgements.md).
