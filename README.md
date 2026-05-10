# zettelkasten-content

Markdown source files for [afranca.me/notebook](https://www.afranca.me/notebook).

Each topic lives in its own folder. Files use standard Markdown + LaTeX math (`$...$` / `$$...$$`) and Obsidian callout syntax for theorem/definition/remark boxes:

```markdown
> [!definition] 1.1 — Normalizing Flow
> A **normalizing flow** is a generative model...

> [!theorem] 2.1 — Bellman Optimality
> For any policy π, the value function satisfies...

> [!remark] 3.1
> This is a sidebar observation...

> [!equation] 4.2
> $$\mathcal{L}(\theta) = \mathbb{E}[\log p_\theta(x)]$$
```

Supported callout types: `definition`, `theorem`, `lemma`, `proposition`, `corollary`, `remark`, `equation`.

Pushing to `main` automatically triggers a Vercel rebuild of the site (via the `VERCEL_DEPLOY_HOOK` secret).
