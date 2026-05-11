---

title: "Proximal Policy Optimization"
subtitle: "The clipped surrogate objective, multiple epochs of minibatch updates, and why PPO dominates practice"

---


TRPO solved the step-size problem in a mathematically complete way and at a practically high cost. The constrained optimization over the surrogate objective — with its conjugate gradient inner loop, its Fisher-vector products, its backtracking line search — was correct, well-founded, and difficult to use. Extending TRPO to recurrent policies, shared actor-critic architectures, or multi-task settings required non-trivial modifications to the constraint machinery. Incorporating it into a larger system meant carrying the conjugate gradient solver as a dependency. Most practitioners who wanted trust region behavior in their policy gradient methods ended up with a carefully tuned fixed step size and a prayer.

The question was whether the essential content of the trust region constraint — keep the policy from changing too much in a single update — could be enforced by a mechanism that lived entirely within a standard first-order optimizer. Not a constrained optimization problem solved by a specialized algorithm, but a modified objective function that a plain call to Adam or SGD could minimize. Schulman, Wolski, Dhariwal, Radford, and Klimov answered yes in 2017 with **Proximal Policy Optimization** (PPO), and the answer was simple enough to implement in a dozen lines.

## The Importance Ratio

The surrogate objective that TRPO maximizes can be written compactly in terms of the **importance ratio**:

$$r_t(\theta) = \frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)}.$$

Note that $r_t(\theta_{\text{old}}) = 1$: at the current policy, the ratio is exactly one. The unclipped surrogate objective is then:

$$L^{\text{CPI}}(\theta) = \mathbb{E}_t\!\bigl[r_t(\theta)\, \hat{A}_t\bigr],$$

where CPI stands for "conservative policy iteration," the theoretical lineage the surrogate inherits. This is simply the policy gradient with the on-policy log-probability gradient $\nabla_\theta \log \pi_\theta(A_t \mid S_t)$ replaced by the ratio $r_t(\theta)$, enabling multiple gradient steps on the same batch of data: after the first gradient step, the new $\theta$ differs from $\theta_{\text{old}}$, and $r_t(\theta) \neq 1$, so subsequent steps on the same data contribute a genuine off-policy correction rather than repeating the same gradient.

Without any constraint, maximizing $L^{\text{CPI}}$ over multiple steps is dangerous. If the advantage $\hat{A}_t$ is positive for some action $a$ in state $s$, $L^{\text{CPI}}$ increases without limit as $\pi_\theta(a \mid s)$ increases — even if $\pi_\theta(a \mid s)$ has already grown far beyond $\pi_{\theta_{\text{old}}}(a \mid s)$ and the importance ratio has long since stopped being a reliable correction. TRPO prevents this with the KL constraint. PPO prevents it differently.

## The Clipped Surrogate

The PPO-Clip objective modifies the surrogate by clipping the importance ratio to the interval $[1 - \varepsilon,\, 1 + \varepsilon]$ before multiplying by the advantage:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\Bigl[\min\bigl(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1 - \varepsilon,\, 1 + \varepsilon)\,\hat{A}_t\bigr)\Bigr].$$

The clip parameter $\varepsilon$ is typically 0.1 or 0.2. The minimum of the unclipped and clipped terms is the key to the design.

Consider the two cases separately. When $\hat{A}_t > 0$ — the action was better than expected and should be made more likely — the objective wants $r_t(\theta)$ to grow. The unclipped term grows without bound as $\pi_\theta(A_t \mid S_t)$ increases; the clipped term is constant once $r_t(\theta) > 1 + \varepsilon$. Taking the minimum means the objective stops benefiting from pushing $r_t$ above $1 + \varepsilon$: the gradient with respect to $\theta$ is zero once the ratio exits the clip region from the top. There is no incentive to increase the policy probability further once it has moved more than $\varepsilon$ above the old probability.

When $\hat{A}_t < 0$ — the action was worse than expected and should be made less likely — the objective wants $r_t(\theta)$ to shrink. The unclipped term becomes less negative (improves) as $r_t$ decreases; the clipped term is constant once $r_t(\theta) < 1 - \varepsilon$. Taking the minimum means, again, the objective stops rewarding further reduction of the probability once it has moved more than $\varepsilon$ below the old probability.

The result is a pessimistic lower bound on the unclipped surrogate: the clipped objective removes the incentive to make large changes to the policy in either direction. It does not prevent large changes — there is no hard constraint — but it removes the gradient signal that would drive them. The policy can still move outside the clip region, but it gets no credit for doing so.

Geometrically, the clipped objective defines a flat region in the landscape wherever $r_t(\theta)$ is outside $[1 - \varepsilon, 1 + \varepsilon]$. Standard gradient ascent on this objective will drive the policy to the boundary of the trust region and then stop, without the penalty of an explicit Lagrange multiplier and without the overhead of a constraint solver.

## Why the Minimum, Not the Clip Alone

A natural question: why take the minimum of the unclipped and clipped terms rather than simply using the clipped term? The minimum has a specific, asymmetric effect.

For $\hat{A}_t > 0$: when $r_t < 1 - \varepsilon$, the clipped term is constant at $(1-\varepsilon)\hat{A}_t$ and the unclipped term $r_t \hat{A}_t$ is smaller — so the minimum equals the unclipped term, and the gradient is nonzero. The policy is still encouraged to increase $r_t$ from below $1 - \varepsilon$ back toward 1. The clip only removes the incentive to go beyond $1 + \varepsilon$, not to move from a bad position in the right direction.

For $\hat{A}_t < 0$: when $r_t > 1 + \varepsilon$, the clipped term is constant at $(1+\varepsilon)\hat{A}_t$ and the unclipped term $r_t \hat{A}_t$ is more negative — so the minimum equals the unclipped term, and the gradient is nonzero. The policy is still discouraged from having $r_t$ far above 1 when the advantage is negative. The clip removes the incentive to push $r_t$ below $1 - \varepsilon$ when $\hat{A}_t < 0$, not to correct $r_t$ that has already grown too large.

The minimum construction thus creates an objective that is conservative in exactly the right way: it penalizes changes that move the ratio in the direction of the advantage beyond $\varepsilon$, while preserving gradients that correct the ratio when it has already moved too far in the wrong direction. This asymmetry ensures the policy is always nudged toward $r_t \approx 1$ from either side, without being given incentive to move too far in any one direction on a single update.

## The Adaptive KL Variant

PPO was published with a second variant alongside the clipped objective: **PPO-KL**, which uses the same unconstrained surrogate as TRPO but adds an adaptive penalty on the KL divergence:

$$L^{\text{KL}}(\theta) = \mathbb{E}_t\!\bigl[r_t(\theta)\,\hat{A}_t\bigr] - \beta\, \overline{\text{KL}}\!\bigl[\pi_{\theta_{\text{old}}},\, \pi_\theta\bigr],$$

where $\beta$ is updated after each epoch: if the observed KL divergence exceeds a target $d_{\text{targ}}$ by more than a factor of 1.5, increase $\beta$ by a factor of 2; if it is below $d_{\text{targ}} / 1.5$, decrease $\beta$ by a factor of 2. The penalty coefficient adapts to enforce the trust region softly, without the conjugate gradient solve required by TRPO's hard constraint.

The adaptive KL variant is theoretically cleaner — the KL divergence is a direct measure of policy change, with a precise connection to the improvement bound. In the original paper's experiments, however, the clipped objective consistently outperformed the adaptive KL variant across MuJoCo locomotion tasks. The clip was not only simpler but empirically stronger. PPO-KL is described in the paper and rarely used in practice; PPO-Clip became the standard.

## Multiple Epochs of Minibatch Updates

The practical advantage of PPO over both REINFORCE and TRPO is not only the clipped objective but what the clipped objective enables: **multiple epochs of minibatch gradient updates on the same batch of trajectory data**.

REINFORCE and standard on-policy actor-critic methods discard a trajectory after one gradient update — taking one step in parameter space per batch of environment interactions. TRPO also takes one update per batch (the conjugate gradient solve produces a single step). This is enormously wasteful: a batch of $N$ transitions cost $N$ environment steps to collect, and a single gradient step extracts only a fraction of the information available in those $N$ transitions.

PPO collects a batch of $T$ transitions from $K$ parallel workers — typically $T = 2048$ per worker, $K = 8$, yielding $16{,}384$ transitions per batch — then runs $M$ epochs of minibatch gradient ascent on the data, with minibatch size $m \ll KT$. Each epoch shuffles the data and processes it in minibatches; the parameter $\theta$ is updated at each minibatch. After $M$ epochs, the batch is discarded and a new batch is collected from the updated policy.

The clipped objective makes this safe. After the first epoch, the updated $\theta$ differs from $\theta_{\text{old}}$, and $r_t(\theta) \neq 1$. Without clipping, later minibatch updates would compute gradients that are corrupted by large importance ratios — the distribution shift between the current $\theta$ and the data-generating $\theta_{\text{old}}$ would accumulate. The clipped objective caps the importance ratio at $1 \pm \varepsilon$, bounding the distribution shift and keeping the gradient estimates reliable throughout all $M$ epochs. In practice, $M = 10$ epochs on the same batch is standard, extracting roughly $10\times$ more gradient signal per environment interaction than a single-step method — a direct improvement in sample efficiency.

## The Combined Objective

The full PPO loss function combines the clipped policy objective with the value function regression and an entropy bonus:

$$\mathcal{L}(\theta) = \mathbb{E}_t\!\Bigl[L^{\text{CLIP}}_t(\theta) - c_1\,\bigl(\hat{V}_\theta(S_t) - V_t^{\text{target}}\bigr)^2 + c_2\,H\!\bigl[\pi_\theta(\cdot \mid S_t)\bigr]\Bigr],$$

where $c_1 \approx 0.5$ and $c_2 \approx 0.01$ are scalar coefficients. The value function loss $c_1(\hat{V}_\theta - V^{\text{target}})^2$ trains the critic using the same $M$ epochs of minibatch updates as the actor; the entropy bonus $c_2 H[\pi_\theta]$ encourages exploration. When the actor and critic share a network backbone — which is standard in the original PPO implementation — the three terms compete for the same parameters, and the coefficients $c_1$ and $c_2$ balance their relative contributions to the shared representation.

The value function target $V_t^{\text{target}}$ is typically the GAE return $\hat{A}_t + \hat{V}_{\theta_{\text{old}}}(S_t)$ — the advantage estimate shifted by the old value estimate to produce an estimate of $Q^{\pi}(S_t, A_t)$, which serves as a bootstrap target for the critic. This couples the actor and critic training through the advantage estimate: a better critic produces better advantage estimates, which produce better actor updates, which collect better trajectories, which improve the critic. The feedback loop is tight and, with PPO's stable updates, constructive rather than oscillatory.

## Hyperparameter Sensitivity and Practical Defaults

PPO is often described as robust to hyperparameter choices — a reputation that is partly deserved and partly a consequence of the specific defaults that the original paper and subsequent open-source implementations established. The canonical PPO configuration, essentially unchanged since 2017, is: clip parameter $\varepsilon = 0.2$, $K = 8$ parallel workers, $T = 2048$ steps per rollout, minibatch size $m = 64$, $M = 10$ update epochs, GAE with $\lambda = 0.95$, $\gamma = 0.99$, Adam with learning rate $3 \times 10^{-4}$, linear learning rate decay to zero over training.

These defaults work well on the MuJoCo locomotion suite and Atari with minor tuning. They work less well on tasks with sparse rewards, very long horizons, or high-dimensional discrete action spaces, where the clip threshold $\varepsilon$, the rollout length $T$, and the number of parallel workers require task-specific tuning. The robustness of PPO is real but bounded: it is robust within the family of continuous control and dense-reward tasks that have been its primary application domain.

The sensitivity that does matter is the clip threshold $\varepsilon$. Too small ($\varepsilon = 0.05$) and PPO degenerates toward a single gradient step per batch — the clip activates immediately, each epoch produces near-zero gradient, and sample efficiency collapses. Too large ($\varepsilon = 0.5$) and PPO loses its stabilizing effect — importance ratios can grow large before the clip activates, reintroducing the instability that the clip was designed to prevent. The value 0.2 represents an empirically calibrated middle ground.

## What PPO Gives Up

The clipped objective is a heuristic rather than a principled constraint. TRPO's hard KL constraint guaranteed that each update remained within the trust region with certainty — the constraint was enforced as a mathematical fact, not as a tendency. PPO's clipped objective creates a flat region in the landscape that discourages large updates, but it does not prevent them. If the gradient of the unclipped surrogate points strongly outside the clip region, a large learning rate can push the policy across the clip boundary, producing an update that the clipped objective assigns zero gradient but the true objective may penalize severely.

PPO therefore lacks TRPO's monotone improvement guarantee. There is no theorem that guarantees each PPO update improves $J(\theta)$. In practice, PPO updates almost always improve or maintain performance — the clip does its job — but occasional catastrophic updates occur, particularly late in training when the policy entropy is low and small changes in the distribution of a near-deterministic policy correspond to large shifts in the KL divergence. Practitioners address this by monitoring KL divergence empirically and stopping the inner optimization loop early if the KL diverges beyond a threshold, a manual approximation to the TRPO hard constraint.

In exchange for the absent guarantee, PPO provides compatibility with standard first-order optimizers, straightforward extension to recurrent and multi-task architectures, and the ability to extract far more gradient signal per environment interaction than any single-step method. The tradeoff proved to be the right one for almost every practical application of 2017–2025 deep RL.

## Why PPO Dominates

The success of PPO is not easily reducible to a single property. It is the combination: the clipped objective is simple enough that the full algorithm — data collection, GAE computation, minibatch updates, value function training — fits in a few hundred lines of standard deep learning code. The multiple epochs are efficient enough that PPO reaches the same performance as TRPO in a fraction of the wall-clock time, using a fraction of the engineering infrastructure. The defaults are good enough that most users get competitive results without extensive tuning.

Beyond locomotion and Atari, PPO was adopted as the policy optimization algorithm for OpenAI Five (Dota 2), AlphaStar's initial league training, InstructGPT and subsequent RLHF systems for language model alignment, and a large fraction of the robotics learning literature after 2018. Each of these applications required modifications — token-level MDPs for language models, distributed implementations for large-scale game-playing, modified reward structures for robot manipulation — but the core PPO loop was preserved in each case. An algorithm that began as a simpler version of TRPO became, within two years of its publication, the most widely deployed policy optimization algorithm in deep RL.

The reason, ultimately, is not theoretical elegance but engineering fit. Deep RL had moved from academic benchmarks to large-scale systems where implementation cost, infrastructure compatibility, and debuggability matter as much as sample efficiency. PPO fit that environment; TRPO did not. The history of the field after 2017 is in large part the history of building on the PPO foundation — for continuous control, for language model alignment, for multi-agent systems — and the questions that motivated TRPO and PPO remain live in each new application domain. Ensuring that gradient steps do not destroy a policy that took millions of environment interactions to build is not a problem that admits a final solution; it recurs in every setting where learning is expensive and failure is costly.

---

*Proximal Policy Optimization is introduced in Schulman, Wolski, Dhariwal, Radford, and Klimov (2017), Proximal Policy Optimization Algorithms. The conservative policy iteration framework underlying the surrogate objective is from Kakade and Langford (2002), Approximately Optimal Approximate Reinforcement Learning. The importance-weighted policy gradient that PPO generalizes is discussed in connection with TRPO in Schulman, Levine, Jordan, Abbeel, and Moritz (2015). The empirical comparisons establishing PPO's advantages over TRPO, A3C, and CEM on MuJoCo and Atari are in Schulman et al. (2017). Applications to large-scale game-playing are documented in OpenAI Five (Berner et al., 2019), Dota 2 with Large Scale Deep Reinforcement Learning, and Vinyals et al. (2019), Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning. Applications to language model alignment appear in Ouyang et al. (2022), Training Language Models to Follow Instructions with Human Feedback (InstructGPT).*
