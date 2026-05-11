---

title: "Natural Policy Gradients and TRPO"
subtitle: "Fisher information, trust regions, conjugate gradients, and the monotone improvement guarantee"

---


The actor-critic methods of the previous chapter resolved the variance problem that made REINFORCE impractical. They did not resolve — and in some ways sharpened — a different problem: the sensitivity of policy gradient updates to the choice of step size.

The difficulty is structural. In supervised learning, the parameters $\theta$ live in a Euclidean space: the Euclidean distance $\|\theta' - \theta\|$ is a meaningful measure of how much the model has changed, and a gradient step of size $\alpha$ in parameter space produces a change in model behavior that scales predictably with $\alpha$. In reinforcement learning with parameterized policies, this correspondence breaks down. A small change in $\theta$ in one region of parameter space can radically alter the policy — collapsing a previously diffuse distribution to a near-deterministic one, or reversing the ordering of action probabilities — while an equally large change in another region may be nearly invisible in behavior space. The Euclidean geometry of parameter space is not the right geometry for measuring policy change.

The consequence in practice: an aggressive step size that works well for ten iterations suddenly destroys the policy on the eleventh, because the gradient happened to point into a region of parameter space where the policy landscape is steep. Practitioners resorted to heuristics — small fixed step sizes, periodic restarts, gradient clipping — that traded learning speed for stability without theoretical backing. The problem was not implementation but geometry.

## The Natural Gradient

The resolution, developed by Amari (1998) in the context of neural network optimization and applied to policy gradients by Kakade (2001), is to replace the Euclidean gradient with the **natural gradient**: the gradient expressed in the intrinsic geometry of the distribution space.

The key observation is that policies $\pi_\theta$ are probability distributions, and probability distributions form a statistical manifold equipped with a canonical metric: the **Fisher information matrix**. For a policy $\pi_\theta$ over actions given states, the Fisher information is:

$$F(\theta) = \mathbb{E}_{\pi_\theta}\!\bigl[\nabla_\theta \log \pi_\theta(A \mid S)\, \nabla_\theta \log \pi_\theta(A \mid S)^\top\bigr],$$

where the expectation is over both states (drawn from the on-policy state distribution) and actions (drawn from $\pi_\theta(\cdot \mid S)$). $F(\theta)$ is a positive semi-definite matrix of the same dimension as $\theta$, and it encodes local curvature of the distribution manifold: directions in parameter space that cause large changes in the policy have large Fisher information in those directions; directions that are nearly redundant — where moving the parameters has little effect on the distribution — have small Fisher information.

The natural gradient $\tilde{\nabla}_\theta J(\theta)$ is obtained by pre-multiplying the ordinary gradient by the inverse Fisher information:

$$\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta).$$

This is gradient ascent not in parameter space but in **distribution space**, measured by the KL divergence. More precisely, the natural gradient solves:

$$\tilde{\nabla}_\theta J(\theta) = \arg\max_{\Delta\theta} \nabla_\theta J(\theta)^\top \Delta\theta \quad \text{subject to} \quad \Delta\theta^\top F(\theta) \Delta\theta \leq \varepsilon,$$

the steepest ascent direction under the constraint that the KL divergence between the old and new policies stays within a ball of radius $\varepsilon/2$. In other words, the natural gradient is the direction of steepest ascent when progress is measured in units of policy change, not parameter change.

The practical meaning is this: if two parameterizations of the same policy produce the same distribution, the natural gradient is the same in both. The update is parameterization-invariant in a way that the Euclidean gradient is not. A reparameterization that stretches parameter space in one direction does not change the natural gradient direction, because the Fisher metric stretches correspondingly.

## The Natural Policy Gradient

Kakade (2001) applied this idea directly to policy optimization, defining the **natural policy gradient** as the natural gradient of the performance objective:

$$\theta \leftarrow \theta + \alpha \, F(\theta)^{-1} \nabla_\theta J(\theta).$$

The theoretical appeal is clear. Empirically, Kakade showed on small problems that the natural policy gradient converges in far fewer iterations than the ordinary policy gradient, because its steps are proportional to changes in policy behavior rather than to changes in the magnitude of parameter updates.

The practical obstruction is equally clear. $F(\theta)$ is a $d \times d$ matrix where $d$ is the number of parameters — millions in a modern deep network. Forming $F(\theta)$ explicitly costs $O(d^2)$ memory and $O(d^3)$ time for the inversion. Neither is remotely feasible. For the natural policy gradient to be useful at scale, a different approach to the inversion is needed.

## The Surrogate Objective

Before reaching the algorithmic machinery, it is worth pausing on the object that trust region methods actually optimize. Define the **surrogate objective**:

$$L_{\theta_{\text{old}}}(\theta) = \mathbb{E}_{(S_t, A_t) \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)} A^{\pi_{\theta_{\text{old}}}}(S_t, A_t)\right].$$

This is the performance objective $J(\theta)$ approximated by an importance-weighted average under the old policy. It has two useful properties. First, $L_{\theta_{\text{old}}}(\theta_{\text{old}}) = J(\theta_{\text{old}})$: the surrogate and the true objective agree at the current policy. Second, $\nabla_\theta L_{\theta_{\text{old}}}(\theta)\big|_{\theta = \theta_{\text{old}}} = \nabla_\theta J(\theta)\big|_{\theta = \theta_{\text{old}}}$: their gradients agree at the current policy. Locally, optimizing the surrogate is equivalent to optimizing the true objective.

The problem is that the surrogate may diverge from the true objective when $\theta$ moves far from $\theta_{\text{old}}$. The importance ratios $\pi_\theta / \pi_{\theta_{\text{old}}}$ can become large or small, making the surrogate an unreliable proxy for the true performance. The gap between surrogate and true objective is controlled by how much the policy has changed — which is precisely what a trust region constraint limits.

Schulman, Levine, Jordan, Abbeel, and Moritz (2015) formalized this in a **policy improvement bound**. Building on earlier work by Kakade and Langford (2002) and Pirotta, Restelli, Pecorino, and Calandriello (2013), they showed:

$$J(\theta) \geq L_{\theta_{\text{old}}}(\theta) - C \cdot \max_s \text{KL}\!\bigl[\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s)\bigr],$$

where $C = \frac{4\gamma \epsilon}{(1 - \gamma)^2}$ and $\epsilon = \max_{s,a} |A^{\pi_{\theta_{\text{old}}}}(s, a)|$. Any $\theta$ that increases $L_{\theta_{\text{old}}}(\theta)$ by more than $C$ times the KL divergence is guaranteed to increase the true objective $J(\theta)$. This is the **monotone improvement guarantee**: a policy update is safe whenever the KL divergence is small enough to keep the surrogate correction term smaller than the surrogate gain.

## TRPO: The Constrained Problem

The **Trust Region Policy Optimization** algorithm (TRPO) translates the monotone improvement bound into a tractable optimization problem. Rather than using the looser theoretical bound — which involves $C$ depending on the worst-case advantage and tends to be extremely conservative — TRPO uses the mean KL divergence as the constraint:

$$\max_\theta \quad L_{\theta_{\text{old}}}(\theta) \quad \text{subject to} \quad \mathbb{E}_{S \sim d^{\pi_{\theta_{\text{old}}}}}\!\bigl[\text{KL}\bigl(\pi_{\theta_{\text{old}}}(\cdot \mid S) \,\|\, \pi_\theta(\cdot \mid S)\bigr)\bigr] \leq \delta.$$

The constraint $\delta$ — typically set to $0.01$ or $0.02$ — defines the **trust region**: the set of policies that are close enough to the current one that the surrogate objective is a reliable proxy for the true objective. Within the trust region, we optimize the surrogate. Outside it, we do not trust the surrogate.

The KL constraint is equivalent, to second order in $(\theta - \theta_{\text{old}})$, to the Fisher information constraint $(\theta - \theta_{\text{old}})^\top F(\theta_{\text{old}}) (\theta - \theta_{\text{old}}) \leq \delta$. This is the connection to the natural gradient: the TRPO update, to first order, is exactly a natural gradient step with step size chosen to saturate the KL constraint.

The Lagrangian relaxation of the constrained problem gives:

$$\theta^* = \theta_{\text{old}} + \sqrt{\frac{2\delta}{g^\top F^{-1} g}} \, F^{-1} g,$$

where $g = \nabla_\theta L_{\theta_{\text{old}}}(\theta)\big|_{\theta_{\text{old}}}$ is the policy gradient. The factor $\sqrt{2\delta / g^\top F^{-1} g}$ is the step size chosen so the natural gradient step exactly saturates the KL constraint $\Delta\theta^\top F \Delta\theta = 2\delta$.

## Conjugate Gradients and Fisher-Vector Products

Computing $F^{-1} g$ directly is infeasible. TRPO avoids explicit inversion by noting that $F^{-1} g$ is the solution to the linear system $Fx = g$, which can be solved iteratively with the **conjugate gradient** method — never forming $F$ explicitly, only computing matrix-vector products $Fv$ for arbitrary vectors $v$.

The Fisher-vector product can be estimated efficiently. Using the definition of $F$ as an expectation:

$$Fv = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\!\bigl[(\nabla_\theta \log \pi_\theta \cdot v)\, \nabla_\theta \log \pi_\theta\bigr],$$

which can be computed by a double backward pass: compute $\nabla_\theta \log \pi_\theta$, take its dot product with $v$, then differentiate again with respect to $\theta$. This costs $O(d)$ memory and $O(d)$ additional compute per conjugate gradient iteration, rather than $O(d^2)$. Alternatively, the Fisher-vector product can be computed as the Hessian-vector product of the KL divergence:

$$Fv \approx \nabla_\theta \bigl(\nabla_\theta \text{KL}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta) \cdot v\bigr),$$

which is equivalent for distributions in the exponential family and is the form used in the original TRPO implementation.

The conjugate gradient method requires roughly 10 iterations to produce a good approximation to $F^{-1} g$ for typical policy network sizes, each costing a Fisher-vector product. The resulting direction $\hat{x} \approx F^{-1} g$ is then normalized to produce the full natural gradient step.

## The Line Search

Computing the natural gradient direction is the main computational step, but following it blindly can still violate the trust region constraint. Quadratic approximations of the KL divergence are only locally accurate, and the Fisher matrix estimated from a finite batch of trajectories introduces sampling noise. TRPO adds a **backtracking line search**: starting from the full natural gradient step, repeatedly halve the step size until both the KL constraint is satisfied and the surrogate objective has increased:

$$\theta_{\text{new}} = \theta_{\text{old}} + \beta^k \hat{x},$$

where $\hat{x} = \sqrt{2\delta / \hat{x}_0^\top F \hat{x}_0} \, \hat{x}_0$ is the normalized step, $\hat{x}_0 = F^{-1} g$ is the conjugate gradient solution, and $k$ is the smallest non-negative integer such that both $L_{\theta_{\text{old}}}(\theta_{\text{new}}) > L_{\theta_{\text{old}}}(\theta_{\text{old}})$ and $\bar{\text{KL}}(\theta_{\text{new}}) \leq \delta$. The factor $\beta \in (0, 1)$, typically $\beta = 0.5$, controls the shrinkage rate.

The line search is essential for correctness: it recovers monotone improvement even when the quadratic approximation of the KL divergence is inaccurate. In practice, the full step is accepted most of the time, and the line search terminates after one or two halvings in cases of unusually curved KL geometry or noisy gradient estimates. Its cost is small relative to the conjugate gradient solve.

## On-Policy Sampling and Practical Computation

TRPO requires a batch of trajectories collected from the current policy $\pi_{\theta_{\text{old}}}$ at every update. Unlike DQN, which amortizes each environment transition across many gradient updates via the replay buffer, TRPO discards each batch of trajectories after a single update. This is not inefficiency — it is correctness. The surrogate objective $L_{\theta_{\text{old}}}(\theta)$ uses importance weights $\pi_\theta / \pi_{\theta_{\text{old}}}$ that are only valid when the data distribution is $\pi_{\theta_{\text{old}}}$. Reusing old trajectories would introduce distribution shift that is not accounted for by the importance weights, invalidating the trust region guarantee.

The practical consequence: TRPO requires many environment interactions per parameter update, and its sample complexity is dominated by trajectory collection rather than the conjugate gradient computation. For environments where each step is cheap — physics simulators, video games — this is acceptable. For environments where each step is expensive — real-world robotics, wet-lab experiments — it is a significant constraint.

A single TRPO update for a continuous control policy with a 128-dimensional state space, 6-dimensional action space, and a two-hidden-layer network requires collecting $O(10^4)$ transitions, running $O(10)$ conjugate gradient iterations with Fisher-vector products, and performing $O(10)$ line search steps. The wall-clock time per update is orders of magnitude greater than a DQN update on the same hardware, but the number of updates required to reach a given performance level is also substantially smaller. The tradeoff is compute per update versus updates required.

## What TRPO Achieved

On the MuJoCo continuous control benchmark — locomotion tasks for simulated bodies including a half-cheetah, ant, hopper, and humanoid — TRPO produced results that had been essentially unachievable by prior methods. The humanoid locomotion task, which requires coordinating over 20 joints to produce a stable walking gait, was solved by TRPO with a few million transitions where REINFORCE and A3C had failed entirely. The trust region constraint prevented the catastrophic policy collapses that made those methods unreliable on tasks where large parameter-space steps changed policy behavior dramatically.

TRPO also unified two lines of work that had developed separately. The natural policy gradient line, from Kakade (2001) through Peters and Schaal (2008) and Bagnell and Schneider (2003), had established the theoretical superiority of the Fisher-metric gradient direction. The policy improvement bound line, from Kakade and Langford (2002) onward, had established that the KL divergence controls the gap between surrogate and true objective. TRPO combined both: the natural gradient direction and the KL trust region constraint are, to second order, the same thing. The theoretical framework and the algorithmic framework converged.

## The Remaining Cost

TRPO's theoretical guarantees come at a price that scales poorly. The conjugate gradient solve requires computing Fisher-vector products, each of which costs a double backward pass through the policy network. For a network with $d$ parameters and a batch of $N$ transitions, each Fisher-vector product costs $O(Nd)$ — the same as a standard gradient step, but the conjugate gradient requires $O(10)$ such products per update. The total compute per update is roughly 10–20 times that of a standard policy gradient step.

More importantly, TRPO is difficult to implement correctly. The conjugate gradient solver must handle rank-deficiency in the empirical Fisher estimate, the damping coefficient $\lambda$ added to $F + \lambda I$ to ensure positive definiteness must be tuned, and the line search must handle both the KL constraint and the surrogate improvement condition. The full implementation is several hundred lines of nontrivial numerical code. None of this is algorithmically deep — it is engineering — but it creates a significant barrier to use and to extending the method.

The question Schulman, Wolski, Dhariwal, Radford, and Klimov would address in 2017 was whether the trust region constraint could be enforced by a simpler mechanism — one that did not require conjugate gradients, did not require a constrained optimization solve, and could be implemented in a dozen lines of standard deep learning code — while preserving the essential property that each update remains within a safe region of policy space. The answer was PPO, and it turned out that a carefully designed penalty rather than a hard constraint was enough.

---

*The natural gradient for statistical models is developed in Amari (1998), Natural Gradient Works Efficiently in Learning. The natural policy gradient is introduced in Kakade (2001), A Natural Policy Gradient. The connection between the natural gradient and Fisher information in the policy gradient context is analyzed in Peters and Schaal (2008), Reinforcement Learning of Motor Skills with Policy Gradients. The policy improvement bound underlying TRPO builds on Kakade and Langford (2002), Approximately Optimal Approximate Reinforcement Learning, and is sharpened in Pirotta, Restelli, Pecorino, and Calandriello (2013), Safe Policy Iteration. Trust Region Policy Optimization itself is in Schulman, Levine, Jordan, Abbeel, and Moritz (2015), Trust Region Policy Optimization. The Fisher-vector product technique for scalable natural gradient computation is described in Pearlmutter (1994), Fast Exact Multiplication by the Hessian, and is applied to TRPO in Schulman et al. (2015). The conjugate gradient method used for the inner solve is the classical Hestenes and Stiefel (1952) algorithm.*
