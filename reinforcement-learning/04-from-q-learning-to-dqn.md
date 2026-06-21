---
title: "From Q-Learning to Deep Q-Networks"
subtitle: "Convergence guarantees, the deadly triad, experience replay, target networks, and the Atari result"
---


Q-learning ended the previous chapter with a clean update rule and an informal guarantee: it performs stochastic value iteration, converging to the optimal action-value function without ever requiring the transition kernel. For twenty years after Watkins introduced it, however, the gap between that theoretical promise and practical reach remained stubbornly wide. The convergence guarantee was complete — but it required a lookup table, and a lookup table required a finite, enumerable state space. Attempts to bridge this gap by substituting neural networks produced erratic results. Neural Q-learning trained, then diverged. It learned part of a task, then forgot what it had learned. It worked on some seeds and catastrophically failed on others.

This chapter does two things. First, it makes the tabular convergence guarantee precise — identifying exactly which conditions it rests on and what breaks when those conditions are violated. Second, it follows that analysis directly into DQN: the 2013 engineering solution that made neural Q-learning work in practice, understood as a response to each specific failure mode the analysis predicted.

---

## The Convergence Theorem

> [!question]
> Q-learning bootstraps from its own estimates — a circular dependency. Why does this not cause the estimates to spiral away from $Q^*$?

The formal convergence result was established by Watkins and Dayan (1992) for deterministic environments and extended to stochastic MDPs by Tsitsiklis (1994).

> [!success]
> **Theorem (Watkins & Dayan 1992; Tsitsiklis 1994).** Let $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ be a finite MDP with bounded rewards $|r| \leq R_{\max}$. Starting from any bounded $Q_0$, let $Q_t$ denote the Q-learning iterates. Suppose:
> 1. Every state-action pair $(s, a)$ is visited infinitely often.
> 2. The step sizes satisfy $\sum_t \alpha_t(s, a) = \infty$ and $\sum_t \alpha_t(s, a)^2 < \infty$ for all $(s, a)$.
>
> Then $Q_t(s, a) \to Q^*(s, a)$ with probability 1 for all $(s, a)$.
>
> Three features are worth noting: the theorem makes no assumptions about the *behavior* policy beyond the visitation condition — Q-learning is truly off-policy; the Robbins-Monro step size conditions are the same as in every stochastic approximation result; and the theorem is existential — it says nothing about how fast convergence occurs.

## Q-Learning as Stochastic Approximation

> [!info]
> **Stochastic approximation** finds the fixed point of an operator $F$ by iterating $x_{t+1}(i) = x_t(i) + \alpha_t(i)[F_t(x_t, i) + \varepsilon_t(i)]$ using noisy estimates of $F$. Convergence holds when $F$ is a contraction and $\varepsilon_t$ is zero-mean noise with bounded variance.

Write the Q-learning update at $(S_t, A_t)$ and decompose the increment into its expectation plus noise:

$$R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - Q_t(S_t, A_t) = \underbrace{\bigl[(\mathcal{T}^* Q_t)(S_t, A_t) - Q_t(S_t, A_t)\bigr]}_{F(Q_t,\, S_t,\, A_t)} + \underbrace{\varepsilon_t}_{\text{zero-mean noise}}.$$

The noise-free update direction is the displacement from the current estimate toward its Bellman target. Its unique root is $Q^*$ — the fixed point of $\mathcal{T}^*$. The noise $\varepsilon_t$ is zero-mean given $Q_t$ and $(S_t, A_t)$, with variance bounded by the reward variance plus the range of Q-values.

> [!note]
> **Why the contraction closes the argument.** For any two Q-functions $Q, Q'$:
> $$\|\mathcal{T}^* Q - \mathcal{T}^* Q'\|_\infty \leq \gamma \|Q - Q'\|_\infty.$$
> The Bellman error $Q - Q^*$ shrinks geometrically under the noise-free dynamics. The stochastic iterates track these dynamics with additive noise; diminishing step sizes ensure the noise is eventually overwhelmed.
>
> Tsitsiklis' proof formalizes this via the **ODE method**: under Robbins-Monro conditions, stochastic iterates concentrate on the trajectories of the ODE $\dot{x} = F(x)$ as time progresses. The ODE has a globally asymptotically stable equilibrium at $Q^*$ — the Lyapunov function $V(Q) = \|Q - Q^*\|_\infty$ satisfies $\dot{V} \leq -cV$ for some $c > 0$ — so the stochastic iterates converge to the same equilibrium.

---

## The Exploration Requirement

> [!question]
> What does "every state-action pair visited infinitely often" actually require from the behavior policy?

A purely greedy policy fails immediately: if current Q-estimates make one action appear optimal at every state, the agent never explores alternatives. State-action pairs with poor initial estimates are never updated and remain poor — the algorithm produces a suboptimal policy.

> [!info]
> **GLIE (Greedy in the Limit with Infinite Exploration).** A behavior policy satisfying:
> $$\varepsilon_t \to 0 \quad \text{and} \quad \sum_t \varepsilon_t = \infty.$$
> The first condition ensures the policy converges to greedy. The second ensures infinitely many exploratory actions, so every pair is visited infinitely often. The simplest GLIE schedule is $\varepsilon_t = c/t$.

In practice, constant or linearly decaying $\varepsilon$ is far more common. With constant $\varepsilon$, iterates converge to a neighborhood of $Q^*$ rather than $Q^*$ itself — but the neighborhood can be made arbitrarily small, and the estimates remain stable. This is often preferable to the slow decay required for exact GLIE convergence.

---

## The Tabular Representation's Hidden Role

> [!tip]
> A detail easy to overlook: the lookup table itself is a technical condition of the convergence proof, not just an implementation detail.
>
> In the tabular setting, $Q(s, a)$ is an independent real number for each $(s, a)$, stored in a separate memory cell. An update to $Q(S_t, A_t)$ changes exactly one cell and leaves all others unchanged. This **independence** is what makes the stochastic approximation argument go through: the update at $(S_t, A_t)$ is an unbiased estimate of $(\mathcal{T}^* Q)(S_t, A_t) - Q(S_t, A_t)$ with zero-mean noise, and averaging over many transitions from that pair recovers the expected Bellman backup entry-by-entry.

The moment a parameterized function approximator is introduced — even a linear one — this independence is destroyed. If $Q(s, a; \theta) = \theta^\top \phi(s, a)$ shares parameters $\theta$ across all state-action pairs, an update based on the transition at $(S_t, A_t)$ changes the Q-value for *every* pair simultaneously. An update designed to reduce the Bellman error at one pair can increase it at others. The per-entry convergence argument breaks completely.

---

## The Deadly Triad

> [!info]
> Sutton's **deadly triad**: three ingredients that, combined, produce reinforcement learning algorithms that can diverge:
> 1. **Function approximation** — representing $Q$ with a parameterized function.
> 2. **Bootstrapping** — using $Q$-estimates to generate targets for updating $Q$.
> 3. **Off-policy learning** — learning about a target policy (greedy) from data generated by a different behavior policy.
>
> Each *pair* of the three is safe: function approximation + Monte Carlo (no bootstrap, on-policy) converges; tabular TD (no function approximation) converges; off-policy Monte Carlo (no bootstrap, no function approximation) converges. All three together: no guarantee.

> [!warning]
> **Baird's counterexample (1995).** A seven-state MDP with two actions, zero rewards everywhere, and $Q^* = 0$ representable by a linear function approximator. Despite these favorable conditions, semi-gradient Q-learning with linear function approximation causes the parameter vector to grow without bound. Q-estimates diverge to infinity.
>
> The mechanism: the norm mismatch. The sup-norm contraction guarantees $\mathcal{T}^*$ shrinks the worst-case error, but semi-gradient descent minimizes the **projected Bellman error** — squared Bellman error weighted by the distribution of states visited by the behavior policy. When the behavior policy visits some states rarely, their Bellman errors can grow without penalty. If bootstrap targets for frequently visited states depend on Q-values of rarely visited states, updates can increase unweighted Bellman errors even as they decrease weighted ones.
>
> Baird's example is not a pathological edge case — it is a demonstration of the fundamental conflict between the sup-norm geometry of $\mathcal{T}^*$ and the $\ell_2$ geometry of gradient-based learning. This conflict is unavoidable whenever all three triad components co-occur, which is precisely the combination practical deep RL requires.

---

## Deep Q-Networks: The Engineering Response

> [!question]
> The deadly triad cannot be eliminated without giving up function approximation, bootstrapping, or off-policy data — all of which are essential at scale. Can it be *suppressed* sufficiently to make neural Q-learning work in practice?

In December 2013, DeepMind posted "Playing Atari with Deep Reinforcement Learning." The same architecture, the same hyperparameters, the same reward structure across seven Atari games — with performance exceeding human experts on three. By the full 2015 *Nature* study (49 games), it had catalyzed a new era. The stabilizing machinery was two ideas — neither theoretically sufficient to eliminate the deadly triad, both empirically sufficient to suppress it.

### The Architecture

The raw input is a sequence of Atari frames: four consecutive $84 \times 84$ grayscale images stacked into a $4 \times 84 \times 84$ tensor.

> [!note]
> **Why stack frames?** A single frame does not reveal velocity — you cannot tell from one image which direction the ball is moving. Without velocity, the state is not Markovian. Four frames capture enough motion history that the Markov assumption is approximately satisfied. This is the simplest solution to the partial observability problem in visual environments.

Three convolutional layers with ReLU activations, followed by two fully connected layers, produce Q-values for all actions simultaneously in a single forward pass ($\approx 1.7$M parameters). The convolutional structure provides **weight sharing**: a feature detector learned in one part of the screen applies everywhere. Similar game states — differing in object positions, not identities — receive similar Q-value estimates. This is the spatial generalization that a lookup table cannot provide.

---

### The First Stabilizer: Experience Replay

> [!question]
> Consecutive frames are nearly identical. How does training on such correlated samples cause instability, and how can it be broken?

The correlation problem: gradient updates computed on consecutive transitions are nearly identical, causing the network to overfit to the current region of the game while forgetting earlier ones — **catastrophic forgetting**.

> [!tip]
> **Experience replay** (Lin, 1992): at each step, store the transition $(S_t, A_t, R_{t+1}, S_{t+1})$ in a fixed-size **replay buffer** $\mathcal{D}$, discarding oldest transitions as the buffer fills. Rather than updating on the current transition, sample a random minibatch from $\mathcal{D}$ and compute the gradient on that minibatch.
>
> Three benefits in one mechanism:
> 1. **Decorrelation** — random sampling breaks temporal correlation; the optimizer sees something approximating i.i.d. data.
> 2. **Data efficiency** — each transition is used for many updates before discarding, extracting more gradient signal per environment interaction.
> 3. **Distribution stabilization** — the buffer contains transitions from many past policies, preventing the training distribution from collapsing to the current policy's experience.

Experience replay makes DQN more deeply off-policy than tabular Q-learning: not only does the greedy target policy differ from the $\varepsilon$-greedy behavior policy, but the data comes from a historical mixture of many past behavior policies. In practice, the buffer covers enough of the relevant state space to make training stable.

---

### The Second Stabilizer: Target Networks

> [!question]
> In supervised learning, targets are fixed. In Q-learning, the target depends on the same parameters being optimized. Why is this a problem, and how can it be broken?

> [!warning]
> **The moving target problem.** The training target for $Q(S_t, A_t; \theta)$ is $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta)$ — a quantity that depends on the same parameters $\theta$ being optimized. Every gradient step changes both predictions and targets simultaneously. The network is chasing its own tail: a step that reduces Bellman error at $(S_t, A_t)$ shifts bootstrap targets for all nearby pairs, which can amplify rather than dampen the error — the mechanism of Baird's counterexample in the neural setting.

> [!tip]
> **Target networks**: a separate copy of the Q-network with parameters $\theta^-$, identical in architecture but updated far less frequently. The training target becomes:
> $$y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a';\, \theta^-).$$
> The online network $\theta$ is updated by gradient descent at every step. The target network $\theta^-$ is held fixed for $C = 10{,}000$ steps, then replaced by a copy of $\theta$.
>
> With a frozen target network, the training objective is locally a supervised regression — fixed targets, gradient descent. The feedback loop causing the moving-target instability is broken. The instability is not eliminated but interrupted every $C$ steps, and if $C$ is large enough, the online network converges substantially toward the current target before the target moves again.
>
> **Analogy to value iteration**: the target network plays the role of the previous iteration's Q-table; the online network updates toward it; the periodic refresh plays the role of the next sweep.

---

### The Loss and the Semi-Gradient

> [!note]
> The DQN training loss is the mean squared Bellman error over a minibatch $\mathcal{B}$ sampled from replay:
> $$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{B}}\!\Bigl[\bigl(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\bigr)^2\Bigr].$$
> The gradient flows through $Q(s, a; \theta)$ but is **stopped** at $Q(s', a'; \theta^-)$, treated as a constant. This is a **semi-gradient** update — taking the full gradient through the target would yield a different, less stable objective. Semi-gradient is standard in all practical deep RL and is one of the implicit conditions making training tractable.

---

## The Atari Result

> [!success]
> On 29 of 49 games, DQN surpassed human expert performance with identical hyperparameters. The benchmark established a standard that the entire subsequent field adopted: percentage of human-level performance, measured by a professional game tester. The pattern of successes and failures was itself informative:
>
> - **Strengths**: games requiring reactive strategies with visually distinct states — Breakout, Pong, Space Invaders — where performance was dramatically superhuman.
> - **Weaknesses**: games requiring long-horizon planning — Montezuma's Revenge, Pitfall — where DQN scored near zero despite millions of training steps. Sparse rewards over long horizons expose the credit propagation bottleneck that the tabular analysis predicted.

---

## Overestimation and Double DQN

> [!warning]
> **Systematic overestimation bias.** The Q-learning target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ maximizes over noisy estimates. By the convexity of max:
> $$\mathbb{E}\bigl[\max_{a'} Q(s', a'; \theta^-)\bigr] \geq \max_{a'} Q^*(s', a').$$
> The maximum of noisy estimates is systematically larger than the true maximum. With ten actions each carrying modest independent noise, the maximum is dominated by whichever has the largest positive noise — regardless of which action is truly optimal. Inflated bootstrap targets propagate positive bias backward; each update uses overestimated targets to generate the next round of overestimated targets.

> [!tip]
> **Double DQN** (van Hasselt et al., 2015): decouple action *selection* from action *evaluation* in the bootstrap target.
>
> | Method | Target |
> |--------|--------|
> | DQN | $r + \gamma Q(s',\, \arg\max_{a'} Q(s', a';\, \theta^-);\, \theta^-)$ |
> | Double DQN | $r + \gamma Q(s',\, \arg\max_{a'} Q(s', a';\, \theta);\, \theta^-)$ |
>
> The online network selects the action; the target network evaluates it. Because both networks carry different noise patterns, the double correction reduces overestimation on average. One additional forward pass per update; consistent performance improvements across the benchmark.

---

## Further Improvements and Rainbow

> [!note]
> By 2018, a half-dozen improvements had accumulated: Double DQN, **dueling networks** (decomposing Q into state-value $V(s)$ plus advantage $A(s,a)$, improving data efficiency when actions matter little), **prioritized experience replay** (sampling transitions with probability $\propto |\delta_t|^\alpha$ to concentrate updates where the current Q-function is most wrong), **multi-step returns**, **distributional Q-learning** (estimating the full return distribution rather than just its expectation), and **noisy networks** (learnable weight noise for exploration).

> [!success]
> **Rainbow** (Hessel et al., 2018) combined all six. The composite was superadditive: substantially outperforming any individual component and achieving far higher sample efficiency than standard DQN. Ablations identified prioritized replay and multi-step returns as the largest individual contributors. Rainbow effectively closed the book on the discrete-action, value-based deep RL program that DQN had opened.

---

## What DQN Left Unsolved

> [!warning]
> DQN's fundamental operation — $\max_{a'} Q(s', a')$ — requires enumerating all actions. For Atari's 18 discrete actions, this is trivial. For continuous control — a robot arm with six joint torques, a vehicle with continuous steering, a molecular design agent proposing conformational changes — the action space is a continuous manifold and exact maximization is intractable. Q-learning cannot be applied directly to continuous control.

The alternative approach was developed in parallel with DQN, drawing not from the value-based tradition but from a different branch of RL theory: **policy gradients**. Rather than learning a value function and deriving a policy from it, policy gradient methods directly parameterize and optimize the policy — making action selection smooth and differentiable, circumventing the argmax entirely. The mathematical foundation was laid by Williams in 1992 and formalized by Sutton et al. in 2000. That theory, and why it took until 2016 to produce algorithms competitive with DQN, is the subject of the next chapter.

---

*Q-learning convergence in deterministic environments is in Watkins and Dayan (1992), Q-learning. The stochastic approximation convergence proof is in Tsitsiklis (1994), Asynchronous Stochastic Approximation and Q-Learning. The ODE method is developed in Borkar (2008), Stochastic Approximation: A Dynamical Systems Viewpoint. The deadly triad is named and analyzed in Sutton and Barto (2018), chapter 11. Baird's counterexample is in Baird (1995), Residual Algorithms. The original DQN papers are Mnih et al. (2013) and Mnih et al. (2015), Human-Level Control through Deep Reinforcement Learning. Double DQN is in van Hasselt, Guez, and Silver (2015). Prioritized experience replay is in Schaul et al. (2015). The dueling architecture is in Wang et al. (2016). Rainbow is in Hessel et al. (2018).*
