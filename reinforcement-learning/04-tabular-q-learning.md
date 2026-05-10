---

title: "Tabular Q-Learning and Convergence"
subtitle: "The Q-learning update, proof of convergence, and the tabular regime's limits"
---


Q-learning ended the previous chapter with a clean update rule and an informal guarantee: it performs stochastic value iteration, converging to the optimal action-value function without ever requiring the transition kernel. The purpose of this chapter is to make that guarantee precise. The convergence of Q-learning is not obvious. The bootstrap target $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$ depends on the very estimates being updated, creating a circular dependency that could in principle cause the estimates to spiral away from $Q^*$ rather than toward it. That this does not happen in the tabular setting — and the identification of exactly which condition breaks when a neural network is substituted for the lookup table — are the central results of this chapter.

## The Convergence Theorem

The formal convergence result was established by Watkins and Dayan (1992) for deterministic environments and extended to stochastic MDPs by Tsitsiklis (1994) using the theory of stochastic approximation. The theorem reads:

**Theorem.** Let $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ be a finite MDP with bounded rewards $|r| \leq R_{\max}$. Starting from any bounded $Q_0$, let $Q_t$ denote the Q-learning iterates. Suppose:
1. Every state-action pair $(s, a)$ is visited infinitely often.
2. The step sizes satisfy $\sum_t \alpha_t(s, a) = \infty$ and $\sum_t \alpha_t(s, a)^2 < \infty$ for all $(s, a)$.

Then $Q_t(s, a) \to Q^*(s, a)$ with probability 1 for all $(s, a)$.

Three things are worth noting immediately. First, the theorem makes no assumptions about the behavior policy beyond the visitation condition: Q-learning is truly off-policy. Second, the step size conditions are the same Robbins-Monro conditions that appear in every stochastic approximation result — they are not specific to RL. Third, the theorem is existential: it guarantees convergence in the limit but says nothing about how fast, or how many samples are required. Both of these caveats have significant practical consequences.

## Q-Learning as Stochastic Approximation

The proof rests on placing Q-learning within the general stochastic approximation framework. The recursion

$$x_{t+1}(i) = x_t(i) + \alpha_t(i) \bigl[F_t(x_t, i) + \varepsilon_t(i)\bigr]$$

converges to the fixed point of $F$ — the root of $x = x + F(x)$ — when $F$ is a contraction and $\varepsilon_t$ is zero-mean noise satisfying appropriate moment conditions. This is the abstract template; the problem is to verify that Q-learning fits it.

Write the Q-learning update at $(S_t, A_t)$ as:

$$Q_{t+1}(S_t, A_t) = Q_t(S_t, A_t) + \alpha_t(S_t, A_t)\Bigl[\underbrace{R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - Q_t(S_t, A_t)}_{\text{observed increment}}\Bigr].$$

Decompose the observed increment into its expectation plus noise:

$$R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - Q_t(S_t, A_t) = \underbrace{\bigl[(\mathcal{T}^* Q_t)(S_t, A_t) - Q_t(S_t, A_t)\bigr]}_{F(Q_t, S_t, A_t)} + \underbrace{\varepsilon_t}_{\text{noise}},$$

where $\varepsilon_t = R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a') - (\mathcal{T}^* Q_t)(S_t, A_t)$ is the deviation of the sample from its expectation. This noise is zero-mean given $Q_t$ and $(S_t, A_t)$ by the definition of $\mathcal{T}^*$, and its variance is bounded by the reward variance plus the range of Q-values, both of which are finite when the rewards are bounded.

The noise-free update direction is $F(Q, s, a) = (\mathcal{T}^* Q)(s, a) - Q(s, a)$: the displacement from the current estimate toward its Bellman target. The unique root of $F$ is $Q^*$, the fixed point of $\mathcal{T}^*$, since $(\mathcal{T}^* Q^*)(s, a) - Q^*(s, a) = 0$ everywhere.

## Why the Contraction Closes the Argument

Stochastic approximation converges when the noise-free dynamics are stable — when the system $\dot{x} = F(x)$ is globally attracted to the fixed point, so that the noise, averaged over many steps, cannot push the iterates far from $Q^*$. For Q-learning, global stability follows directly from the contraction property established in Chapter 01.

For any two Q-functions $Q$ and $Q'$:

$$\|\mathcal{T}^* Q - \mathcal{T}^* Q'\|_\infty \leq \gamma \|Q - Q'\|_\infty.$$

This means that the Bellman error $Q - Q^*$ shrinks geometrically under the noise-free dynamics: if the current estimates are at distance $d$ from $Q^*$ in the sup-norm, the update $F(Q, s, a) = (\mathcal{T}^* Q)(s, a) - Q(s, a)$ points toward $Q^*$ with a force proportional to the error, and after one full sweep the error is at most $\gamma d$. The stochastic iterates track this dynamics with additive noise; the diminishing step sizes ensure the noise is eventually overwhelmed.

Tsitsiklis' proof formalizes this via the **ODE method**: under the Robbins-Monro conditions, the stochastic iterates concentrate on the trajectories of the ODE $\dot{x} = F(x)$ as time progresses. The ODE has a globally asymptotically stable equilibrium at $Q^*$ — a Lyapunov function is $V(Q) = \|Q - Q^*\|_\infty$, which satisfies $\dot{V} \leq -c V$ for some $c > 0$ along trajectories — so the stochastic iterates converge to the same equilibrium.

## The Exploration Requirement

Condition 1 of the theorem — every state-action pair visited infinitely often — is an asymptotic requirement with concrete practical consequences.

A purely greedy policy fails immediately: if the current Q-estimates make one action appear optimal at every state, the agent may never explore alternatives. State-action pairs with poor initial estimates are never updated and remain poor. The Q-values for those pairs converge to whatever wrong value they started at, and the algorithm produces a suboptimal policy rather than $Q^*$.

The standard fix is an **$\varepsilon$-greedy** behavior policy with a decaying exploration schedule. For convergence, $\varepsilon_t$ must satisfy two conditions simultaneously:

$$\varepsilon_t \to 0 \quad \text{and} \quad \sum_t \varepsilon_t = \infty.$$

The first ensures the policy converges to greedy — otherwise Q-learning converges to the optimal $\varepsilon$-greedy Q-function, not $Q^*$. The second ensures infinitely many exploratory actions occur, so every pair is visited infinitely often. Policies satisfying both conditions are called **GLIE** (Greedy in the Limit with Infinite Exploration). The simplest GLIE schedule is $\varepsilon_t = c/t$ for any $c > 0$.

In practice, constant or linearly decaying $\varepsilon$ is far more common. With constant $\varepsilon$, the asymptotic convergence guarantee no longer holds exactly — the iterates converge to a neighborhood of $Q^*$ rather than $Q^*$ itself — but the neighborhood can be made small by choosing $\varepsilon$ small, and the estimates remain stable. This is often preferable to the slow decay required for exact GLIE convergence, which ensures exploration at the cost of very slow convergence in later training.

## What the Tabular Representation Makes Possible

A detail of the convergence proof that is easy to overlook is the role of the lookup table itself. In the tabular setting, $Q(s, a)$ is an independent real number for each $(s, a)$, stored in a separate memory cell. An update to $Q(S_t, A_t)$ changes exactly one cell and leaves all others unchanged.

This independence is the technical condition that makes the stochastic approximation argument go through. The update to $Q(S_t, A_t)$ is an unbiased estimate of $(\mathcal{T}^* Q)(S_t, A_t) - Q(S_t, A_t)$ with zero-mean noise. Averaging over many transitions from $(S_t, A_t)$ — while all other entries of the Q-table remain fixed — recovers the expected Bellman backup for that entry exactly. The law of large numbers applies entry-by-entry.

The moment we introduce a parameterized function approximator — even a linear one — this independence is destroyed. If $Q(s, a; \theta) = \theta^\top \phi(s, a)$ shares parameters $\theta$ across all state-action pairs, an update based on the transition at $(S_t, A_t)$ changes the Q-value for every pair simultaneously, by an amount proportional to $\phi(s, a)^\top \phi(S_t, A_t)$. An update designed to reduce the Bellman error at $(S_t, A_t)$ can increase it at other pairs. The per-entry convergence argument breaks completely.

## The Deadly Triad

Sutton introduced the term **deadly triad** to describe three ingredients that, combined, produce reinforcement learning algorithms that can diverge:

1. **Function approximation**: representing $Q$ with a parameterized function rather than a table.
2. **Bootstrapping**: using $Q$-estimates to generate targets for updating $Q$, as in all TD methods.
3. **Off-policy learning**: learning about a target policy (greedy) from data generated by a different behavior policy.

The striking fact is that each pair of the three is safe:

*Function approximation without bootstrapping and without off-policy*: supervised regression on targets computed from full Monte Carlo returns. The targets do not depend on the model being fit, so the regression is stable. This is Monte Carlo policy evaluation with neural networks, and it converges.

*Bootstrapping without function approximation and without off-policy*: tabular TD(0) or SARSA. Convergence is guaranteed by the analysis above. The independence of table entries ensures the contraction argument holds.

*Off-policy without function approximation and without bootstrapping*: off-policy Monte Carlo evaluation. Compute full returns under the behavior policy, apply importance sampling corrections, and average. Statistically sound and convergent.

All three together: no guarantee. In fact, simple examples demonstrate divergence — Q-estimates that grow without bound even when $Q^*$ is well-defined and representable by the function approximator.

## Baird's Counterexample

In 1995, Leemon Baird exhibited the simplest known instance of deadly-triad instability. The MDP has seven states and two actions. One action, called the *dashed* action, transitions each state to a single absorbing state with probability 1. The other, the *solid* action, transitions each state to any of the seven states uniformly at random. All rewards are zero. The optimal Q-function is zero everywhere, and this is representable by a linear function approximator with the feature vectors Baird constructed.

Despite this, semi-gradient Q-learning applied to this MDP with linear function approximation causes the parameter vector to grow without bound. The Q-estimates diverge to infinity even though $Q^* = 0$. The algorithm is chasing a moving target — each update changes the bootstrap target $\gamma \max_{a'} Q(S_{t+1}, a'; \theta)$ as well as the estimate being updated — and in this MDP the updates amplify rather than dampen each other.

The mechanism is a norm mismatch. The sup-norm contraction guarantees that $\mathcal{T}^*$ shrinks the worst-case error. But semi-gradient descent minimizes the **projected Bellman error**: the squared Bellman error weighted by the distribution of states visited by the behavior policy. When the behavior policy visits some states rarely, those states receive little weight, and their Bellman errors can grow without penalty to the objective. If the bootstrap targets for frequently visited states depend on the Q-values of rarely visited states — as in Baird's MDP, where the solid action scatters the agent widely — the update can increase unweighted Bellman errors even as it decreases the weighted ones.

Baird's example is not a pathological edge case. It is a demonstration of the fundamental conflict between the sup-norm geometry of the Bellman optimality operator and the $\ell_2$ geometry of gradient-based learning. This conflict is unavoidable whenever function approximation, bootstrapping, and off-policy data co-occur — which is precisely the combination that practical deep RL algorithms require.

## Credit Assignment and the Horizon Problem

Beyond the theoretical stability issues, tabular Q-learning faces a practical challenge that is independent of function approximation: in environments with sparse rewards over long horizons, credit propagates backward painfully slowly.

Consider an agent navigating a maze where reward is received only upon reaching the goal, $H$ steps from the start. At initialization, all Q-values are zero. The first time the agent reaches the goal, only the Q-value of the final step is updated positively. On subsequent episodes, value information propagates backward one step per episode: the second-to-last step receives a positive update only after the agent has taken the same path twice, the third-to-last step after three traversals, and so on. Reliable Q-value estimates near the start require $\Omega(H)$ successful episodes, and reliable estimates at every state require many more.

TD($\lambda$) with eligibility traces accelerates this propagation: a positive reward signal at the goal updates all recently visited states simultaneously, with weight decaying geometrically at rate $\lambda\gamma$ per step backward. But for $\lambda < 1$, states more than $O(1/(1-\lambda))$ steps in the past receive negligible updates, and the horizon problem persists for tasks longer than this effective window. For $\lambda = 1$, the update covers the full episode but returns to Monte Carlo variance.

The deep RL era addressed the horizon problem partly through architecture — convolutional networks that extract useful representations from pixel inputs — and partly through clever reward shaping and multi-step returns. But the underlying difficulty is intrinsic to the sequential decision problem: the Bellman equations propagate value information one step at a time, and efficient long-horizon credit assignment requires either many trials or mechanisms that explicitly span multiple steps. This remains an open area of research.

## The Floor of the Tabular Era

By 1995, the tabular convergence theory was complete. Q-learning, SARSA, and TD(0) all converged to their respective targets under well-specified conditions. The conditions themselves — visitation, step size schedules, finite state and action spaces — precisely characterized what was required for the theoretical guarantee to hold.

The same analysis identified the floor: the guarantee stopped exactly at the boundary of the tabular representation. Function approximation, required for any real-world problem of interest, introduced instabilities that the tabular theory could neither predict nor fix. Baird's counterexample showed that the instabilities were not artifacts of poor implementation but fundamental properties of the interaction between gradient-based learning and bootstrapped targets.

Two questions remained. The first was empirical: could neural Q-learning be made to work in practice, despite the lack of a convergence guarantee? The second was theoretical: what weaker conditions would be sufficient to make neural Q-learning stable, if not convergent? The first question was answered in 2013. The second remains partly open.

The answer to the first question was the Deep Q-Network: a combination of Q-learning with neural function approximation, stabilized by two engineering innovations that partially addressed the deadly triad without eliminating any of its three components. Understanding why those innovations worked — and what they fundamentally did to the learning dynamics — requires understanding the architecture they operated on, which is the subject of the next chapter.

---

*Q-learning convergence in deterministic environments is in Watkins and Dayan (1992), Q-learning. The stochastic approximation convergence proof for the stochastic case is in Tsitsiklis (1994), Asynchronous Stochastic Approximation and Q-Learning. The ODE method for stochastic approximation is developed systematically in Borkar (2008), Stochastic Approximation: A Dynamical Systems Viewpoint. The deadly triad is named and analyzed in Sutton and Barto (2018), Reinforcement Learning: An Introduction, chapter 11. Baird's counterexample is in Baird (1995), Residual Algorithms: Reinforcement Learning with Function Approximation. GLIE policies and their role in convergence are analyzed in Singh et al. (2000), Convergence Results for Single-Step On-Policy Reinforcement-Learning Algorithms.*
