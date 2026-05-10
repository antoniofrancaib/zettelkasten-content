---

title: "Deep Q-Networks"
subtitle: "Experience replay, target networks, the Atari result, and the function approximation pathology"
---


For nearly two decades after Watkins introduced Q-learning, the gap between the algorithm's theoretical elegance and its practical reach remained stubbornly wide. The convergence guarantee was clean and complete — but it required a lookup table, and a lookup table required a finite, enumerable state space. Researchers attempted to bridge this gap by substituting neural networks for the table, but the results were erratic. Neural Q-learning trained, then diverged. It learned part of a task, then forgot what it had learned. It worked on some seeds and catastrophically failed on others. The deadly triad was not a theoretical curiosity. It manifested as practical instability that made neural Q-learning unreliable for anything but toy problems.

In December 2013, a team at DeepMind posted a preprint titled "Playing Atari with Deep Reinforcement Learning." The paper introduced a single algorithm — the **Deep Q-Network**, or DQN — that learned to play seven Atari 2600 games from raw pixel inputs, achieving performance exceeding human experts on three of them. The same architecture, the same hyperparameters, the same reward structure across all games. The result was not just an engineering success: it demonstrated that the combination of deep neural networks and reinforcement learning could, with the right stabilizing machinery, solve the problems that had defeated the field for twenty years. By the time the full study was published in *Nature* in 2015 — covering 49 games — it had catalyzed a new era of research.

The stabilizing machinery was two ideas. Neither was theoretically sufficient to eliminate the deadly triad. Both were empirically sufficient to suppress it in the specific setting of Atari games, and their logic is general enough to inform every deep value-based method that followed.

## The Architecture

The raw input to DQN is a sequence of frames from the Atari emulator: $84 \times 84$ grayscale images, preprocessed from the original $210 \times 160$ RGB output. Four consecutive frames are stacked into a single state representation, yielding a $4 \times 84 \times 84$ input tensor. Stacking frames is the simplest solution to a Markov property problem: a single frame does not reveal velocity — you cannot tell from one image which direction the ball is moving — and without velocity, the state is not Markovian. Four frames capture enough motion history that the Markov assumption is approximately satisfied.

The network processes this input through three convolutional layers with ReLU activations, followed by two fully connected layers. The final output layer has one unit per action, producing the estimated Q-value $Q(s, a; \theta)$ for all actions simultaneously in a single forward pass. For Atari games, the action space has up to 18 discrete actions. The architecture contains roughly 1.7 million parameters.

The convolutional structure is essential, not incidental. Convolutional filters share weights across spatial positions, allowing a feature detector learned in one part of the screen to be applied everywhere. A filter that detects a ball in the upper-left quadrant automatically detects the same ball in the lower-right quadrant. This **weight sharing** provides the generalization across observations that a lookup table cannot: similar game states, which differ in the position of objects rather than their identity, receive similar Q-value estimates. It is the spatial analogue of the feature-based generalization that was missing from the tabular setting.

## The First Stabilizer: Experience Replay

The most obvious symptom of the deadly triad in neural Q-learning was the correlation between consecutive transitions. When an agent moves through a game environment, consecutive frames are nearly identical: the game state at time $t+1$ differs from the state at time $t$ by a single step of physics. Gradient updates computed on such correlated samples are nearly identical to each other, which causes the network to overfit to the current region of the game while forgetting earlier ones. This is the phenomenon of **catastrophic forgetting**: new learning overwrites old learning because the gradient signal is dominated by recent experience.

The solution was **experience replay**, borrowed from an earlier proposal by Lin (1992). At each step, the agent stores the transition $(S_t, A_t, R_{t+1}, S_{t+1})$ in a fixed-size **replay buffer** $\mathcal{D}$, discarding the oldest transitions as the buffer fills. Rather than updating the network immediately on the current transition, the agent samples a random minibatch of transitions from $\mathcal{D}$ and computes the gradient update on that minibatch.

The benefits are several. First, by sampling uniformly at random from the buffer, consecutive updates are drawn from different time points and different regions of the game — the correlation between successive gradient updates is broken, and the optimizer sees something approximating i.i.d. data. Second, each transition is potentially used for many updates before being discarded, improving data efficiency: the agent extracts more gradient signal per environment interaction. Third, the buffer contains transitions from many different policies — the agent's past selves — which prevents the training distribution from collapsing to the current policy's experience.

This last point is worth dwelling on. Experience replay makes DQN a more deeply off-policy algorithm than tabular Q-learning: not only does the greedy target policy differ from the $\varepsilon$-greedy behavior policy, but the data comes from a historical mixture of many past behavior policies. The tabular convergence proof required visiting each state-action pair infinitely often, and off-policy data was acceptable as long as the visitation condition held. With experience replay, this is roughly satisfied for states that have been visited recently enough to remain in the buffer, but not for states visited far in the past whose transitions were overwritten. In practice, the buffer covers enough of the relevant state space to make training stable.

## The Second Stabilizer: Target Networks

The deeper instability in neural Q-learning is the **moving target problem**. In supervised learning, the targets for training are fixed labels — they do not change as the model is updated. In Q-learning, the target for training $Q(S_t, A_t; \theta)$ is $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta)$ — a quantity that depends on the same parameters $\theta$ being optimized. Every gradient step changes both the predictions and the targets simultaneously. The network is, in a concrete sense, chasing its own tail.

The pathological consequence: a gradient step that reduces the Bellman error at $(S_t, A_t)$ also shifts the bootstrap target for all nearby state-action pairs. This perturbation can amplify rather than dampen: the target moves in the direction of the update, the next update chases the moved target, and the iterates spiral outward. This is the mechanism of Baird's counterexample in the neural setting.

DQN addresses this with **target networks**: a separate copy of the Q-network with parameters $\theta^-$, identical in architecture to the online network but updated far less frequently. The training target becomes:

$$y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a';\, \theta^-).$$

The online network parameters $\theta$ are updated by gradient descent at every step to minimize $\mathbb{E}[(y_t - Q(S_t, A_t; \theta))^2]$. The target network parameters $\theta^-$ are held fixed for $C$ steps, then replaced by a copy of $\theta$. Mnih et al. used $C = 10{,}000$.

With a frozen target network, the training objective is locally a supervised regression: fixed targets, gradient descent. The target does not respond to gradient steps on $\theta$, breaking the feedback loop that caused the moving-target instability. After $C$ steps, the target network is updated to the current online network, and the process repeats. The instability is not eliminated but is interrupted every $C$ steps, and if $C$ is large enough relative to the learning rate, the online network converges substantially toward the current target before the target moves again.

A useful analogy: tabular value iteration updates all Q-values in a single full sweep, then re-evaluates the max over the updated table. Target networks approximate this structure in the neural setting — the target network plays the role of the previous iteration's table, the online network updates toward it, and the periodic refresh plays the role of the next sweep.

## The Loss Function and Training

The DQN training loss is the mean squared Bellman error over a minibatch $\mathcal{B}$ sampled from the replay buffer:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{B}}\!\Bigl[\bigl(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\bigr)^2\Bigr].$$

The gradient with respect to $\theta$ is:

$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{\mathcal{B}}\!\Bigl[-2\bigl(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\bigr)\,\nabla_\theta Q(s, a; \theta)\Bigr].$$

This is a **semi-gradient** update: the gradient flows through $Q(s, a; \theta)$ but is stopped at the target $Q(s', a'; \theta^-)$, which is treated as a constant. Taking the full gradient — differentiating through the target as well — would correspond to a different, less stable objective. The semi-gradient approximation is standard in all practical deep RL and is one of the implicit conditions that makes the training tractable.

The outer loop of DQN is simple: at each step, select an action $\varepsilon$-greedily, execute it in the environment, store the transition, sample a minibatch, and take one gradient step. The inner mechanics — replay buffer, target network refresh, preprocessing — are engineering. The outer loop is Q-learning with a neural approximator.

## The Atari Benchmark

The choice of Atari 2600 games as a benchmark was itself a methodological contribution. Previous deep RL results had been on individually engineered environments: one paper, one game, custom preprocessing, custom hyperparameters. The DQN paper ran the same algorithm — same architecture, same learning rate, same replay buffer size, same $\varepsilon$ schedule, same $\gamma$, same $C$ — on 49 different games. Each game had different visual styles, different reward structures, different optimal strategies, and different temporal dependencies.

On 29 of the 49 games, DQN surpassed human expert performance. On games requiring precise timing and spatial awareness — Breakout, Pong, Space Invaders — the performance was dramatically superhuman. On games requiring long-horizon planning — Montezuma's Revenge, Pitfall — DQN performed poorly, often scoring near zero despite millions of training steps. This pattern was informative: DQN's strengths lay in reactive strategies that required discriminating between visually distinct states, and its weaknesses lay in tasks demanding exploration of a large state space before encountering a single reward.

The benchmark established a standard that the entire subsequent field adopted. When a new algorithm claimed to improve on DQN, it was evaluated on the same 49 games with the same performance metric: percentage of human-level performance, where human performance was measured by a professional game tester. This normalization made results comparable across papers in a way that had not previously been possible in RL research.

## Overestimation and the Q-value Pathology

DQN worked. But the function approximation pathology predicted by the theoretical analysis did not disappear — it manifested in a different form.

The Q-learning target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ contains a maximization over a set of estimated values. Each Q-value estimate $Q(s', a'; \theta^-)$ contains noise — approximation error from the neural network. The maximum of a set of noisy estimates is systematically larger than the true maximum:

$$\mathbb{E}\bigl[\max_{a'} Q(s', a'; \theta^-)\bigr] \geq \max_{a'} Q^*(s', a'),$$

with equality only when there is no noise. This is an elementary consequence of the convexity of the max operator: $\max_{a'} \mathbb{E}[X_a] \leq \mathbb{E}[\max_{a'} X_a]$. In practice, the overestimation can be substantial: if ten actions each have Q-values with modest independent noise, the maximum is dominated by whichever has the largest positive noise, regardless of which action is truly optimal.

Overestimation causes the bootstrap targets to be too large, which propagates inflated values backward through the Bellman updates. The result is a systematic positive bias in learned Q-values that becomes more severe as training progresses, since each update uses overestimated targets to generate the next round of overestimated targets.

**Double DQN** (van Hasselt et al., 2015) addresses this by decoupling the action selection from the action evaluation in the bootstrap target. Instead of selecting and evaluating the best next action with the same network:

$$y_t^{\text{DQN}} = r + \gamma Q\!\bigl(s',\, \arg\max_{a'} Q(s', a';\, \theta^-);\, \theta^-\bigr),$$

Double DQN selects the best next action with the online network and evaluates it with the target network:

$$y_t^{\text{DDQN}} = r + \gamma Q\!\bigl(s',\, \arg\max_{a'} Q(s', a';\, \theta);\, \theta^-\bigr).$$

The online and target networks have different parameters and therefore different noise patterns. The action selected by one is evaluated by the other; if both networks are noisy in the same direction for some action, the double Q correction still reduces overestimation on average. The modification requires one additional forward pass per update and consistently improves performance across the Atari benchmark, particularly on games where DQN's overestimation was most severe.

## The Dueling Architecture

A separate architectural insight, introduced by Wang et al. (2016), addresses the representation rather than the target. For many states in a video game, most actions produce very similar outcomes — moving slightly left versus slightly right when the ball is far away barely matters. The Q-values of different actions differ by small amounts that are difficult to estimate accurately from a single shared representation.

The **dueling network** decomposes the Q-function into two components with separate network streams:

$$Q(s, a; \theta) = V(s;\, \theta_V) + A(s, a;\, \theta_A) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a';\, \theta_A).$$

Here $V(s)$ is the state value (how good is this state regardless of action) and $A(s, a)$ is the advantage (how much better is action $a$ compared to the average action). The subtracted mean ensures the decomposition is unique: setting the average advantage to zero makes $V(s)$ the true state value under the current policy.

The payoff is data efficiency in states where actions matter little. When $V(s)$ can be estimated from the aggregate stream without needing to distinguish between actions, gradients from many transitions — where the action taken was irrelevant — flow directly into improving $V$. In states where the action-advantage structure matters, the advantage stream sharpens. The architecture learns which aspects of the Q-function require detailed estimation and which do not, implicitly allocating network capacity.

## Prioritized Experience Replay

The third major improvement to DQN addressed the replay buffer. Uniform sampling from the buffer treats all stored transitions as equally informative. In practice, transitions with large TD errors are more informative — they are the transitions where the current Q-function is most wrong, and where gradient updates would have the most effect.

**Prioritized experience replay** (Schaul et al., 2015) assigns each transition a sampling priority proportional to its TD error magnitude $|\delta_t|$, raised to a power $\alpha \in (0, 1)$ to prevent extremely high-error transitions from monopolizing the buffer. The sampling probability for transition $i$ is:

$$P(i) = \frac{|\delta_i|^\alpha}{\sum_j |\delta_j|^\alpha}.$$

Since high-priority transitions are sampled more frequently than their proportion in the buffer warrants, the gradient updates are no longer unbiased estimates of the full-buffer gradient. This is corrected with **importance sampling weights** $w_i = (N \cdot P(i))^{-\beta}$ applied to each gradient, where $\beta$ is annealed from 0 to 1 over the course of training to restore unbiasedness in the limit.

Prioritized replay consistently accelerates learning on Atari, particularly in early training when the Q-function is far from $Q^*$ and high-error transitions are the most informative. The intuition is the same as prioritized sweeping from Chapter 02: concentrating updates on the most informative states is more efficient than uniform sampling, whether in the tabular or neural setting.

## Rainbow: The Composite

By 2018, a half-dozen improvements to the original DQN had accumulated: Double DQN, dueling networks, prioritized replay, multi-step returns, distributional Q-learning (which estimates the full return distribution rather than just its expectation), and noisy networks (which add learnable noise to network weights as an exploration mechanism). Each had been evaluated in isolation with consistent improvements.

Hessel et al. (2018) combined all six in a single agent — **Rainbow** — and found that the combination was superadditive: the composite agent substantially outperformed any individual component and achieved far higher sample efficiency than standard DQN. The paper also performed ablations: each component was individually removed from Rainbow, and the resulting performance drop identified which components were most critical. Prioritized replay and multi-step returns provided the largest individual contributions; removing either significantly degraded performance.

Rainbow essentially closed the book on the discrete-action, value-based deep RL program that DQN had opened. Further gains on Atari came from larger networks, longer training, and improved exploration — engineering rather than algorithmic novelty.

## What DQN Got Right, and What It Left Unsolved

DQN made neural Q-learning practical by addressing the two most acute manifestations of the deadly triad: the correlation problem (experience replay) and the moving-target problem (target networks). It demonstrated that deep function approximation could generalize across the enormous visual diversity of 49 different games with identical hyperparameters — a form of transfer that no tabular method could provide.

What it did not address was the structural asymmetry between discrete and continuous action spaces. Atari games have at most 18 discrete actions; the $\max_{a'} Q(s', a')$ operator can be evaluated exactly by a forward pass over all actions. In continuous control — a robot arm with six joint torques, a vehicle with continuous steering and throttle, a molecular design agent proposing continuous conformational changes — the action space is a continuous manifold, and exact maximization is intractable. Q-learning's fundamental operation, maximizing over actions, is the reason it cannot be applied directly to continuous control.

The alternative approach to this problem was developed in parallel with DQN, drawing not from the value-based tradition but from a different branch of the RL theory: policy gradients. Rather than learning a value function and deriving a policy from it, policy gradient methods directly parameterize and optimize the policy — making the action selection smooth and differentiable, and circumventing the argmax entirely. The mathematical foundation for this approach was laid by Williams in 1992 and formalized by Sutton et al. in 2000. Why it took until 2016 to produce algorithms competitive with DQN — and what was required to get there — is the story of the next part of the arc.

---

*The original DQN paper is Mnih et al. (2013), Playing Atari with Deep Reinforcement Learning, with the expanded Nature version in Mnih et al. (2015), Human-Level Control through Deep Reinforcement Learning. Experience replay was proposed by Lin (1992), Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching. Double DQN is in van Hasselt, Guez, and Silver (2015), Deep Reinforcement Learning with Double Q-Learning. Prioritized experience replay is in Schaul et al. (2015), Prioritized Experience Replay. The dueling network architecture is in Wang et al. (2016), Dueling Network Architectures for Deep Reinforcement Learning. The Rainbow composite is in Hessel et al. (2018), Rainbow: Combining Improvements in Deep Reinforcement Learning. The overestimation bias in Q-learning was analyzed theoretically by van Hasselt (2010), Double Q-Learning.*
