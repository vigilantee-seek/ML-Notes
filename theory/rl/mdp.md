## Finite Markov Decision Processes

### Preface

A somewhat aggressive idea proposed by the book:

> In this chapter we introduce the problem that we try to solve in the rest of the book. For us, this problem deﬁnes the ﬁeld of reinforcement learning: any method that is suited to solving this problem we consider to be a reinforcement learning method.

> The general rule we follow is that anything that cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of its environment. ... In fact, in some cases the agent may know *everything* about how its environment works and still face a difficult reinforcement learning task, just as we may know exactly how a puzzle like Rubik’s cube works, but still be unable to solve it. The agent–environment boundary represents the limit of the agent’s *absolute control*, not of its knowledge.

Well, something beside the point. I noticed an interesting idea mentioned here that even if the agent knows everything about how its environment works, it may still face a difficult learning task. It's something philosophical, I think. You know all but you can't make it. It indicates the gap between methodology and epistimology to some degree. 

A model of learning goal-directed behavior:

**Three signals passing back and forth between an agent and its environment: one signal to represent the choices made by the agent (the actions), one signal to represent the basis on which the choices are made (the states), and one signal to deﬁne the agent’s goal (the rewards).**

Some tips:

* The reward signal is your way of communicating to the robot *what* you want it to achieve, not *how* you want it achieved.
* Rewards are computed in the environment rather than in the agent. 
* The agent's ultimate goal should be something over which it has imperfect control: it should not be able, for example, to simply decree that the reward has been received in the same way that it might arbitrarily change its actions. 

### Returns

The agents' goal is to maximize the cumulative reward it receives in the long run. Suppose the sequence of rewards received after time step $$t$$ is denoted $$R_{t+1}, R_{t+2}, R_{t+3},...$$, then the *expected return*, where the return $$G_t$$ is defined as some specific function of the reward sequence. 

#### Discrete Situation

In the simplest case the return is the sum of the rewards: 
$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T
$$
where $$T$$ is a final time step. 

This approach makes sense in applications in which there is a natural notion of ﬁnal time step, that is, when the agent–environment interaction breaks naturally into subsequences, which we call *episodes*, such as plays of a game, trips through a maze, or any sort of repeated interactions. Each episode ends in a special state called the *terminal state*, followed by a reset to a standard starting state or to a sample from a standard distribution of *starting states*. Tasks with episodes of this kind are called *episodic tasks*. 

#### Continual Situation

On the other hand, in many cases the agent–environment interaction does not break naturally into identiﬁable episodes, but goes on continually without limit. For example, this would be the natural way to formulate a continual process-control task, or an application to a robot with a long life span. We call these *continuing tasks*. 

The additional concept that we need is that of *discounting*. According to this approach, the agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized. In particular, it chooses At to maximize the expected *discounted return*:
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$
where \gamma is a parameter, $$0 \le \gamma \le 1$$, called the *discount rate*.

The discount rate determines the present value of future rewards: a reward received k time steps in the future is worth only $$\gamma^{k−1}$$ times what it would be worth if it were received immediately. If $$\gamma < 1$$, the inﬁnite sum has a ﬁnite value as long as the reward sequence $$\{R_k\}$$ is bounded. If $$\gamma = 0$$, the agent is “myopic” in being concerned only with maximizing immediate rewards: its objective in this case is to learn how to choose At so as to maximize only $$R_{t+1}$$. If each of the agent’s actions happened to inﬂuence only the immediate reward, not future rewards as well, then a myopic agent could maximize the equation by separately maximizing each immediate reward. But in general, acting to maximize immediate reward can reduce access to future rewards so that the return may actually be reduced. As $$\gamma$$ approaches 1, the objective takes future rewards into account more strongly: the agent becomes more farsighted.

### Markov Decision Processes

#### Definition

A reinforcement learning task that satisﬁes the Markov property is called a *Markov decision process*, or *MDP*. If the state and action spaces are finite, then it is called a *finite Markov decision process* (*ﬁnite MDP*).

A particular ﬁnite MDP is deﬁned by its state and action sets and by the one-step dynamics of the environment. Given any state and action $$s$$ and $$a$$, the probability of each possible pair of next state and reward, $$s'$$,$$r$$, is denoted
$$
p(s',r|s,a) = Pr\{S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a\}
$$ 

These quantities completely specify the dynamics of a ﬁnite MDP. 

With the dynamics, we can compute anything we want to know, such as the expected rewards for state-action pairs,
$$
r(s,a) = \mathbb{E} \left[ R_{t+1} | S_t=s, A_t=a \right] = \sum_{r\in \mathcal{R}}r \sum_{s' \in \mathcal{S}} p(s',r | s,a),
$$

the state-transition probabilities,
$$
p(s'|s,a) = Pr\{ S_{t+1}=s' | S_t=s, A_t=a \} = \sum_{r\in R} p(s',r| s,a),
$$

and the expected reweards for the state-action-next state triples,
$$
r(s,a,s') = \mathbb{E}_{\pi} \left[ R_{t+1} | S_t=s, A_t=s, S_{t+1}=s' \right] = \frac{\sum_{r\in \mathcal{R}} rp(s',r | s,a)}{p(s'|s,a)}
$$

### Value Functions

Recall that a policy, $$\pi$$, is a mapping from each state, $$s \in \mathcal{S}$$, and action, $$a \in A(s)$$, to the probability $$\pi(a|s)$$ of taking action a when in state $$s$$. For MDPs, we can define $$v_{\pi}(s)$$ (the value of a state $$s$$ under a policy $$\pi$$) formally as 
$$
v_{\pi}(s) = \mathbb{E}\left[ G_t | S_t=s \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s \right],
$$
where $$ \mathbb{E}_{\pi} [ \centerdot ] $$ denotes the expected value of a random variable given that the agent follows policy $$\pi$$, and $$t$$ is any time step. Note that the value of the terminal state, if any, is always zero. We call the function $$v_\pi$$ the *state-value function for policy* $$\pi$$.

Similarly, we deﬁne the value of taking action $$a$$ in state $$s$$ under a policy $$\pi$$, denoted $$q_{\pi}(s,a)$$, as the expected return starting from $$s$$, taking the action $$a$$, and thereafter following policy $$\pi$$: 
$$
q_{\pi}(s,a) = \mathbb{E}_{\pi} \left[ G_t | S_t=s, A_t=a \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s, A_t=a \right]
$$
We call $$q_{\pi}$$ the *action-value function for poliy* $$\pi$$.

For any policy $$\pi$$ and any state $$s$$, the following consistency condition holds between the value of $$s$$ and the value of its possible successor states: 
$$
\begin{align*}
v_{\pi}(s) &= \mathbb{E}_{\pi} \left[ G_t | S_t=s \right] \\
           &= \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s \right]  \\
           &= \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | S_t=s \right],
\end{align*}
$$
here, because of the Markov property
$$
\mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | S_{t+1}=s', S_t=s \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | S_{t+1}=s' \right]        \\
\Longrightarrow \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | S_t=s \right] = \sum_{s'} \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | S_{t+1}=s' \right] p(s'|s),
$$
and we know
$$
r(s,a) = \mathbb{E} \left[ R_{t+1} | S_t=s, A_t=a \right] = \sum_{r\in \mathcal{R}}r \sum_{s' \in \mathcal{S}} p(s',r | s,a)    \\
p(s'|s,a) = Pr\{ S_{t+1}=s' | S_t=s, A_t=a \} = \sum_{r\in \mathcal{R}} p(s',r| s,a)  \\
$$
Therefore,
$$
\begin{align*}
v_{\pi}(s) &= \sum_a \pi(a|s) \sum_{s'} \sum_r p(s',r|s,a) \left[ r + \gamma                    \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | S_{t+1}             =s' \right] \right]  \\
           &= \sum_a \pi(a|s) \sum_{s'} \sum_r p(s',r|s,a) \left[ r + \gamma v_{\pi}(s') \right],
\end{align*} 
$$

This is the *Bellman Equation for* $$v_{\pi}$$. It expresses a relationship between the value of a state and the values of its successor states. It averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way. 

### Optimal Value Function

For finite MDPs, we can precisely define an optimal policy in the following way. Value functions define a partial ordering over policies. A policy $$\pi$$ is deﬁned to be better than or equal to a policy $$\pi'$$ if its expected return is greater than or equal to that of π0 for all states. In other words, $$\pi \ge \pi'$$ if and only if $$v_{\pi}(s) \ge v_{\pi'}(s)$$ for all $$s \in \mathcal{S}$$. There is always at least one policy that is better than or equal to all other policies. This is an *optimal policy*. Although there may be more than one, we denote all the optimal policies by $$\pi_*$$. They share the same state-value function, called the *optimal state-value function*, denoted $$v_*$$, and defined as 
$$
v_*(s) = \max_{\pi} v_{\pi}(s),
$$
for all $$s \in \mathcal{S}$$.

Optimal policies also share the same optimal action-value function, denoted q∗, and defined as 
$$
q_*(s,a) = \max_{\pi} q_{\pi}(s,a),
$$

**Noted**: Here $$v_*(s)$$ DOES NOT have the exactly same meaning as the $$v_{\pi}(s)$$ we mentioned above. Precisely speaking,
$$
\begin{align*}
v_*(s) &= \max_{a\in \mathcal{A}(s)} q_{\pi_*}(s,a)      \\
       &= \max_a \mathbb{E}_{\pi_*} \left[ G_t | S_t=s, A_t=a \right]    \\
       &= \max_a \mathbb{E}_{\pi_*} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s, A_t=a \right]    \\
       &= \max_a \mathbb{E}_{\pi_*} \left[ R_{t+1} + \gamma \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | S_t=s, A_t=a \right]        \\
       &= \max_a \mathbb{E} \left[ R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s, A_t=a \right]        \\
       &= \max_{a\in \mathcal{A}(s)} \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_*(s') \right]
\end{align*}
$$
Similarly,
$$
\begin{align*}
q_*(s,a) &= \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} q_*(S_{t+1},a') | S_t=s,              A_t=a \right]   \\
         &= \sum_{s',r} p(s',r|s,a) \left[ r + \gamma \max_{a'} q_*(s',a') \right]
\end{align*}
$$

### Optimality and Approximation

Even if the agent has a complete and accurate environment model, the agent is typically unable to perform enough computation per time step to fully use it. The memory available is also an important constraint. Memory may be required to build up accurate approximations of value functions, policies, and models. In most cases of practical interest there are far more states than could possibly be entries in a table, and approximations must be made.

