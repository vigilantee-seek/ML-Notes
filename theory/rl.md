## Reinforcement Learning

This chapter is mainly organised according to *Reinforcement Learning: An Introduction* by **Richard S. Sutton and Andrew G. Barto**.

### Basic Ideas

**Conception/Definition**: To capture the most important aspects of the real problem facing a learning agent interacting with its environment to achieve a goal.

**Characteristics**: These three characteristics—being closed-loop in an essential way, not having direct instructions as to what actions to take, and where the consequences of actions, including reward signals, play out over extended time periods—are the three most important distinguishing features of reinforcement learning problems.

**Differences**: According to the book:

> Supervised learning is learning from a training set of labeled examples provided by a knowledgable external supervisor. Each example is a description of a situation together with a speciﬁcation—the label—of the correct action the system should take to that situation, which is often to identify a category to which the situation belongs. The object of this kind of learning is for the system to extrapolate, or generalize, its responses so that it acts correctly in situations not present in the training set. This is an important kind of learning, but alone it is not adequate for learning from interaction. In interactive problems it is often impractical to obtain examples of desired behavior that are both correct and representative of all the situations in which the agent has to act.

To summarize, I think, the major discrimination lies on the environment. Supervised learning is for static environmental conditions while a reinforcement learning method is more suitable to handle dynamic scenes. Popularly speaking, supervised learning is to rear livestock in pens, by contrast, reinforcement learning is to leave livestock in wild by themselves. 

As for unsupervised learning:

> Although one might be tempted to think of reinforcement learning as a kind of unsupervised learning because it does not rely on examples of correct behavior, reinforcement learning is trying to maximize a reward signal instead of trying to ﬁnd hidden structure. Uncovering structure in an agent’s experience can certainly be useful in reinforcement learning, but by itself does not address the reinforcement learning agent’s problem of maximizing a reward signal.

Another key feature: 

> Another key feature of reinforcement learning is that it explicitly considers the whole problem of a goal-directed agent interacting with an uncertain environment. This is in contrast with many approaches that consider subproblems without addressing how they might ﬁt into a larger picture. 

In fact, the gradient descent algorithm applied in the widespread back propagation network is a typical case of the local and static optimization of the current supervised learning. Although it have yielded many useful results, the focus on isolated subproblems itself is a significant limitation.

**Elements**: Beyond the agent and the environment, one can identify four main subelements of a reinforcement learning system: *a policy, a reward signal, a value function, and, optionally, a model of the environment*.

- policy: A mapping from perceived states of the environment to actions to be taken when in those states.
- reward signal: The goal in a reinforcement learning problem.
- value function: Specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Whereas *rewards* determine the immediate, intrinsic desirability of environmental states, *values* indicate the long-term desirability of states after taking into account the states that are likely to follow, and the rewards available in those states.
- model: Something that mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave.

**Key**: In fact, the most important component of almost all reinforcement learning algorithms we consider is a method for eﬃciently estimating values. The central role of value estimation is arguably the most important thing we have learned about reinforcement learning over the last few decades.

---

### Multi-armed Bandits

#### Preface

The most important feature distinguishing reinforcement learning from other types of learning is that it uses training information that evaluates the actions taken rather than instructs by giving correct actions.  Evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the action taken. 

#### Problem Statement

Consider the following learning problem. You are faced repeatedly with a choice among n diﬀerent options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or *time steps*. 

According to Wikipedia:

> The multi-armed bandit problem models an agent that simultaneously attempts to acquire new knowledge (called "exploration") and optimize his or her decisions based on existing knowledge (called "exploitation"). The agent attempts to balance these competing tasks in order to maximize his total value over the period of time considered. There are many practical applications of the bandit model, for example:
* clinical trials investigating the effects of different experimental treatments while minimizing patient losses
* adaptive routing efforts for minimizing delays in a network,
* financial portfolio design

#### Balancing Methods

In any speciﬁc case, whether it is better to explore or exploit depends in a complex way on the precise values of the estimates, uncertainties, and the number of remaining steps. That means both exploration and exploitation are important in our learning and we need to balance them well. There are many sophisticated methods for balancing exploration and exploitation but most of them make strong assumptions about stationarity and prior knowledge that are either violated or impossible to verify in applications. 

##### Initial Value Estimates

###### Action-Value Methods

We denote the true (actual) value of action $$a$$ as $$q(a)$$, and the estimated value on the tth time step as $$Q_t(a)$$. Recall that the true value of an action is the mean reward received when that action is selected. One natural way to estimate this is by averaging the rewards actually received when the action was selected. In other words, if by the $$t$$th time step action a has been chosen $$N_t(a)$$ times prior to $$t$$, yielding rewards $$R_1,R_2,...,R_{N_t}(a)$$, then its value is estimated to be 
$$
Q_t(a) = \frac{R_1+R_2+...+R_{N_t}(a)}{N_t(a)}
$$
If $$N_t(a) = 0$$, then we define $$Q_t(a)$$ instead as some default value, such as $$Q_1(a)=0$$. As $$N_t(a) \rightarrow \infty$$, by the law of large numbers, $$Q_t(a)$$ converges to $$q(a)$$.

The simplest action selection rule is to select the action (or one of the actions) with highest estimated action value. The *greedy* action selection method can be written as 
$$
A_t = argmax_a Q_t(a)
$$

A simple alternative is to behave greedily most of the time, but every once in a while, say with small probability $$\epsilon$$, instead to select randomly from amongst all the actions with equal probability independently of the actionvalue estimates. We call methods using this near-greedy action selection rule $$\epsilon$$-greedy methods. An advantage of these methods is that, in the limit as the number of plays increases, every action will be sampled an inﬁnite number of times, guaranteeing that $$N_t(a) \rightarrow \infty$$ for all $$a$$, and thus ensuring that all the $$Q_t(a)$$ converge to $$q(a)$$.

Despite the theoretically precise estimate, the cost of exploration is obviously too vast to cover especially taking the nonstationary practical environment, that is, that the true values of the actions changed over time into account. Even if the underlying task is stationary and deterministic, the learner faces a set of banditlike decision tasks each of which changes over time due to the learning process itself. Therefore, a more practical method is in need. 

###### Incremental Implementation

However if we take the straightforward implementation mentioned above, the memory and computational requirements will grow over time without bound. Luckily, this is not really necessary, we can devise incremental update formulas for computing averages with small, constant computation required to process each new reward. For some action, let $$Q_k$$ denote the estimate for its $$k$$th reward, that is, the average of its ﬁrst $$k−1$$ rewards. Given this average and a $$k$$th reward for the action, $$R_k$$, then the average of all $$k$$ rewards can be computed by
$$
\begin{align*}
Q_{k+1} &= \frac{1}{k} \sum_{i=1}^k R_i  \\
        &= \frac{1}{k} \left( R_k + \sum_{i=1}^{k-1} R_i \right)  \\
        &= \frac{1}{k} [R_K + (k-1)Q_k]  \\
        &= Q_k + \frac{1}{k} (R_k-Q_k)
\end{align*}
$$
which holds even for $$k = 1$$, obtaining $$Q_2 = R_1$$ for arbitrary $$Q_1$$. This implementation requires memory only for $$Q_k$$ and $$k$$, and only the small computation above for each new reward.

The update rule is of a form that occurs frequently throughout this book. The general form is 
$$
NewEstimate \leftarrow OldEstimate + StepSize [Target−OldEstimate].
$$

###### Tracking a Nonstationary Problem

The averaging methods discussed so far are appropriate in a stationary environment, but not if the bandit is changing over time. One of the most popular ways of doing nonstationary scenes is to use a constant step-szie parameter. For example, the incremental update rule (2.3) for updating an average Qk of the k−1 past rewards is modiﬁed to be 
$$
Q_{k+1} = Q_k + \alpha (R_k - Q_k)
$$
where the step-size parameter $$\alpha \in (0,1]$$ is constant. This results in $$Q_{k+1}$$ being a weighted average of past rewards and the initial estimate $$Q_1$$:
$$
\begin{align*}
Q_{k+1} &= Q_k + \alpha (R_k - Q_k)   \\
        &= \alpha R_k + (1-\alpha)Q_k \\
        &= \alpha R_k + (1-\alpha)[\alpha R_k + (1-\alpha)Q_{k-1}]  \\
        &= \alpha R_k + (1-\alpha)\alpha R_{k-1} + (1-\alpha)^2 \alpha R_{k-2} + ...  \\
        & + (1-\alpha)^{k-1}\alpha R_1 + (1-\alpha)^k Q_1  \\
        &= (1-\alpha)^k Q_1 + \sum_{i=1}^k \alpha(1-\alpha)^{k-i}R_i
\end{align*}
$$

Let $$\alpha_k(a)$$ denote the step-size parameter used to process the reward received after the $$k$$th selection of action $$a$$. To assure convergence with probability 1, we must the limit the conditions to:
$$
\sum_{k=1}^{\infty} \alpha_k(a) = \infty , \quad \sum_{k=1}^{\infty} \alpha_k^2(a) < \infty
$$
The ﬁrst condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random ﬂuctuations. The second condition guarantees that eventually the steps become small enough to assure convergence.

###### Optimistic Initial Values

This method applies a wildly optimistic initial estimate to encourage exploration. Suppose that instead of setting the initial action values to zero, as we did in the 10-armed testbed, we set them all to $$+5$$. Recall that the $$q(a)$$ in this problem are selected from a normal distribution with mean 0 and variance 1. An initial estimate of +5 is thus wildly optimistic. But this optimism encourages action-value methods to explore. Whichever actions are initially selected, the reward is less than the starting estimates; the learner switches to other actions, being “disappointed” with the rewards it is receiving. The result is that all actions are tried several times before the value estimates converge. The system does a fair amount of exploration even if greedy actions are selected all the time.

However, it is not well suited to nonstationary problems because its drive for exploration is inherently temporary. If the task changes, creating a renewed need for exploration, this method cannot help. Indeed, any method that focuses on the initial state in any special way is unlikely to help with the general nonstationary case. The beginning of time occurs only once, and thus we should not focus on it too much. This criticism applies as well to the sample-average methods, which also treat the beginning of time as a special event, averaging all subsequent rewards with equal weights. Nevertheless, all of these methods are very simple, and one of them or some simple combination of them is often adequate in practice.

##### Upper-Confidence-Bound Action Selection

Now let's come back to the action selection strategy. It has been witnessed that $$\epsilon$$-greedy action selection is more effective than the greedy actions in practice, but this strategy show no preference for those that are nearly greedy or paticularly uncertain. It would be better to select among the non-greedy actions according to their potential for actually being optimal, taking into account both how close their estimates are to being maximal and the uncertainties in those estimates. One eﬀective way of doing this is to select actions as
$$
A_t = argmax_a \left[Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]
$$
where $$\ln t$$ denotes the natural logarithm of $$t$$ (the number that $$e \approx 2.71828$$ would have to be raised to in order to equal $$t$$), and the number $$c > 0$$ controls the degree of exploration. If $$N_t(a) = 0$$, then $$a$$ is considered to be a maximizing action.

##### Gradient Bandits

*The following prove process is so beautiful that I made a nearly complete copy from the book except that I made some adjustment to make the document fit the markdown grammar and gitbook typesetting better.*

In this section we consider learning a numerical *preference* $$H_t(a)$$ for each action a. The larger the preference, the more often that action is taken, but the preference has no interpretation in terms of reward. Only the relative preference of one action over another is important; if we add 1000 to all the preferences there is no aﬀect on the action probabilities, which are determined according to a soft-max distribution (i.e., Gibbs or Boltzmann distribution) as follows:
$$
\Pr\{ A_t=a \} = \frac{e^{H_t(a)}}{\sum_{b=1}^n e^{H_t(b)}} = \pi_t(a)
$$
where here we have also introduced a useful new notation $$\pi_t(a)$$ for the probability of taking action $$a$$ at time $$t$$. Initially all preferences are the same (e.g., $$H_1(a)=0,\forall a$$) so that all actions have an equal probability of being selected. 

On each step, after selecting the action $$A_t$$ and receiving the reward $$R_t$$, the preferences are updated by: 
$$
H_{t+1}(A_t) = H_t(A_t) + \alpha(R_t - \bar{R_t})(1 - \pi_t(A_t))  \\
H_{t+1}(a) = H_t(a) - \alpha(R_t - \bar{R_t})\pi_t(a) \quad \forall a \neq A_t
$$
where $$\alpha > 0$$ is a step-size parameter, and $$\bar{R_t} \in \mathbb{R}$$ is the average of all the rewards up through and including time $$t$$.

To give a deeper insight into this algorithm, we can understand it as a stochastic approximation to gradient ascent. 
$$
H_{t+1}(a) = H_t(a) + \alpha \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}
$$
where the measure of performance here is the expected reward: 

$$
\mathbb{E}[R_t] = \sum_b \pi_t(b) q(b)  
$$

Of course, it is not possible to implement gradient ascent exactly in our case because by assumption we do not know the $$q(b)$$, but in fact these two algorithms are equal in the meaning of expected value, making the algorithm an instance of stochastic gradient ascent.

The calculations showing this require only beginning calculus, but take several steps. If you are mathematically inclined, then you will enjoy the rest of this section in which we go through these steps. First we take a closer look at the exact performance gradient:
$$
\begin{align*}
\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)} \left[ \sum_b \pi_t(b) q(b) \right]  \\
                                                 &= \sum_b q(b) \frac{\partial \pi_t(b)}{\partial H_t(a)}   \\
                                                 &= \sum_b \left( q(b) - X_t \right) \frac{\partial \pi_t(b)}{\partial H_t(a)}  \\
                                                 &= \sum_b \pi_t(b) \left( q(b) - X_t \right) \frac{\partial \pi_t(b)}{\partial H_t(a)} / \pi_t(b) ,
\end{align*}
$$
where $$X_t$$ can be any scalar that does not depend on $$b$$. We can include it here because the gradient sums to zero over the all the actions,$$\sum_b \partial \frac{\pi_t(b)}{\partial H_t(a)} = 0$$. As $$H_t(a)$$ is changed, some actions’ probabilities go up and some down, but the sum of the changes must be zero because the sum of the probabilities must remain one.

The equation is now in the form of an expectation, summing over all possible values b of the random variable At, then multiplying by the probability of taking those values. Thus:
$$
\begin{align*}
                                                &= \mathbb{E} \left[ \left( q(A_t) - X_t \right) \frac{\partial \pi_t(A_t)}{\partial H_t(a)} / \pi_t(A_t) \right] \\
                                                &= \mathbb{E} \left[ \left( R_t - \bar{R_t} \right) \frac{\partial \pi_t(A_t)}{\partial H_t(a)} / \pi_t(A_t) \right]
\end{align*}
$$

where here we have chosen $$X_t = \bar{R_t}$$ and substituted $$R_t$$ for $$q(A_t)$$, which is permitted because $$\mathbb{E}[R_t] = q(A_t)$$ and because all the other factors are nonrandom. Shortly we will establish that $$\frac{\partial \pi_t(b)}{\partial H_t(a)} = \pi_t(b) \left( \Pi_{a=b}−\pi_t(a) \right)$$, where $$\Pi_{a=b}$$ is deﬁned to be 1 if $$a = b$$, else 0. Assuming that for now we have
$$
\begin{align*}
                                                &= \mathbb{E}\left[ (R_t - \bar{R_t})\pi_t(A_t)(\Pi_{a=A_t}-\pi_t(a)) / \pi_t(A_t) \right]  \\
                                                &= \mathbb{E}\left[ (R_t-\bar{R_t})(\Pi_{a=A_t}-\pi_t(a)) \right] .
\end{align*}
$$

Recall that our plan has been to write the performance gradient as an expectation of something that we can sample on each step, as we have just done, and then update on each step proportional to the sample. Substituting a sample of the expectation above for the performance gradient yields: 
$$
H_{t+1} = H_t(a) + \alpha(R_t-\bar{R_t})(\Pi_{a=A_t} - \pi_t(a)), \quad \forall a
$$
which you will recognize as being equivalent to our original algorithm.

Thus it remains only to show that $$\frac{\partial \pi_t(b)}{\partial H_t(a)} = \pi_t(b) \left( \Pi_{a=b}−\pi_t(a) \right)$$, as we assumed earlier. Recall the standard quotient rule for derivatives:
$$
\frac{\partial}{\partial x} \left[ \frac{f(x)}{g(x)} \right] = \frac{ \frac{\partial f(x)}{\partial x} g(x) - f(x) \frac{\partial g(x)}{\partial x} }{g(x)^2}
$$

Using this, we can write
$$
\begin{align*}
\frac{\partial \pi_t(b)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)} \pi_t(b)  \\
                                          &= \frac{\partial}{\partial H_t(a)} \left[ \frac{e^{H_t(b)}}{\sum_{c=1}^n e^{H_t(c)} } \right]  \\
                                          &= \frac{ \frac{\partial e^{H_t(b)}}{\partial H_t(a)} \sum_{c=1}^n e^{H_t(c)} - e^{H_t(b)} \frac{\partial \sum_{c=1}^n e^{H_t(c)}}{\partial H_t(a)} }{\left( \sum_{c=1}^n e^{H_t(c)} \right)^2} \\
                                          &= \frac{\Pi_{a=b}e^{H_t(a)} \sum_{c=1}^n e^{H_t(c)} - e^{H_t(b)}e^{H_t(a)}}{ \left( \sum_{c=1}^n e^{H_t(c)} \right)^2}    \\
                                          &= \frac{\Pi_{a=b}e^{H_t(b)}}{\sum_{c=1}^n e^{H_t(c)}} - \frac{e^{H_t(b)+H_t(a)}}{\left( \sum_{c=1}^n e^{H_t(c)} \right)^2}  \\
                                          &= \Pi_{a=b} \pi_t(b) - \pi_t(b)\pi_t(a)  \\
                                          &= \pi_t(b)\left( \Pi_{a=b}-\pi_t(a) \right)
\end{align*}
$$

We have just shown that the expected update of the gradient-bandit algorithm is equal to the gradient of expected reward, and thus that the algorithm is an instance of stochastic gradient ascent. This assures us that the algorithm has robust convergence properties.

Note that we did not require anything of the reward baseline other than that it not depend on the selected action. For example, we could have set is to zero, or to 1000, and the algorithm would still have been an instance of stochastic gradient ascent. The choice of the baseline does not aﬀect the expected update of the algorithm, but it does aﬀect the variance of the update and thus the rate of convergence. Choosing it as the average of the rewards may not be the very best, but it is simple and works well in practice.

##### Associative Search (Contextual Bandits)

The above problems we consider are only nonassociative tasks, where there is no need to associate different actions with different situations. However, in a general reinforcement learning task there is more than one situation, and the goal is to learn a policy: a mapping from situations to the actions that are best in those situations. 

---

### Finite Markov Decision Processes

#### Preface

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

#### Returns

The agents' goal is to maximize the cumulative reward it receives in the long run. Suppose the sequence of rewards received after time step $$t$$ is denoted $$R_{t+1}, R_{t+2}, R_{t+3},...$$, then the *expected return*, where the return $$G_t$$ is defined as some specific function of the reward sequence. 

##### Discrete Situation

In the simplest case the return is the sum of the rewards: 
$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T
$$
where $$T$$ is a final time step. 

This approach makes sense in applications in which there is a natural notion of ﬁnal time step, that is, when the agent–environment interaction breaks naturally into subsequences, which we call *episodes*, such as plays of a game, trips through a maze, or any sort of repeated interactions. Each episode ends in a special state called the *terminal state*, followed by a reset to a standard starting state or to a sample from a standard distribution of *starting states*. Tasks with episodes of this kind are called *episodic tasks*. 

##### Continual Situation

On the other hand, in many cases the agent–environment interaction does not break naturally into identiﬁable episodes, but goes on continually without limit. For example, this would be the natural way to formulate a continual process-control task, or an application to a robot with a long life span. We call these *continuing tasks*. 

The additional concept that we need is that of *discounting*. According to this approach, the agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized. In particular, it chooses At to maximize the expected *discounted return*:
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$
where \gamma is a parameter, $$0 \le \gamma \le 1$$, called the *discount rate*.

The discount rate determines the present value of future rewards: a reward received k time steps in the future is worth only $$\gamma^{k−1}$$ times what it would be worth if it were received immediately. If $$\gamma < 1$$, the inﬁnite sum has a ﬁnite value as long as the reward sequence $$\{R_k\}$$ is bounded. If $$\gamma = 0$$, the agent is “myopic” in being concerned only with maximizing immediate rewards: its objective in this case is to learn how to choose At so as to maximize only $$R_{t+1}$$. If each of the agent’s actions happened to inﬂuence only the immediate reward, not future rewards as well, then a myopic agent could maximize the equation by separately maximizing each immediate reward. But in general, acting to maximize immediate reward can reduce access to future rewards so that the return may actually be reduced. As $$\gamma$$ approaches 1, the objective takes future rewards into account more strongly: the agent becomes more farsighted.

#### Markov Decision Processes

##### Definition

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

#### Value Functions

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

#### Optimal Value Function

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

#### Optimality and Approximation

Even if the agent has a complete and accurate environment model, the agent is typically unable to perform enough computation per time step to fully use it. The memory available is also an important constraint. Memory may be required to build up accurate approximations of value functions, policies, and models. In most cases of practical interest there are far more states than could possibly be entries in a table, and approximations must be made.


### Dynamic Programming

