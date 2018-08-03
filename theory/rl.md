## Reinforcement Learning

This chapter is mainly organised according to *Reinforcement Learning: An Introduction* by {{"Richard2014"|cite}}.

### Basic Ideas

**Conception/Definition**: To capture the most important aspects of the real problem facing a learning agent interacting with its environment to achieve a goal.

**Characteristics**: These three characteristics—being closed-loop in an essential way, not having direct instructions as to what actions to take, and where the consequences of actions, including reward signals, play out over extended time periods—are the three most important distinguishing features of reinforcement learning problems.

**Differences**: According to the book:

> Supervised learning is learning from a training set of labeled examples provided by a knowledgable external supervisor. Each example is a description of a situation together with a speciﬁcation—the label—of the correct action the system should take to that situation, which is often to identify a category to which the situation belongs. The object of this kind of learning is for the system to extrapolate, or generalize, its responses so that it acts correctly in situations not present in the training set. This is an important kind of learning, but alone it is not adequate for learning from interaction. In interactive problems it is often impractical to obtain examples of desired behavior that are both correct and representative of all the situations in which the agent has to act.

To smmarize, I think, the major discrimination lies on the envirionment. Supervised learning is for static environmental conditions while a reinforcement learning method is more suitable to handle dynamic scenes. Popularly speaking, supervised learning is to rear livestock in pens, by contrast, reinforcement learning is to leave livestock in wild by themselves. 

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

