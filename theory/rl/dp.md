## Dynamic Programming

**Note Before All**: We assume that the environment is a finite MDP in this chapter.

The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP). 

The key idea of DP, and of reinforcement learning generally, is the use of value functions to organize and structure the search for good policies.

DP algorithms are obtained by turning Bellman equations such as these into assignments, that is, into update rules for improving approximations of the desired value functions.

### Policy

#### Evaluation

Let's use the Bellman equation for $$v_{\pi}$$ as an update rule:
$$
\begin{align*}
v_{k+1}(s) &= \mathbb{E} \left[ R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s \right] \\
           &= \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_k(s') \right]
\end{align*}
$$

for all $$s \in \mathcal{S}$$. Clearly, $$v_k = v_{\pi}$$ is a ﬁxed point for this update rule because the Bellman equation for vπ assures us of equality in this case. Indeed, the sequence $$\{v_k\}$$ can be shown in general to converge to $$v_{\pi}$$ as $$k \rightarrow \infty$$ under the same conditions that guarantee the existence of $$v_{\pi}$$. This algorithm is called *iterative policy evaluation*. 

To produce each successive approximation, $$v_{k+1}$$ from $$v_k$$, iterative policy evaluation applies the same operation to each state $$s$$: it replaces the old value of $$s$$ with a new value obtained from the old values of the successor states of $$s$$, and the expected immediate rewards, along all the one-step transitions possible under the policy being evaluated. We call this kind of operation a *full backup*. Each iteration of iterative policy evaluation *backs up* the value of every state once to produce the new approximate value function $$v_{k+1}$$.

#### Improvement

The reason for computing the value function for a policy is to help find better policies. With the value function $$v_{\pi}$$ we know how good it is to follow the policy $$\pi$$ but we do not know whether it is better or worse to choose another action $$a \neq \pi(s)$$ ($$s$$ is the current state). Therefore, naturally we would like to consider the value of action $$a$$ in the state $$s$$. That is,
$$
\begin{align*}
q_{\pi}(s,a) &= \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t=s,                   A_t=a \right]        \\
             &= \sum_{s',r} p(s',r| s,a) \left[ r + \gamma v_{\pi}(s') \right]
\end{align*}
$$

The key criterion is whether this is greater than or less than $$v_{\pi}(s)$$. If it is greater—that is, if it is better to select a once in s and thereafter follow $$\pi$$ than it would be to follow $$\pi$$ all the time—then one would expect it to be better still to select $$a$$ every time $$s$$ is encountered, and that the new policy would in fact be a better one overall.

Well, now let's generalize this to a more universal situation. Let $$\pi$$ and $$\pi'$$ be any pair of deterministic policies such that, for all $$s \in \mathcal{S}$$,
$$
q_{\pi}(s, \pi'(s)) \ge v_{\pi}(s).
$$

Then the policy $$\pi'$$ must be as good as, or better than, $$\pi$$. That is, it must obtain greater or equal expected return from all states $$s \in \mathcal{S}$$: 
$$
v_{\pi'}(s) \ge v_{\pi}(s).
$$

The proof of the policy improvement theorem:
$$
\begin{align*}
v_{\pi}(s) &\le q_{\pi}(s, \pi'(s))     \\
           &= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t=s \right]        \\
           &\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma q_{\pi}(S_{t+1}, \pi'(S_{t+1})) | S_t=s \right]        \\
           &= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma \mathbb{E}_{\pi'} [R_{t+2} + \gamma v_{\pi}(S_{t+2})] | S_t=s \right]   \\
           &= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\pi}(S_{t+2}) | S_t=s \right]    \\
           &\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_{\pi}(S_{t+3}) | S_t=s \right]    \\
           &\cdots \\
           &\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots | S_t=s \right]    \\
           &= v_{\pi'}(s).
\end{align*}
$$

Then it is a natural extension to consider changes at all states and to all possible actions, selecting at each state the action that appears best according to $$q_{\pi}(s,a)$$. Consider the new greedy policy $$\pi'$$:
$$
\begin{align*}
\pi'(s) &= \mathop{\arg\max}_a q_{\pi}(s,a)     \\
        &= \mathop{\arg\max}_a \mathbb{E}\left[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t=s, A_t=a \right]        \\
        &= \mathop{\arg\max}_a \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_{\pi}(s') \right]
\end{align*}
$$

So far in this section we have considered the special case of deterministic policies, but it's natural to extend this to stochastic policies in the meaning of expectation:
$$
q_{\pi}(s,\pi'(s)) = \sum_a \pi'(a|s) q_{\pi}(s,a)
$$

#### Iteration

Once a policy, $$\pi$$, has been improved using $$v_{\pi}$$ to yield a better policy, $$\pi'$$, we can then compute $$v_{\pi}'$$ and improve it again to yield an even better $$π''$$. We can thus obtain a sequence of monotonically improving policies and value functions: 
$$
\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*,
$$
where $$\xrightarrow{E}$$ denotes a policy *evaluation* and $$\xrightarrow{I}$$ denotes a policy *improvement*. 

### Asynchronous Dynamic Programming

Asynchronous DP algorithms are in-place iterative DP algorithms that are not organized in terms of systematic sweeps of the state set. These algorithms back up the values of states in any order whatsoever, using whatever values of other states happen to be available. The values of some states may be backed up several times before the values of others are backed up once. To converge correctly, however, an asynchronous algorithm must continue to backup the values of all the states: it can’t ignore any state after some point in the computation.

Note that we can not get away with less computation. It just means that an algorithm does not need to get locked into any hopelessly long sweep before it can make progress improving a policy. 

Asynchronous algorithms are especially applicable when it comes to real-time interaction. To solve a given MDP, we can run an iterative DP algorithm *at the same time that an agent is actually experiencing the MDP*. 

### Generalized Policy Iteration

Policy iteration consists of two simultaneous, interacting processes, one making the value function consistent with the current policy (policy evaluation), and the other making the policy greedy with respect to the current value function (policy improvement). In policy iteration, these two processes alternate, each completing before the other begins, but this is not really necessary. In value iteration, for example, only a single iteration of policy evaluation is performed in between each policy improvement. In asynchronous DP methods, the evaluation and improvement processes are interleaved at an even finer grain.

We use the term *generalized policy iteration* (GPI) to refer to the general idea of letting policy evaluation and policy improvement processes interact, independent of the granularity and other details of the two processes. 

The evaluation and improvement processes in GPI can be viewed as both competing and cooperating. They compete in the sense that they pull in opposing directions. Making the policy greedy with respect to the value function typically makes the value function incorrect for the changed policy, and making the value function consistent with the policy typically causes that policy no longer to be greedy.

### Efficiency of Dynamic Programming

DP may not be practical for very large problems, but compared with other methods for solving MDPs, DP methods are actually quite efficient. If we ignore a few technical details, then the (worst case) time DP methods take to ﬁnd an optimal policy is polynomial in the number of states and actions. 

On problems with large state spaces, *asynchronous* DP methods are often preferred. 

### Summary

Finally, we note one last special property of DP methods. All of them update estimates of the values of states based on estimates of the values of successor states. That is, they update estimates on the basis of other estimates. We call this general idea bootstrapping. Many reinforcement learning methods perform bootstrapping, even those that do not require, as DP requires, a complete and accurate model of the environment.