## Reinforcement Learning

<p style="font-size: 11pt">This chapter is mainly organised according to <i>Reinforcement Learning: An Introduction</i> by {{"Richard2014"|cite}}.</p>

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

