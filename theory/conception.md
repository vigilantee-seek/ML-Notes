### Learning Algorithms

#### Definition of learning

> A computer program is said to learn from experience $$E$$ with respect to some class of tasks $$T$$ and performance measure $$P$$, if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$.
>
> --- by Mitchell (1997)

##### The Task, T

Machine learning tasks are usually described in terms of how the machine learning system should process an example. An example is a collection of features that have been quantitatively measured from some object or event that we want the machine learning system to process.

Most common machine learning tasks:

* **Classification**: In this type of task, the computer program is asked to specify which of $$k$$ categories some input belongs to. To solve this task, the learning algorithm is usually asked to produce a function $$f$$ that functions on the input vector $$x$$. The model assigns the input vector $$x$$ to a category identified by numeric code $$y$$. 

* **Classification with missing inputs**: Classiﬁcation becomes more challenging if the computer program is not guaranteed that every measurement in its input vector will always be provided. To solve the classiﬁcation task, the learning algorithm only has to deﬁne a single function mapping from a vector input to a categorical output. When some of the inputs may be missing, rather than providing a single classiﬁcation function, the learning algorithm must learn a set of functions. Each function corresponds to classifying $$x$$ with a diﬀerent subset of its inputs missing. This kind of situation arises frequently in medical diagnosis because many kinds of medical tests are expensive or invasive.

* **Regression**: The computer is asked to predict a numerical value given some input. Different from calssfication, the learning algorithm is asked to output a function rather than categories.

* **Transcription**: In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form. For example, in optical character recognition, the computer program is shown a photograph containing an image of text and is asked to return this text in the form of a sequence of characters (e.g., in ASCII or Unicode format).

* **Machine translation**: The input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language. 

* **Structured output**: This category subsumes the transcription and translation tasks. One example is parsing--mapping a natural language sentence into a tree that describes its grammatical structure by tagging nodes of the trees as being verbs, nouns, adverbs, and so on.

* **Anomaly detection**: In this type of task, the computer program sifts through a set of events or objects and ﬂags some of them as being unusual or atypical. An example of an anomaly detection task is credit card fraud detection. By modeling your purchasing habits, a credit card company can detect misuse of your cards.

* **Synthesis and sampling**: In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data. A typical example is speech synthesis. We provide a written sentence and ask the program to emit an audio waveform containing a spoken version of that sentence. This task is a kind of structured output task, but with the added qualiﬁcation that there is no single correct output for each input, and we explicitly desire a large amount of variation in the output, in order for the output to seem more natural and realistic.

* **Imputation of missing values**: In this type of task, the machine learning

  algorithm is given a new example $$x$$, but with some entries of $$x$$ missing. The algorithm must provide a prediction of the values of the missing entries.

* **Density estimation**: In the density estimation problem, the machine learning algorithm is asked to learn a function model $$p : R^n \rightarrow R $$, where $$p(x)$$ can be interpreted as a probability density function (if $$x$$ is continuous) or a probability mass function (if $$x$$ is discrete) on the space that the examples were drawn from.

##### The Experience, E

The difference between supervised learning and unsupervised learning:

Roughly speaking, unsupervised learning involves observing several examples of a random vector $$x$$ and attempting to implicitly or explicitly learn the probablity distribution $$p(x)$$, or some interesting properties of that distribution; while supervised learning involves observing several examples of a random vector $$x$$ and an associated value or vector $$y$$, then learning to predict $$y$$ from $$x$$, usually by estimating $$p(y|x)$$.

 #### Capacity, Overfitting and Underfitting

**How can we aﬀect performance on the test set when we can observe only the training set?**

> We must make some basic assumptions. The assumptions are that the examples in each dataset are independent from each other, and that the training set and test set are identically distributed, drawn from the same probability distribution as each other.

The factors determining how well a machine learning algorithm will perform are its ability to 

1. Make the training error small
2. Make the gap between training and test error small

These two factors correspond to the two central challenges in machine learning: **underﬁtting** and **overﬁtting**. Underfitting occurs when the gap between the training error value on the training set. Overfitting occurs when the gap between the training error and test error is too large.

We can control whether a model is more likely to overﬁt or underﬁt by altering its **capacity**. One way to control the capacity of a learning algorithm is by choosing its hypothesis space, the set of functions that the learning algorithm is allowed to select as being the solution. 

Then what is the source of overfitting? The answer is sampling noise. We know that deep neural networks are expressive models that can learn very complicated relationships between their inputs and outputs.  With limited training data, however, many of these complicated relationships will be the result of sampling noise, so they will exist in the training set but not in real test data even if it is drawn from the same distribution. This leads to overﬁtting.

##### The No Free Lunch Theorem

Averaged over all possible data-generating distributions, every classiﬁcation algorithm has the same error rate when classifying previously unobserved points. In other words, no machine learning way is universally any better than any other. 

Fortunately, these results hold only when we average over all possible data-generating distributions. If we make assumptions about the kinds of probability distributions we encounter in real-world applications, then we can design learning algorithms that perform well on these distributions.

##### Dropout

###### Targets

Overﬁtting is a serious problem in such networks. Large networks are also slow to use, making it diﬃcult to deal with overﬁtting by combining the predictions of many diﬀerent large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. 

###### Method

During training, dropout samples from an exponential number of diﬀerent “thinned” networks. At test time, it is easy to **approximate** the eﬀect of averaging the predictions of all these thinned networks by simply using a single "unthinned" network that has smaller weights.

A unit at training time that is present with probability p and is connected to units in the next layer with weights w while at test time, the unit is always present and the weights are multiplied by p. The output at test time is same as the expected output at training time.

###### Motivation

> The ability of a set of genes to be able to work well with another random set of genes makes them more robust. Since a gene cannot rely on a large set of partners to be present at all times, it must learn to do something useful on its own or in collaboration with a small number of other genes. According to this theory, the role of sexual reproduction is not just to allow useful new genes to spread throughout the population, but also to facilitate this process by reducing complex co-adaptations that would reduce the chance of a new gene improving the ﬁtness of an individual.

