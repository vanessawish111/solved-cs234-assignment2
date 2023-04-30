Download Link: https://assignmentchef.com/product/solved-cs234-assignment2
<br>
In this assignment we will implement deep Q-learning, following DeepMind’s paper ([1] and [2]) that learns to play Atari games from raw pixels. The purpose is to demonstrate the effectiveness of deep neural networks as well as some of the techniques used in practice to stabilize training and achieve better performance. In the process, you’ll become familiar with PyTorch. We will train our networks on the Pong-v0 environment from OpenAI gym, but the code can easily be applied to any other environment.

In Pong, one player scores if the ball passes by the other player. An episode is over when one of the players reaches 21 points. Thus, the total return of an episode is between −21 (lost every point) and +21 (won every point). Our agent plays against a decent hard-coded AI player. Average human performance is −3 (reported in [2]). In this assignment, you will train an AI agent with super-human performance, reaching at least +10 (hopefully more!).

1

<h1>0           Distributions induced by a policy</h1>

In this problem, we’ll work with an infinite-horizon MDP M = hS<em>,</em>A<em>,</em>R<em>,</em>T <em>,γ</em>i and consider stochastic policies of the form <em>π </em>: S → ∆(A)<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. Additionally, we’ll assume that M has a single, fixed starting state <em>s</em><sub>0 </sub>∈ S for simplicity.

<ul>

 <li>(<strong>written</strong>, 3 pts) Consider a fixed stochastic policy and imagine running several rollouts of this policy within the environment. Naturally, depending on the stochasticity of the MDP M and the policy itself, some trajectories are more likely than others. Write down an expression for <em>ρ<sup>π</sup></em>(<em>τ</em>), the likelihood of sampling a trajectory <em>τ </em>= (<em>s</em><sub>0</sub><em>,a</em><sub>0</sub><em>,s</em><sub>1</sub><em>,a</em><sub>1</sub><em>,…</em>) by running <em>π </em>in M. To put this distribution in context,</li>

</ul>

recall that<em> .</em>

<ul>

 <li>(<strong>written</strong>, 5 pts) Just as <em>ρ<sup>π </sup></em>captures the distribution over trajectories induced by <em>π</em>, we can also examine the distribution over states induced by <em>π</em>. In particular, define the <em>discounted, stationary state distribution </em>of a policy <em>π </em>as</li>

</ul>

<em>,</em>

where <em>p</em>(<em>s<sub>t </sub></em>= <em>s</em>) denotes the probability of being in state <em>s </em>at timestep <em>t </em>while following policy <em>π</em>; your answer to the previous part should help you reason about how you might compute this value. Consider an arbitrary function <em>f </em>: S × A → R. Prove the following identity:

<em>.</em>

<em>Hint: You may find it helpful to first consider how things work out for f</em>(<em>s,a</em>) = 1<em>,</em>∀(<em>s,a</em>) ∈ S × A<em>.</em>

<em>Hint: What is p</em>(<em>s<sub>t </sub></em>= <em>s</em>)<em>?</em>

<ul>

 <li>(<strong>written</strong>, 5 pts) For any policy <em>π</em>, we define the following function</li>

</ul>

<em>A<sup>π</sup></em>(<em>s,a</em>) = <em>Q<sup>π</sup></em>(<em>s,a</em>) − <em>V <sup>π</sup></em>(<em>s</em>)<em>.</em>

Prove the following statement holds for all policies <em>π,π</em><sup>0</sup>:

<em> .</em>

<h1>1           Test Environment</h1>

Before running our code on Pong, it is crucial to test our code on a test environment. In this problem, you will reason about optimality in the provided test environment by hand; later, to sanity-check your code, you will verify that your implementation is able to achieve this optimality. You should be able to run your models on CPU in no more than a few minutes on the following environment:

<ul>

 <li>4 states: 0<em>,</em>1<em>,</em>2<em>,</em>3</li>

 <li>5 actions: 0<em>,</em>1<em>,</em>2<em>,</em>3<em>,</em> Action 0 ≤ <em>i </em>≤ 3 goes to state <em>i</em>, while action 4 makes the agent stay in the same state.</li>

 <li>Rewards: Going to state <em>i </em>from states 0, 1, and 3 gives a reward <em>R</em>(<em>i</em>), where <em>R</em>(0) = 0<em>.</em>2<em>,R</em>(1) =</li>

</ul>

−0<em>.</em>1<em>,R</em>(2) = 0<em>.</em>0<em>,R</em>(3) = −0<em>.</em>3. If we start in state 2, then the rewards defind above are multiplied by −10. See Table 1 for the full transition and reward structure.

<ul>

 <li>One episode lasts 5 time steps (for a total of 5 actions) and always starts in state 0 (no rewards at the initial state).</li>

</ul>

<table width="332">

 <tbody>

  <tr>

   <td width="67">State (<em>s</em>)</td>

   <td width="76">Action (<em>a</em>)</td>

   <td width="103">Next State (<em>s</em><sup>0</sup>)</td>

   <td width="85">Reward (<em>R</em>)</td>

  </tr>

  <tr>

   <td width="67">0</td>

   <td width="76">0</td>

   <td width="103">0</td>

   <td width="85">0.2</td>

  </tr>

  <tr>

   <td width="67">0</td>

   <td width="76">1</td>

   <td width="103">1</td>

   <td width="85">-0.1</td>

  </tr>

  <tr>

   <td width="67">0</td>

   <td width="76">2</td>

   <td width="103">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="67">0</td>

   <td width="76">3</td>

   <td width="103">3</td>

   <td width="85">-0.3</td>

  </tr>

  <tr>

   <td width="67">0</td>

   <td width="76">4</td>

   <td width="103">0</td>

   <td width="85">0.2</td>

  </tr>

  <tr>

   <td width="67">1</td>

   <td width="76">0</td>

   <td width="103">0</td>

   <td width="85">0.2</td>

  </tr>

  <tr>

   <td width="67">1</td>

   <td width="76">1</td>

   <td width="103">1</td>

   <td width="85">-0.1</td>

  </tr>

  <tr>

   <td width="67">1</td>

   <td width="76">2</td>

   <td width="103">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="67">1</td>

   <td width="76">3</td>

   <td width="103">3</td>

   <td width="85">-0.3</td>

  </tr>

  <tr>

   <td width="67">1</td>

   <td width="76">4</td>

   <td width="103">1</td>

   <td width="85">-0.1</td>

  </tr>

  <tr>

   <td width="67">2</td>

   <td width="76">0</td>

   <td width="103">0</td>

   <td width="85">-2.0</td>

  </tr>

  <tr>

   <td width="67">2</td>

   <td width="76">1</td>

   <td width="103">1</td>

   <td width="85">1.0</td>

  </tr>

  <tr>

   <td width="67">2</td>

   <td width="76">2</td>

   <td width="103">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="67">2</td>

   <td width="76">3</td>

   <td width="103">3</td>

   <td width="85">3.0</td>

  </tr>

  <tr>

   <td width="67">2</td>

   <td width="76">4</td>

   <td width="103">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="67">3</td>

   <td width="76">0</td>

   <td width="103">0</td>

   <td width="85">0.2</td>

  </tr>

  <tr>

   <td width="67">3</td>

   <td width="76">1</td>

   <td width="103">1</td>

   <td width="85">-0.1</td>

  </tr>

  <tr>

   <td width="67">3</td>

   <td width="76">2</td>

   <td width="103">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="67">3</td>

   <td width="76">3</td>

   <td width="103">3</td>

   <td width="85">-0.3</td>

  </tr>

  <tr>

   <td width="67">3</td>

   <td width="76">4</td>

   <td width="103">3</td>

   <td width="85">-0.3</td>

  </tr>

 </tbody>

</table>

Table 1: Transition table for the Test Environment

An example of a trajectory (or episode) in the test environment is shown in Figure 1, and the trajectory can be represented in terms of <em>s<sub>t</sub>,a<sub>t</sub>,R<sub>t </sub></em>as: <em>s</em><sub>0 </sub>= 0<em>,a</em><sub>0 </sub>= 1<em>,R</em><sub>0 </sub>= −0<em>.</em>1<em>,s</em><sub>1 </sub>= 1<em>,a</em><sub>1 </sub>= 2<em>,R</em><sub>1 </sub>= 0<em>.</em>0<em>,s</em><sub>2 </sub>= 2<em>,a</em><sub>2 </sub>= 4<em>,R</em><sub>2 </sub>= 0<em>.</em>0<em>,s</em><sub>3 </sub>= 2<em>,a</em><sub>3 </sub>= 3<em>,R</em><sub>3 </sub>= 3<em>.</em>0<em>,s</em><sub>4 </sub>= 3<em>,a</em><sub>4 </sub>= 0<em>,R</em><sub>4 </sub>= 0<em>.</em>2<em>,s</em><sub>5 </sub>= 0.

Figure 1: Example of a trajectory in the Test Environment

(a) (<strong>written </strong>6 pts) What is the maximum sum of rewards that can be achieved in a single trajectory in the test environment, assuming <em>γ </em>= 1? Show first that this value is attainable in a single trajectory, and then briefly argue why no other trajectory can achieve greater cumulative reward.

<h1>2           Tabular Q-Learning</h1>

If the state and action spaces are sufficiently small, we can simply maintain a table containing the value of <em>Q</em>(<em>s,a</em>), an estimate of <em>Q</em><sup>∗</sup>(<em>s,a</em>), for every (<em>s,a</em>) pair. In this <em>tabular setting</em>, given an experience sample (<em>s,a,r,s</em><sup>0</sup>), the update rule is

(1)

where <em>α &gt; </em>0 is the learning rate, <em>γ </em>∈ [0<em>,</em>1) the discount factor.

<strong>-Greedy Exploration Strategy </strong>For exploration, we use an -greedy strategy. This means that with probability , an action is chosen uniformly at random from A, and with probability 1−, the greedy action (i.e., argmax<em><sub>a</sub></em><sub>∈A </sub><em>Q</em>(<em>s,a</em>)) is chosen.

<ul>

 <li>(<strong>coding</strong>, 3 pts) Implement the getaction and update functions in py. Test your implementation by running python q2schedule.py.</li>

</ul>

<strong>Overestimation bias </strong>We will now examine the issue of overestimation bias in Q-learning. The crux of the problem is that, since we take a max over actions, errors which cause Q to overestimate will tend to be amplified when computing the target value, while errors which cause Q to underestimate will tend to be suppressed.

<ul>

 <li>(<strong>written</strong>, 5 pts) Assume for simplicity that our Q function is an unbiased estimator of <em>Q</em><sup>∗</sup>, meaning that E[<em>Q</em>(<em>s,a</em>)] = <em>Q</em><sup>∗</sup>(<em>s,a</em>) for all states <em>s </em>and actions <em>a</em>. (Note that this expectation is over the randomness in <em>Q </em>resulting from the stochasticity of the exploration process.) Show that, even in this seemingly benign case, the estimator overestimates the real target in the following sense:</li>

</ul>

∀<em>s, </em>E<sup>h</sup>max<em>Q</em>(<em>s,a</em>)<sup>i </sup>≥ max<em>Q</em><sup>∗</sup>(<em>s,a</em>) <em>a                     a</em>

<h1>3           Q-Learning with Function Approximation</h1>

Due to the scale of Atari environments, we cannot reasonably learn and store a Q value for each state-action tuple. We will instead represent our Q values as a parametric function <em>Q<sub>θ</sub></em>(<em>s,a</em>) where <em>θ </em>∈ R<em><sup>p </sup></em>are the parameters of the function (typically the weights and biases of a linear function or a neural network). In this <em>approximation setting</em>, the update rule becomes

)                                              (2)

where (<em>s,a,r,s</em><sup>0</sup>) is a transition from the MDP.

To improve the data efficiency and stability of the training process, DeepMind’s paper [1] employed two strategies:

<ul>

 <li>A <em>replay buffer </em>to store transitions observed during training. When updating the <em>Q </em>function, transitions are drawn from this replay buffer. This improves data efficiency by allowing each transition to be used in multiple updates.</li>

 <li>A <em>target network </em>with parameters <em>θ</em><sup>¯ </sup>to compute the target value of the next state, max<em><sub>a</sub></em>0 <em>Q</em>(<em>s</em><sup>0</sup><em>,a</em><sup>0</sup>). The update becomes</li>

</ul>

)                                         (3)

Updates of the form (3) applied to transitions sampled from a replay buffer D can be interpreted as performing stochastic gradient descent on the following objective function:

<em>L</em><sub>DQN</sub>                                            (4)

Note that this objective is also a function of both the replay buffer D and the target network <em>Q<sub>θ</sub></em>¯. The target network parameters <em>θ</em><sup>¯ </sup>are held fixed and not updated by SGD, but periodically – every <em>C </em>steps – we synchronize by copying <em>θ</em><sup>¯</sup>← <em>θ</em>.

We will now examine some implementation details.

<ul>

 <li>(<strong>written </strong>3 pts) DeepMind’s deep Q network (DQN) takes as input the state <em>s </em>and outputs a vector of size |A|, the number of actions. What is one benefit of computing the <em>Q </em>function as <em>Q<sub>θ</sub></em>(<em>s,</em>) ∈ R<sup>|A|</sup>, as opposed to <em>Q<sub>θ</sub></em>(<em>s,a</em>) ∈ R?</li>

 <li>(<strong>written </strong>5 pts) Describe the tradeoff at play in determining a good choice of <em>C</em>. In particular, why might it not work to take a very small value such as <em>C </em>= 1? Conversely, what is the consequence of taking <em>C </em>very large? What happens if <em>C </em>= ∞?</li>

 <li>(<strong>written</strong>, 5 pts) In supervised learning, the goal is typically to minimize a predictive model’s error on data sampled from some distribution. If we are solving a regression problem with a one-dimensional output, and we use mean-squared error to evaluate performance, the objective writes</li>

</ul>

<em>L</em>(<em>θ</em>) = E [(<em>y </em>− <em>f<sub>θ</sub></em>(<strong>x</strong>))<sup>2</sup>]

(<strong>x</strong><em>,y</em>)∼D

where <strong>x </strong>is the input, <em>y </em>is the output to be predicted from <strong>x</strong>, D is a dataset of samples from the (unknown) joint distribution of <strong>x </strong>and <em>y</em>, and <em>f<sub>θ </sub></em>is a predictive model parameterized by <em>θ</em>.

This objective looks very similar to the DQN objective (4). How are these two scenarios different? (There are at least two significant differences.)

<h1>4           Linear Approximation</h1>

<ul>

 <li>(<strong>written</strong>, 5 pts) Suppose we represent the <em>Q </em>function as <em>Q<sub>θ</sub></em>(<em>s,a</em>) = <em>θ</em><sup>&gt;</sup><em>δ</em>(<em>s,a</em>), where <em>θ </em>∈ R<sup>|S||A| </sup>and <em>δ </em>: S × A → R<sup>|S||A| </sup>with</li>

</ul>

1        if <em>s</em><sup>0 </sup>= <em>s,a</em><sup>0 </sup>= <em>a</em>

0    otherwise

Compute ∇<em><sub>θ</sub>Q<sub>θ</sub></em>(<em>s,a</em>) and write the update rule for <em>θ</em>. Argue that equations (1) and (2) from above are exactly the same when this form of linear approximation is used.

<ul>

 <li>(<strong>coding</strong>, 15 pts) We will now implement linear approximation in PyTorch. This question will set up the pipeline for the remainder of the assignment. You’ll need to implement the following functions in py (please read through q3lineartorch.py):

  <ul>

   <li>initializemodels</li>

   <li>getqvalues</li>

   <li>updatetarget • calcloss</li>

   <li>addoptimizer</li>

  </ul></li>

</ul>

Test your code by running python q3lineartorch.py <strong>locally on CPU</strong>. This will run linear approximation with PyTorch on the test environment from Problem 0. Running this implementation should only take a minute.

<ul>

 <li>(<strong>written</strong>, 3 pts) Do you reach the optimal achievable reward on the test environment? Attach the plot png from the directory results/q3linear to your writeup.</li>

</ul>

<h1>5           Implementing DeepMind’s DQN</h1>

<ul>

 <li>(<strong>coding </strong>10pts) Implement the deep Q-network as described in [1] by implementing initializemodels and getqvalues in py. The rest of the code inherits from what you wrote for linear approximation. Test your implementation <strong>locally on CPU </strong>on the test environment by running python q4naturetorch.py. Running this implementation should only take a minute or two.</li>

 <li>(<strong>written </strong>3 pts) Attach the plot of scores, png, from the directory results/q4nature to your writeup. Compare this model with linear approximation. How do the final performances compare?</li>

</ul>

How about the training time?

<h1>6           DQN on Atari (21 pts)</h1>

Reminder: Please remember to kill your VM instances when you are done using them!!

The Atari environment from OpenAI gym returns observations (or original frames) of size (210×160×3), the last dimension corresponds to the RGB channels filled with values between 0 and 255 (uint8). Following DeepMind’s paper [1], we will apply some preprocessing to the observations:

<ul>

 <li>Single frame encoding: To encode a single frame, we take the maximum value for each pixel color value over the frame being encoded and the previous frame. In other words, we return a pixel-wise max-pooling of the last 2 observations.</li>

 <li>Dimensionality reduction: Convert the encoded frame to grey scale, and rescale it to (80 × 80 × 1). (See Figure 2)</li>

</ul>

The above preprocessing is applied to the 4 most recent observations and these encoded frames are stacked together to produce the input (of shape (80×80×4)) to the Q-function. Also, for each time we decide on an action, we perform that action for 4 time steps. This reduces the frequency of decisions without impacting the performance too much and enables us to play 4 times as many games while training. You can refer to the <em>Methods Section </em>of [1] for more details.

<ul>

 <li>Original input (210 × 160 × 3) with RGB colors</li>

 <li>After preprocessing in grey scale of shape (80×80×1)</li>

</ul>

Figure 2: Pong-v0 environment

<ul>

 <li>(<strong>coding and written</strong>, 5 pts). Now we’re ready to train on the Atari Pong-v0 First, launch linear approximation on pong with python q5trainatarilinear.py <strong>on Azure’s GPU</strong>. This will train the model for 500,000 steps and should take approximately an hour. Briefly qualitatively describe how your agent’s performance changes over the course of training. Do you think that training for a larger number of steps would likely yield further improvements in performance? Explain your answer.</li>

 <li>(<strong>coding and written</strong>, 10 pts). In this question, we’ll train the agent with DeepMind’s architecture on the Atari Pong-v0 Run python q6trainatarinature.py <strong>on Azure’s GPU</strong>. This will train the model for 4 million steps. To speed up training, we have trained the model for 2 million steps. You are responsible for training it to completion, which should take <strong>12 hours</strong>. Attach the plot scores.png from the directory results/q6trainatarinature to your writeup. You should get a score of around 11-13 after 4 million total time steps. As stated previously, the DeepMind paper claims average human performance is −3.</li>

</ul>

As the training time is roughly 12 hours, you may want to check after a few epochs that your network is making progress. The following are some training tips: • If you terminate your terminal session, the training will stop. In order to avoid this, you should use screen to run your training in the background.

<ul>

 <li>The evaluation score printed on terminal should start at -21 and increase.</li>

 <li>The max of the q values should also be increasing</li>

 <li>The standard deviation of q shouldn’t be too small. Otherwise it means that all states have similar q values</li>

 <li>You may want to use Tensorboard to track the history of the printed metrics. You can monitor your training with Tensorboard by typing the command tensorboard –logdir=results and then connecting to ip-of-you-machine:6006. Below are our Tensorboard graphs from one training session:</li>

</ul>

<ul>

 <li>(<strong>written</strong>, 3 pts) In a few sentences, compare the performance of the DeepMind DQN architecture with the linear Q value approximator. How can you explain the gap in performance?</li>

 <li>(<strong>written</strong>, 3 pts) Will the performance of DQN over time always improve monotonically? Why or why not?</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> For a finite set X, ∆(X) refers to the set of categorical distributions with support on X or, equivalently, the ∆<sup>|X|−1 </sup>probability simplex.