## Goal

This document specifies a concrete implementation of the method called **Janus** for sequential task memory control. In sequential task streams, existing methods optimize memory myopically, so the final memory $M_T$ may be biased toward recent tasks and may not be the best memory state for future unseen tasks from the same distribution. We aim to design a plug-in memory-selection/update controller that makes the final memory more globally useful. We have implemented several baselines in the methods folder. For Janus, we mainly consider plugging it into dynamic cheatsheet retrieval-and-synthesis (DC-RS) and ExpeL. Each method has a function `get_step_memories()` that returns a dictionary with keys "old_memory" and "new_memory". After solving task $x_t$ and generating the candidate new memory, we can use this function to get $M_{t-1}$ and $\hat{M}_t$. **Janus** can be applied to DC-RS and ExpeL as a plug-in that adaptively decides whether to keep the old memory or use the new one.

---

## 1. Problem Setting

Given a LLM memory method $P$ and a stream of tasks: $(x_1,y_1), (x_2,y_2), \dots, (x_T,y_T)$

At step $t$:

1. We have the current deployed memory $M_{t-1}$.
2. We solve task $x_t$ with the LLM conditioned on $M_{t-1}$, producing answer $\hat{y}_t$.
3. The ground-truth signal $y_t$ tells us whether $\hat{y}_t$ is correct.
4. The base memory method $P$ proposes a candidate updated memory $\hat{M}_t$.
5. Our method decides whether to:
   - keep $M_{t-1}$,
   - accept $\hat{M}_t$,
   - and when triggered, compare old vs new memory on a compact support set.

Our method is **not** just improving the current task. The goal is to make the final memory $M_T$ after the stream more useful for unseen future tasks from the same distribution.

---

## 2. High-Level Method Overview

Our method has three components:

1. **Base Memory Updater**
   - Given existing sequential memory method.
   - Input: current memory $M_{t-1}$, current task $x_t$ , current answer $\hat{y}_t$, correctness feedback.
   - Output: candidate updated memory $\hat{M}_t$.

2. **Memory Momentum Trigger (MMT)**
   - Decides **when** to run the expensive comparison between $M_{t-1}$ and $\hat{M}_t$.
   - Triggered only when the memory update trajectory shows a meaningful directional shift.
   - If it's not triggered at time $t$, we set $M_t$ as $\hat{M}_t$ and move to the next task.

3. **Support Set Comparison**
   - When MMT is triggered, compare $M_{t-1}$ and $\hat{M}_t$ on a compact support set of size $K$ and the fresh set of a small set of newly seen tasks since the last trigger of size $L$.
   - The support set is split into:
     - **Coverage Set** of size $K'$
     - **Boundary Set** of size $K-K'$

---

## 3. Core State Maintained During the Stream

Maintain the following objects:

<!-- ### 3.1 Current deployed memory `old_memory` and Candidate memory `new_memory` -->

### 3.1 Pending tasks set

Store the tasks between two MMT triggers, i.e., between two memory evaluations.

Each entry stores:
- `input`: $x_t$
- `output`: $\hat{y}_t$
- `gt`: $y_t$
- `embedding`: embedding of $x_t$


### 3.2 Seen tasks set

After solving task $x_t$, the set should be a list of $t$ entries.

Each entry stores:
- `input`: $x_t$
- `output`: $\hat{y}_t$
- `gt`: $y_t$
- `embedding`: embedding of $x_t$




### 3.3 Coverage set
A list of `K_prime` entries.

Each entry stores:
- `centroid`: centroid embedding vector
- `rep_task`: representative task object
- `rep_embedding`: embedding of representative task

Suggested structure:
```python
CoverageEntry = {
    "centroid": np.ndarray,
    "rep_task": task_object,
    "embedding": np.ndarray,
}
coverage_set: list[CoverageEntry]
```

### 3.4 Boundary set
A list of `K - K_prime` tasks.

Each entry may store:
- `task`
- `embedding`
- optional metadata like source trigger step

Suggested structure:
```python
BoundaryEntry = {
    "task": task_object,
    "embedding": np.ndarray,
    "tag": "helpful_flip" or "harmful_flip" or "bootstrap"
}
boundary_set: list[BoundaryEntry]
```

### 3.5 MMT momentum state
```python
momentum_vector: np.ndarray | None
```

---

## 4. Warm-Up Phase

We do **not** run support-set comparison before the first $K$ tasks.

### 4.1 Process first K tasks
For tasks $(x_1,y_1), (x_2,y_2), \dots, (x_K,y_K)$:
1. Solve task with current memory.
2. Verify answer.
3. Use base updater $P$ to produce next memory.
4. Deploy that updated memory directly.
5. Store task and task embedding.

### 4.2 Initialize support set after first K tasks

#### Coverage set initialization
Run k-means with `K_prime` clusters on the embeddings of the first `K` tasks.

For each cluster:
- compute centroid
- choose the actual task nearest to the centroid as the representative

You can use the following code to do the k-means
```python
from kmeans_pytorch import kmeans
cluster_ids, cluster_centroid = kmeans(
    X=embeddings, num_clusters=K_prime, distance='euclidean', device=device
)
```

These `K_prime` representatives form the initial coverage set.

#### Boundary set initialization
Use the remaining `K - K_prime` warm-up tasks as the initial boundary set.

This is only a bootstrap. Later, the boundary set will be gradually replaced by true flip tasks discovered during comparisons.

---

## 5. Memory Momentum Trigger (MMT)

MMT decides whether we should trigger comparison between `old_memory` and `new_memory`.

### 5.1 Update vector

Let $z_t = \phi(\hat M_t) - \phi(M_{t-1})$


where:
- $\phi(\cdot)$ is a text encoder model such as Qwen/Qwen3-Embedding-0.6B
- $\hat{M}_t$ is the candidate memory (new)
- $M_{t-1}$ is the current deployed memory (old)

Implementation:
```python
z_t = embed_text(new_memory) - embed_text(old_memory)
```

### 5.2 Momentum update

Maintain exponential moving average $m_t = \beta m_{t-1} + (1 - \beta) z_t$

If no previous momentum exists, initialize:
```python
momentum_vector = z_t
```

### 5.3 Trigger rule

Preferred simple version:

Trigger if

$\cos(z_t, m_{t-1}) < \tau$


Implementation note:
- if `momentum_vector` is `None`, do not trigger based on MMT yet
- define cosine robustly with epsilon denominator protection



### 5.4 When MMT is checked
At each task $x_t$ after warm-up:
1. solve task
2. generate new memory
3. compute $z_t$
4. run MMT
5. if not triggered:
   - deploy `new_memory` directly

   if triggered:
   - seen tasks set = seen tasks set + pending tasks set
   - refresh Coverage set
   - evaluate `old_memory` vs `new_memory`
   - refresh Boundary set
   - set pending tasks set as empty

6. add current task to pending tasks set

**Recommended default:** if MMT is **not** triggered, deploy `new_memory` directly.  

---

## 6. Support Set Structure

At trigger time, evaluate on:

$S = S^{cov} \cup S^{bdry}$

where:
- `len(S_cov) = K_prime`
- `len(S_bdry) = K - K_prime`

### 6.1 Coverage set maintenance

Coverage set should adapt to new regions of the seen task distribution efficiently. At trigger time, run k-means with `K_prime` clusters on the embeddings of `seen tasks set` with the previous cluster_centroid as the cluster centers to reduce the clustering iteration.

```python
from kmeans_pytorch import kmeans
cluster_ids, cluster_centroid = kmeans(
    X=embeddings, num_clusters=K_prime, cluster_centers = cluster_centroid, distance='euclidean', device=device
)
```

To refresh Coverage set, for each updated centroid choose the actual task in the `seen tasks set` nearest to the centroid and update `coverage_set`.


### 6.2 Boundary Set Maintenance

Stores memory-sensitive tasks where previous memory comparisons actually changed correctness.
The boundary set should contain tasks that were historically informative for distinguishing memories. The initial boundary set from warm-up is just bootstrap data. Over time, it should be progressively replaced by true flip tasks.

When MMT fires, we do **not** evaluate only on the old stored support set.  
Instead, we use:

```python
eval_tasks = support_set + fresh_set
```
where the fresh_set is a small set of size $L$ of newly seen tasks since the last trigger (sampled from the pending tasks).


After comparing `old_memory` and `new_memory` on `eval_tasks`, define new_flip_tasks:

$F = \{x \in eval\_tasks-coverage\_set  : \texttt{correct}_{old}(x) \neq \texttt{correct}_{new}(x)\}$

These are the **flip tasks**. $\texttt{correct}_{old}(x)=1$ if the LLM answers task $x$ correctly using `old_memory`, and 0 if it answers incorrectly. $\texttt{correct}_{new}(x)=1$ if the LLM answers task $x$ correctly using `new_memory`, and 0 if it answers incorrectly.



Maintain a fixed-size buffer of size `K_b = K - K_prime`.

```python
if len(F) >= K_b:
    new_boundary_set = sample(new_flip_tasks, K_b)
else:
    remaining = [
        x for x in old_boundary_set
        if x not in F and x not in coverage_set
    ]
    new_boundary_set = F + remaining[: K_b - len(F)]
```
