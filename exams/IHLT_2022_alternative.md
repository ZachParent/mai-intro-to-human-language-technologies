# Alternative Practice Exam: IHLT 2022

## Exercise 1: Parsing with Probabilistic CKY Algorithm

### Problem
Given the sentence:

```
Cats chase mice in darkness
```

and the PCFG:

```
S -> NP VP (1.0)
NP -> DT NN (0.4) | NN NN (0.3) | NN (0.3)
VP -> V NP (0.5) | V PP (0.5)
PP -> P NP (1.0)
NN -> cats (0.3) | mice (0.4) | darkness (0.3)
V -> chase (1.0)
DT -> the (1.0)
P -> in (1.0)
```

Apply the CKY algorithm to parse the sentence and compute probabilities for all valid parse trees. Convert the grammar into CNF if necessary.

### Solution
#### Step 1: Convert Grammar to CNF
Convert multi-word rules into binary productions:

```
S -> NP VP
NP -> DT NN | NN NN | NN
VP -> V NP | V PP
PP -> P NP
NN -> cats | mice | darkness
V -> chase
DT -> the
P -> in
```

#### Step 2: Apply CKY Algorithm
Build a CKY dynamic table for the sentence:

| i \ j | 1:Cats       | 2:Chase      | 3:Mice      | 4:In         | 5:Darkness    |
|-------|--------------|--------------|-------------|--------------|---------------|
| 1     | NN:0.3       | V:1.0        | NN:0.4      | P:1.0        | NN:0.3        |
| 2     |              | VP:0.15      | S:0.12      | PP:0.3       |               |
| 3     |              |              | VP:0.15     | S:0.09       |               |
| 4     |              |              |             |              |               |

#### Step 3: Parse Trees
1. **Tree 1:** "Cats chase mice" (S -> NP VP).
   - Probability: 0.3 (NP) * 0.5 (VP) = 0.15.
2. **Tree 2:** "Cats chase mice in darkness" (S -> NP VP -> NP V PP).
   - Probability: 0.3 (NP) * 0.5 (VP) * 0.3 (PP) = 0.045.

#### Final Result
- Best Parse Tree: Tree 1 with probability 0.15.

---

## Exercise 2: PoS Tagging with CRFs

### Problem
Define feature templates for PoS tagging using the tagset {DT, NN, V, P}. Given the history:

```
h = <ti-2, ti-1, ti, w[1:n], i> = <DT, NN, V, the cat chases the mouse, 4>
```

Identify correct features and compute the number of possible histories for a given sequence.

### Solution
#### Part 1: Feature Templates
1. **Template Examples:**
   - `f1(h): 1 if ti = NN and ti-1 = DT.`
   - `f2(h): 1 if ti = V and wi = mouse.`

#### Part 2: Possible Histories
- Number of tags: \(|T| = 4\).
- Sequence length \(n\): Possible histories = \(n \times |T|^3\).

---

## Exercise 3: WordNet Similarity for WSD

### Problem
Disambiguate "chase" in "Cats chase mice in darkness" using WordNet.

### Solution
#### Step 1: Synset Definitions
From WordNet:
- S1: (v) chase (to pursue something).
- S2: (v) chase (to engrave metal).
- S3: (v) chase (to cause to flee).

#### Step 2: Lesk Algorithm
Overlap contexts "chase mice in darkness" with definitions:
- S1: Matches "pursue something" (mice context).
- S2: No match.
- S3: Matches "cause to flee" (partial context).

#### Final Result
- Disambiguated sense: S1 (chase, to pursue something).

---

## Summary
- **Exercise 1:** Best parse tree resolves "Cats chase mice" as S -> NP VP.
- **Exercise 2:** CRF features and histories are structured based on contextual relationships.
- **Exercise 3:** Lesk algorithm disambiguates "chase" to mean "to pursue something."

