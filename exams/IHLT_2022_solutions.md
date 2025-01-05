# 2022 IHLT Exam Solutions

## Exercise 1: Parsing with Probabilistic CKY Algorithm

### Problem
Given the sentence:

```
Time flies like an arrow
```

and the PCFG:

```
S -> NP VP (1.0)
NP -> DT NN (0.4) | NN NN (0.25) | NN (0.35)
VP -> V NP (0.6) | V ADVP (0.4)
ADVP -> ADV NP (1.0)
NN -> time (0.4) | flies (0.2) | arrow (0.4)
V -> flies (0.5) | like (0.5)
ADV -> like (1.0)
DT -> an (1.0)
```

Apply the CKY algorithm to parse the sentence and compute probabilities for all valid parse trees. Convert the grammar into CNF if necessary.

### Solution
#### Step 1: Convert Grammar to CNF
Convert multi-word rules into binary productions:

```
S -> NP VP
NP -> DT NN | NN NN | NN
VP -> V NP | V ADVP
ADVP -> ADV NP
NN -> time | flies | arrow
V -> flies | like
ADV -> like
DT -> an
```

#### Step 2: Apply CKY Algorithm
Build a CKY dynamic table for the sentence:

| i \ j | 1:Time       | 2:Flies      | 3:Like      | 4:An         | 5:Arrow    |
|-------|--------------|--------------|-------------|--------------|------------|
| 1     | NN:0.4       | NN:0.2, V:0.5 | ADV:1.0, V:0.5 | DT:1.0       | NN:0.4     |
| 2     |              | S:0.1, VP:0.24| VP:0.2       | NP:0.4       |            |
| 3     |              |              | ADVP:0.4     | VP:0.24      | S:0.06     |
| 4     |              |              |             |              |            |

#### Step 3: Parse Trees
1. **Tree 1:** "Time flies" (S -> NP VP).
   - Probability: 0.4 (NP) * 0.6 (VP) = 0.24.
2. **Tree 2:** "Time flies like an arrow" (S -> NP VP -> NP V NP).
   - Probability: 0.4 (NP) * 0.6 (V NP) * 0.4 (NP) = 0.096.

#### Final Result
- Best Parse Tree: Tree 1 with probability 0.24.

---

## Exercise 2: PoS Tagging with CRFs

### Problem
Define feature templates for PoS tagging using the tagset {DT, NN, V, ADV}. Given the history:

```
h = <ti-2, ti-1, ti, w[1:n], i> = <DT, NN, V, the man saw the dog, 3>
```

Identify correct features and compute the number of possible histories for a given sequence.

### Solution
#### Part 1: Feature Templates
1. **Template Examples:**
   - `f1(h): 1 if ti = NN and ti-1 = DT.`
   - `f2(h): 1 if ti = V and wi = dog.`

#### Part 2: Possible Histories
- Number of tags: \(|T| = 4\).
- Sequence length \(n\): Possible histories = \(n \times |T|^3\).

---

## Exercise 3: WordNet Similarity for WSD

### Problem
Disambiguate "flies" in "Time flies like an arrow" using WordNet.

### Solution
#### Step 1: Synset Definitions
From WordNet:
- S1: (n) fly, flies (insects).
- S2: (v) fly (to move through air).
- S3: (v) fly (to run away).

#### Step 2: Lesk Algorithm
Overlap contexts "flies like an arrow" with definitions:
- S1: No match.
- S2: Matches "move through air" (arrow context).
- S3: Matches "run away" (no arrow relevance).

#### Final Result
- Disambiguated sense: S2 (fly, move through air).

---

## Summary
- **Exercise 1:** Best parse tree resolves "Time flies" as S -> NP VP.
- **Exercise 2:** CRF features and histories are structured based on contextual relationships.
- **Exercise 3:** Lesk algorithm disambiguates "flies" to mean "move through air."
