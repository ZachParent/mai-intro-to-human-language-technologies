# 2023 IHLT Exam Solutions

## Exercise 1: PoS Tagging with Viterbi Algorithm

### Problem
Given the morphologically analyzed sentence:

```
I saw bats yesterday
PRP VBD NNS NN
NN NN JJ ADV
VBZ
```

Apply the Viterbi algorithm using the provided HMM matrices to find the best PoS tag sequence. Justify the result with a dynamic table.

### Solution
#### Step 1: Initialization
- HMM states: PRP, VBD, NNS, NN, VBZ, ADV.
- Observations: "I," "saw," "bats," "yesterday."
- Transition probabilities (`A`) and emission probabilities (`B`) are extracted from the exam matrix.

#### Step 2: Dynamic Programming Table
Construct the Viterbi table with states and observations. Compute probabilities step by step using:

**Formula:**

\[ v_{t}(s) = \, \max_{s'} \, [v_{t-1}(s') \cdot A(s', s) \cdot B(s, o_t)] \]

#### Step 3: Backtracking
Backtrack from the highest probability in the last column to find the optimal sequence.

#### Final Result:
- Best sequence: PRP VBD NNS ADV.
- Probability: Computed by multiplying the probabilities along the optimal path.

## Exercise 2: Noun-Phrase Chunking with CRF and Parsing

### Problem
Given the sentence:

```
Anne saw my mother eating with her glasses
NNP VBD PRP$ NN VBG IN PRP$ NNS
```

1. Design a feature template for noun-phrase chunking.
2. Draw parse trees derived by the PCFG and determine the best tree.
3. Annotate mentions, coreference chains, singletons, and training pairs for a mention-pair model.

### Solution
#### Part 1: Feature Template for CRF
**Template:**
- Current word and PoS tag.
- Previous and next words/tags.
- Prefixes/suffixes of current word.

**Example Feature:**
\[ f(i) = \text{"If current word is NN and next word is PRP$, label = NP."} \]

#### Part 2: Parse Trees with PCFG
1. Apply CKY algorithm using the grammar:
   ```
   S -> NP VP (1.0)
   NP -> PRP$ NN (0.5) | PRP$ NNS (0.3) | NNP (0.1)
   VP -> VBD NP (0.4) | VP AP (0.6)
   ```
2. Resulting trees:
   - Tree 1: "Anne saw [NP my mother] eating [PP with her glasses]."
   - Tree 2: "Anne saw [NP my mother eating] [PP with her glasses]."

**Best Tree:** Tree 1 (higher probability).

#### Part 3: Coreference Model
1. Mentions: Anne, my mother, her, her glasses.
2. Chains: (Anne, her), (my mother, her glasses).
3. Singletons: eating.
4. Training Pairs:
   - Positive: (Anne, her), (my mother, her glasses).
   - Negative: (Anne, my mother), (her, glasses).

## Exercise 3: Word Sense Disambiguation (WSD)

### Problem
Disambiguate "bats" in:

```
John saw bats ready to eat food.
```

### Solution
#### Step 1: Synset Definitions
From WordNet:
- S1: (n) bat, chiropteran (mammal with wings).
- S2: (n) bat (baseball equipment).
- S3: (v) bat (hit with a bat).

#### Step 2: Simplified Lesk Algorithm
- Context: "bats ready to eat food" overlaps with S1's definition: "mammal with wings."

#### Final Result
- Disambiguated sense: S1 (bat, chiropteran).
- Additional Information Needed: Context indicating "flying" or "mammal."

## Summary
- **Exercise 1:** Best PoS tag sequence is PRP VBD NNS ADV.
- **Exercise 2:** Best parse tree separates "eating" as part of VP.
- **Exercise 3:** WSD for "bats" resolves to the flying mammal sense.
