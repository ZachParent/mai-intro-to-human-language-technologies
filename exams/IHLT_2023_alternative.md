# Alternative Practice Exam: IHLT 2023

## Exercise 1: PoS Tagging with Viterbi Algorithm

### Problem
Given the morphologically analyzed sentence:

```
She reads books daily
PRP VBZ NNS ADV
NN VBZ NN JJ
VBP
```

Apply the Viterbi algorithm using a hypothetical HMM to find the best PoS tag sequence. Justify the result with a dynamic table.

### Solution
#### Step 1: Initialization
- HMM states: PRP, VBZ, NNS, NN, VBP, ADV, JJ.
- Observations: "She," "reads," "books," "daily."

#### Step 2: Dynamic Programming Table
Compute probabilities step by step using the formula:

\[ v_{t}(s) = \, \max_{s'} \, [v_{t-1}(s') \cdot A(s', s) \cdot B(s, o_t)] \]

#### Step 3: Backtracking
Trace back from the highest probability in the last column to find the optimal sequence.

#### Final Result:
- Best sequence: PRP VBZ NNS ADV.
- Probability: Computed by multiplying the probabilities along the optimal path.

---

## Exercise 2: Noun-Phrase Chunking with CRF and Parsing

### Problem
Given the sentence:

```
Tom visited her friend wearing a hat
NNP VBD PRP$ NN VBG DT NN
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
\[ f(i) = \text{"If current word is NN and next word is VBG, label = NP."} \]

#### Part 2: Parse Trees with PCFG
1. Apply CKY algorithm using the grammar:
   ```
   S -> NP VP (1.0)
   NP -> PRP$ NN (0.5) | DT NN (0.3) | NNP (0.2)
   VP -> VBD NP (0.6) | VP AP (0.4)
   ```
2. Resulting trees:
   - Tree 1: "Tom visited [NP her friend] wearing [NP a hat]."
   - Tree 2: "Tom visited [NP her friend wearing] [NP a hat]."

**Best Tree:** Tree 1 (higher probability).

#### Part 3: Coreference Model
1. Mentions: Tom, her friend, a hat.
2. Chains: (Tom, her friend).
3. Singletons: a hat.
4. Training Pairs:
   - Positive: (Tom, her friend).
   - Negative: (Tom, a hat).

---

## Exercise 3: Word Sense Disambiguation (WSD)

### Problem
Disambiguate "read" in:

```
Sarah likes to read mystery novels.
```

### Solution
#### Step 1: Synset Definitions
From WordNet:
- S1: (v) read (interpret written text).
- S2: (v) read (study and comprehend information).
- S3: (v) read (examine and understand symbols).

#### Step 2: Simplified Lesk Algorithm
- Context: "read mystery novels" overlaps with S1's definition: "interpret written text."

#### Final Result
- Disambiguated sense: S1 (read, interpret written text).
- Additional Information Needed: Context indicating "books" or "text."

---

## Summary
- **Exercise 1:** Best PoS tag sequence is PRP VBZ NNS ADV.
- **Exercise 2:** Best parse tree separates "wearing" as part of VP.
- **Exercise 3:** WSD for "read" resolves to the sense of interpreting written text.

