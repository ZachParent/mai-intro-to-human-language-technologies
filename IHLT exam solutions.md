# IHLT Exam Cheat Sheet: Problems and Solutions

## Morphological Analysis

### Problem
**2019 Exam:** Draw a Finite State Transducer (FST) to compute morphotactics and analyze lexical forms for PoS tags CD, NN, NNS, VBP, VBZ, and VBG.

### Solution
- **States:** Create states for lexical roots (e.g., "fly," "control") and transitions for suffixes (-s, -ing).
- **Transitions:** Include rules for:
  - Singular/plural (NN -> NNS): Add transition with suffix "-s."
  - Gerunds (VB -> VBG): Add transition with suffix "-ing."
- **Spelling Rules:** Handle doubling consonants and vowel drop.

## PoS Tagging

### Problem
**2023 Exam:** Apply the Viterbi algorithm to tag "I saw bats yesterday" using an HMM.

### Solution
1. **Dynamic Table Construction:**
   - Rows represent states (PRP, VBD, NNS, etc.).
   - Columns represent words in the sentence.
2. **Backtracking:** Trace back from the most probable state to determine tags.
3. **Result:** Best sequence: PRP VBD NNS NN.

## Parsing

### Problem
**2022 Exam:** Use CKY algorithm with a PCFG to parse "Time flies like an arrow."

### Solution
1. **Grammar in CNF:** Convert rules to binary productions.
2. **Dynamic Table:** Populate cells with probabilities of rules generating substrings.
3. **Parse Tree:** Resulting parse tree combines "Time" as NP and "flies like an arrow" as VP.

## Coreference Resolution

### Problem
**2020 Exam:** Identify mentions and coreference chains for the sentence:
"Mark visited his grandma. She gave him a book."

### Solution
1. **Mentions:** Mark, his grandma, She, him, a book.
2. **Coreference Chains:**
   - (Mark, his, him).
   - (his grandma, She).
3. **Positive Pairs:** (Mark, his), (Mark, him).
4. **Negative Pairs:** (Mark, She), (his grandma, a book).

## Named Entity Recognition (NERC)

### Problem
**2018 Exam:** Write regular expressions for monetary amounts and names.

### Solution
1. **Regex for Money:** `[€$]?\d+(\.\d{1,2})?|\d+\s(thousand|million)\s\w+`
2. **Regex for Names:** `\b(Mr\.?|Mrs\.?|Dr\.?)\s[A-Z][a-z]+\b`

## Word Sense Disambiguation (WSD)

### Problem
**2019 Exam:** Use Simplified Lesk algorithm to disambiguate "tongue" in "I’m eating cow tongue."

### Solution
1. **Synset Definitions:** Compare contexts with definitions.
2. **Overlap:** Maximum overlap with "edible muscular organ" sense.

## Temporal Expressions

### Problem
**2017 Exam:** Create a CFG for temporal expressions like "At 12 am" or "Between 12 and 12:15."

### Solution
1. **Grammar:**
   ```
   TIME -> "At" DIGIT "am" | "Between" DIGIT "and" DIGIT
   ```
2. **Semantic Features:** Add lambda calculus annotations for time intervals.

## Semantic Parsing

### Problem
**2019 Exam:** Identify semantic roles in "The man showed a book to Mary."

### Solution
1. **Roles:**
   - Agent: "The man."
   - Theme: "a book."
   - Recipient: "Mary."

## Algorithms to Practice
1. **Viterbi Algorithm:** PoS tagging.
2. **CKY Algorithm:** Parsing with CFG/PCFG.
3. **Simplified Lesk:** Word Sense Disambiguation.
