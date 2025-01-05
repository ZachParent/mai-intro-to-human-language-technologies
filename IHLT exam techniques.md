# Exam Preparation Summary for IHLT Exam

## Key Problem Types and Algorithms

### 1. **Morphological Analysis**
   - **Finite State Transducers (FSTs)**: Used for analyzing lexical and surface forms.
     - Examples: Building morphotactics for PoS tags (e.g., NN, NNS, VBP, VBZ).
     - Incorporating spelling rules like doubling consonants or changing suffixes.
   - **Finite State Automata (FSA)**: Useful for detecting valid word forms and prefixes/suffixes.

### 2. **Part-of-Speech (PoS) Tagging**
   - **Hidden Markov Models (HMMs)**:
     - Viterbi Algorithm: Finds the best sequence of PoS tags.
	     - Dynamic table: Represents states and probabilities.
   - **Conditional Random Fields (CRFs)**:
     - Feature templates include word position, neighboring tags, and specific word patterns.
	     - parameters like state has to be A and prev state has to be B
     - Feature functions
	     - when you assign specific values to A and B
	 - BIO

### 3. **Parsing**
   - **Context-Free Grammar (CFG) and Probabilistic CFG (PCFG):**
     - Transforming grammars into Chomsky Normal Form (CNF).
     - Using CKY Algorithm for parsing:
       - Dynamic tables to calculate probabilities and generate parse trees.
       - Identifying syntactic ambiguities and resolving them.
   - ~~**Dependency Parsing:** Focuses on syntactic relationships between words.~~
   - We use **Constituent Trees**

### 4. **Coreference Resolution**
   - **Mention Pair Models:**
     - Closest-first and best-first strategies for identifying coreferences.
     - Training examples with positive and negative pairs.
   - **CRF for Coreference:** Incorporates features like pronouns, distances, and syntactic patterns.
   - ==practice this==

### 5. **Named Entity Recognition and Classification (NERC)**
   - **Regular Expressions for NERC:**
     - Patterns for extracting entities like persons (PER), locations (LOC), and monetary values (MON).
     - Use of linguistic tags and external resources for genericity.
   - **BIO Tagging with CRFs:**
     - Feature templates involving current and neighboring words, states, and prefixes/suffixes.

- for NERC, we are interested in individual types of NE. for NPs we don't care

### 6. **Word Sense Disambiguation (WSD)**
   - **Simplified Lesk Algorithm:**
     - Determines the best WordNet synset based on overlapping contexts.
     - Requires additional information like synonyms and collocations.
     - ==practice this==
   - **Support Vector Machines (SVMs) and CRFs:**
     - Used to learn WSD models with features like surrounding words and syntactic dependencies.

### 7. ~~**Temporal Expressions**~~
   - **Representation Formalisms:**
     - CFG or semantic grammars extended with semantic features.
     - Lambda calculus for time interval analysis.

### 8. ~~**Semantic Parsing**~~
   - **Semantic Role Labeling:**
     - Identifying agent, patient, and other semantic roles using annotated corpora.
   - **Lambda Calculus and Semantic Features:**
     - Extending CFGs to include semantic information for tasks like dialogue understanding.

### 9. **Algorithms and Tools to Review**
   - Viterbi Algorithm for HMM.
   - CKY Algorithm for CFG/PCFG parsing.
   - Lesk Algorithm for WSD.
   - Feature engineering for CRFs.
   - Basics of FSTs and FSAs for morphological analysis.

## Suggested Study Approach
1. **Practice Problems:** Focus on exercises involving algorithms like Viterbi, CKY, and Lesk.
2. **Understand Concepts:** Review core concepts like CFGs, CRFs, HMMs, and FSTs.
3. **Work Through Dynamic Tables:** Practice building and interpreting dynamic tables for algorithms.
4. **Regular Expressions:** Write and refine patterns for tasks like NERC and morphological analysis.
5. **Annotated Corpora:** Review annotated examples for coreference and semantic role labeling.


HMMs = generative
CRFs = discriminative

generative = maximizing *joint* probability P(A,B)
	find a sentence that best fits in the ruleset
discriminative = maximizing *conditional* probability P(A|B)
	classification
	e.g. mention pair model