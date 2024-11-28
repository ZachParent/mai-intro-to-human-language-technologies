# [10-Coreference-resolution](ihlt/slides/10-Coreference-resolution.pdf)

## Goal of coreference resolution

> ([10-Coreference-resolution, p.4](ihlt/slides/10-Coreference-resolution.pdf#page=4&selection=21,0,22,50))
> Determining which mentions in a discourse refer to the same real world entity, property or situation. #definition

![10-Coreference-resolution, p.5](ihlt/slides/10-Coreference-resolution.pdf#page=5&rect=59,61,348,125)
## Noun-Phrase Coreference Resolution

> ([10-Coreference-resolution, p.7](ihlt/slides/10-Coreference-resolution.pdf#page=7&selection=21,0,21,47))
> Most commonly investigated type of coreference.

In this course Coreference Resolution <-> Noun-Phrase Coreference Resolution
### Motivation
---
**Examples**

> ([10-Coreference-resolution, p.8](ihlt/slides/10-Coreference-resolution.pdf#page=8))
> - Information extraction
> 	- Ex: Extract organizations in which a person has worked.
> - Automatic summarization 
> 	- Ex: Find relevant sentences related to a particular person. 
> - Question Answering: 
> 	- Ex: Answer factual questions such as *Where has Mary Doe been working?*

### Types of referent (Ways of referring)

> ([10-Coreference-resolution, p.10](ihlt/slides/10-Coreference-resolution.pdf#page=10&selection=21,0,42,34))
> - Referring expressions:
> 	- world knowledge can be necessary for the resolution
> 	- Ex: Laporta, the president, FC Barcelona president Joan Laporta
> 	- Ex: Lionel Messi, the young Argentine
> - Pronouns: linguistic information can be useful for the resolution (number, genre, grammatical constraints)
> 	- Ex: Laporta said: ”I will answer”
> 	- Ex: the president said: ”I will answer

## [Coreferent, Anaphor, Cataphor](ihlt/slides/10-Coreference-resolution.pdf#page=11&selection=19,0,19,29)

> ([10-Coreference-resolution, p.11](ihlt/slides/10-Coreference-resolution.pdf#page=11&selection=22,0,40,45))
> - **Coreferent**: two mentions refer to the same world entity #definition 
> - **Anaphora**: a mention (anaphor) refers to a preceding mention (antecedent) and the interpretation of the anaphor depends on the antecedent. #definition 
> 	- **John Smith** had been writting for months. **He** ended up sleeping on the bed. 
> - Cataphora: the antecedent occurs after the anaphor (cataphor)
> 	- **He** had been writting for months. **John Smith** ended up sleeping on the bed.
> 
> - Anaphor is more frequently used than cataphor

![10-Coreference-resolution, p.12](ihlt/slides/10-Coreference-resolution.pdf#page=12&rect=58,75,349,211)
> (1) Not a real world entity - a quantified concept
> (2) Not the same entity
> (3) They don’t need each other to be interpreted

## Methodology of a Coreference Solver

![10-Coreference-resolution, p.14](ihlt/slides/10-Coreference-resolution.pdf#page=14&rect=85,172,323,199)
### 1. Mention Detection
a. POS-tagging, NERC and parsing (constituent parsing or dependence parsing)
b. ([10-Coreference-resolution, p.16](ihlt/slides/10-Coreference-resolution.pdf#page=16&selection=20,0,31,24))
> Recursiverly visiting the parse tree, accept the following as mention
> - Pronouns (filter out pleonastic pronouns, e.g., It is raining)
> - Proper names
> - Maximal noun-phrase projections
> - Coordinated noun phrases

[Examples of maximal NP projections with constituent parses:](ihlt/slides/10-Coreference-resolution.pdf#page=17&selection=18,0,18,59)
![10-Coreference-resolution](ihlt/slides/10-Coreference-resolution.pdf#page=17&rect=84,99,330,174)
Drop out NPs sharing the same head. (head: essentially, right-most noun in first sub-constituent)

> ([10-Coreference-resolution, p.20](ihlt/slides/10-Coreference-resolution.pdf#page=20&selection=34,13,34,18))
> CO/NC -> Coreferent, non-coreferent

### 2. Mention-Pair model

> ([10-Coreference-resolution, p.20](ihlt/slides/10-Coreference-resolution.pdf#page=20))
> Learn a classifier of mention pairs. Ex: 
> - Decision Trees
> - Rule induction (RIPPER) [Ng & Cardie, 02] 
> - Maximum Entropy [Denis & Baldrige, 07], [Ji et al., 05] 
> - SVMs [Yang et al., 06]

> ([10-Coreference-resolution, p.20](ihlt/slides/10-Coreference-resolution.pdf#page=20&selection=56,0,75,29))
> Generate chains. Ex:
> - Closest-first strategy [Soon et al., 01]
> - Best-first strategy [Ng & Cardie, 02][Bengtson & Roth, 08]
> - Clustering [Klenner & Ailloud 2008]...
> - Global optimization (ILP) [Klenner, 07][Finkel & Manning, 08]
> - Graph partitioning [McCallum & Wellner,05][Nicolae & Nicolae, 06][Sapena et al, 10

#### Creating training examples from annotated corpus

#### Closest-first strategy
> ([10-Coreference-resolution, p.21](ihlt/slides/10-Coreference-resolution.pdf#page=21&selection=23,0,75,1))
> [Soon et al, 01] 
> - the classifier is biased to select the closest antecedent 
> - $e^+=⇒(m_i, m_j)$ Anaphor $m_j$ and closest antecedent $m_i$ 
> - $e^−=⇒∀k : i < k < j$ : $(m_k , m_j)$

![10-Coreference-resolution, p.24](ihlt/slides/10-Coreference-resolution.pdf#page=24&rect=130,129,274,180)

#### Best-first strategy
> ([10-Coreference-resolution, p.21](ihlt/slides/10-Coreference-resolution.pdf#page=21&selection=77,0,151,1))
> [Ng & Cardie, 02]
> - the classifier is biased to select the best antecedent
> - $e^+=⇒(m_i , m_j)$ Anaphor $m_j$ and closest antecedent $m_i$
> 	- *but for non-pronominal anaphor $m_j$ select the closest non-pronominal antecedent $m_i$* 
> - $e^−=⇒∀k : i < k < j$ : $(m_k , m_j)$ where $m_k$ is not in the same chain that $m_j$

![10-Coreference-resolution, p.25](ihlt/slides/10-Coreference-resolution.pdf#page=25&rect=123,128,278,179)

We are looking for counter examples in between the noun-phrase and its antecedent

### Features functions for a mention pair
![10-Coreference-resolution](ihlt/slides/10-Coreference-resolution.pdf#page=23&rect=57,40,362,208)

## [Exercise 1](ihlt/slides/exercises/10-Coreference-resolution-exercises.pdf#page=24&selection=5,0,5,10)

![10-Coreference-resolution-exercises, p.16](ihlt/slides/exercises/10-Coreference-resolution-exercises.pdf#page=24&rect=54,13,339,211)
## [Exercise 2](ihlt/slides/exercises/10-Coreference-resolution-exercises.pdf#page=26)

### Closest-first
The closest antecedent is independent of the final probability, if you use a probabilistic classifier
> ([10-Coreference-resolution-exercises, p.20](ihlt/slides/exercises/10-Coreference-resolution-exercises.pdf#page=30&selection=7,0,12,29))
> m7 antecendent using closest-first strategy assuming a coreference threshold of 0.6.
![10-Coreference-resolution-exercises, p.20](ihlt/slides/exercises/10-Coreference-resolution-exercises.pdf#page=30&rect=91,83,319,178)
![10-Coreference-resolution-exercises, p.20](ihlt/slides/exercises/10-Coreference-resolution-exercises.pdf#page=31&rect=90,80,317,179)

### Best-first
![10-Coreference-resolution, p.27](ihlt/slides/10-Coreference-resolution.pdf#page=29&rect=55,101,353,185)
## Drawbacks of the Mention-Pair model
![10-Coreference-resolution-exercises, p.21](ihlt/slides/exercises/10-Coreference-resolution-exercises.pdf#page=34&rect=89,78,321,180)
> ([10-Coreference-resolution, p.27](ihlt/slides/10-Coreference-resolution.pdf#page=29&selection=21,0,21,20))
> Lack of information.
> ([10-Coreference-resolution, p.27](ihlt/slides/10-Coreference-resolution.pdf#page=29&selection=23,0,23,32))
> Contradictions in classification.
## [Mention-ranking models](ihlt/slides/10-Coreference-resolution.pdf#page=31&selection=20,0,20,22)

> ([10-Coreference-resolution, p.30](ihlt/slides/10-Coreference-resolution.pdf#page=32&selection=22,0,22,36))
> Ex: rankers [Denis and Baldrige, 08]

> ([10-Coreference-resolution, p.30](ihlt/slides/10-Coreference-resolution.pdf#page=32&selection=24,0,65,1))
> Example = $(m_i , α_i , A_i)$, where $α_i$ is the first antecedent of $m_i$ (the non-pronominal one if $m_i$ is non-pronominal) and $A_i$ is the set of non-antecedents in a context of 2 sentences around αi

### Example

![10-Coreference-resolution, p.30](ihlt/slides/10-Coreference-resolution.pdf#page=32&rect=107,53,322,95)
$m_0 ... m_1 ... m_2 | ... m_3 | ... m_4 ... m_5$

$m_i = m_5$

| $\alpha_i$ | $P(m_i, \alpha_i, A_i)$             |
| ---------- | ----------------------------------- |
| $m_0$      | $P(m_5, m_0, \{m_1, m_2, m_3\})$    |
| $m_1$      | $P(m_5, m_1, \{m_0, m_2, m_3\})$    |
| $m_2$      | $P(m_5, m_2, {m_0, m_1, m_3})$      |
| $m_3$      | $P(m_5, m_3, {m_0, m_1, m_2, m_4})$ |
| $m_4$      | $P(m_5, m_4, {m_3})$                |
> ([10-Coreference-resolution, p.31](ihlt/slides/10-Coreference-resolution.pdf#page=33&selection=22,0,25,48))
> - Pros: take profit of decisions involving all the candidate antecedents.
> - Cons: always pick an antecedent from the candidates, although the mention in course is not anaphoric.

## [Entity-Mention model](ihlt/slides/10-Coreference-resolution.pdf#page=35&selection=20,0,20,20)

> ([10-Coreference-resolution, p.33](ihlt/slides/10-Coreference-resolution.pdf#page=35&selection=26,0,31,50))
> Partial entity: a set of mentions considered coreferent during the resolution Each partial entity is represented as the set of features of its mentions. Each partial entity has its representative mention

![10-Coreference-resolution, p.33](ihlt/slides/10-Coreference-resolution.pdf#page=35&rect=105,18,303,108)
### [Stanford Easy-first approach](ihlt/slides/10-Coreference-resolution.pdf#page=37&selection=16,0,16,27)

![10-Coreference-resolution, p.35](ihlt/slides/10-Coreference-resolution.pdf#page=37&rect=66,2,347,220)

> ([10-Coreference-resolution, p.35](ihlt/slides/10-Coreference-resolution.pdf#page=37&selection=32,20,32,31))
> **appositives**: a noun that immediately follows and renames another noun in order to clarify or classify it. #definition 


