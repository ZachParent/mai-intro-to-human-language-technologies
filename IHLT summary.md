# subareas of linguistics
[[01-introduction.pdf#page=7&selection=17,0,17,10|Definitions]]
- phonetics
- morphology
- syntax
- semantics
- pragmatics
## [[02-doc-structure.pdf#page=7&selection=12,8,12,20|tokenization]]
- split plain text into basic units
- hand-crafted, supervised, unsupervised methods
- hardest problems: abbreviations and acronyms
- **Punkt** is a common tokenization algorithm

[[maximum likelihood estimator (MLE)]]
- can be used for language identification on unigrams
![[02-doc-structure.pdf#page=30&rect=74,128,324,192]]

## [[03-morphology.pdf#page=6&selection=14,13,14,23|morphology]]
- stems (e.g. 'work', 'make')
- affixes: always occur combined with other morphemes, including prefixes, suffixes, infixes and cirumfixes
### [[03-morphology.pdf#page=14&selection=32,8,32,36|finite state automata (FSAs)]]
FSAs for lexical recognition
![[03-morphology.pdf#page=18&rect=112,31,292,186]]
- FSAs can be useful for recognizing words
- FSAs are not able to output a word analysis
### [[03-morphology.pdf#page=23&selection=15,0,15,31|Finite state transducers (FSTs)]]
- these map from one language to another
- `a:d` maps from an input d to an output a
### [[03-morphology.pdf#page=25&selection=15,0,15,31|FSTs for morphological analysis]]
![[03-morphology.pdf#page=25&rect=51,45,309,193]]
- T_lex: a FST that computes morphotactics
	- Ex rule: *Reg Nˆs#* → **Reg N+N+PL**
	- Ex use : *dogˆs#* → **dog+N+PL**
- T_inter: FSTs each computing a spelling rule (orthographic regularization)
- Ex: *-{z,x,s,sh,ch}es* → **-{z,x,s,sh,ch}ˆs#**
![[03-morphology.pdf#page=26&rect=183,23,329,94]]
### [[03-morphology.pdf#page=36&selection=11,0,11,16|Spell correctors]]![[03-morphology.pdf#page=36&rect=54,17,358,210]]
## [[04-POS-tagging.pdf#page=1&selection=9,3,9,14|POS tagging]]: ==next==