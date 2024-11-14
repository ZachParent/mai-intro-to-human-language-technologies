## Notes 2024-10-17

* this time we are forced to use TextServer (only session)
* spoiler alert: using word sense will not work as well
    * best approach is morphology
* if we combine approaches from symset, lemmas, tokens into one model, we will probably get better results
* could use lemma tokens as a fallback
* could combine similarity metrics, such as all the previous methods, and use a model, like KNN or decision tree, to get better results on the project
* could use metadata for features, like document length, parts of speech
* lesk is quite bad, UKB is slightly better
* the wn response is a different format than we are used to, e.g. `10285313-n`
    * to get the symset, use `wn.synset_from_pos_and_offset('n',10285313)`
* textserver python API is not working, but we could use the browser console to hit the web API
* the browser GUI works too
* it's a good idea to cache the results from textserver
* we will decide what to do with tokens that don't have a sense, we could ignore the ones without a sense. if we keep the ones, we will apply the same pre-processing as in session 2 and 3
    * he's being coy, but we should probably not discard the words
    * if the API doesn't work, it will be OK to not use UKB

## Notes 2024-10-24

* We will discuss project today too
* Lab 2024-11-14 will be tutoring for project
* spacy works better for ___ because it uses CNN, not max entropy like NLTK
* CRF is a transfer based model
* it will be worse
    * we are giving less weight to the tokens that matter most. 1 match instead of 2 if a firstname-lastname get joined
    * could help with disambiguation for first names with diff last names
* there is a method for bringing trees together for NLTK to joing first name last names
* idea: could merge these then repeat them n times for n words
* the reason CRFs are so good is they also take into account position
    * they understand that it's impossible to go from `out` to `in` 
* combining CNNs and CRF was the best method before transformers

### Project

* SemEval 2012
* https://smedina-upc.github.io/ihlt/sts/index.html
* so far we've only been consider europal corpus, here we will consider all in the corpus
* trial was given before the competition, can start building models
    * we assume that this is what we have been doing already
* train dataset 
* test dataset -> don't use!
* can use a dictionary, but not more data or greater corpus
* Compare and comment the results achieved by these approaches among them and among the official results.
* Submission
    * Jupyter notebook: sts-[Student1]-[Student2].ipynb
    * Slides: sts-[Student1]-[Student2].pdf
* can use any techniques from class or otherwise, as long as it's before 2012
* can use spacy, but not word embeddings
```
ProjectGrade = 0.1  ∗ Code Effectiveness +
               0.05 * Code Readability and Efficiency +
               0.05 * Use of NLP Libraries and Resources +
               0.4  * Analysis and Representation of Results +
               0.2  * Results + 
               0.2  ∗ Oral Presentation
```
```
Results: 
10	if the pearson is over 10th participant (.7562)
0	if the pearson is under the baseline (.311)
proportional	in other case
```

* We should divide the train set to tune a model that combines signals
* Jaccard distance is not mandatory
* Pearson is mandatory for scoring
![alt text](<Screenshot 2024-10-24 at 3.53.33 PM.png>)

## Notes 2024-11-14

* no need to compare NLTK with Spacy, just that it works
* if we include a particular feature, we should justify how it improves the results
* we can compute similarity in other ways, e.g. cosine similarity
* random forest works pretty well
* maybe try 3 different algorithms, one from each family
* if you take a look at the papers, you'll see some teams considered the datasets
* keep in mind that during the lab session, we have only used one dataset
* but in the corpora, there are other types of datasets
   * we could use different models for different datasets
* we must get top 10% on the all set
* the all set is a concatenation of all the datasets
* feature selection is a good idea
   * feature importance will vary widely