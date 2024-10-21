## Notes

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