- naïve semantic text similarity
	- the "naïve" indicates that we are applying rather basic processing steps on the sentences without deciding which might be important
	- and allowing our model to learn and report the most important
- we created features using permutations of processing pipelines
	- we perform unbiased feature analysis using random forests
- like I said, the naïve approach requires no knowledge of the corpus and does not include hand picked features
- this permutation strategy leads to 520 possible processing pipelines
	- here's an example
- with these 520 permutations, we apply each of 4 different similarity metrics
- and this allows us to let the data guide the features we use

- we stuck mostly to the processing steps we used throughout the course
- this is powerful because it demonstrates these simple approaches can be joined together to create powerful insights
- the additions
	- we also added some steps like character extraction, n-gramming and word-pairs
	- these were chosen based on the methodology used by the winning team in the 2012 STS competition from UKP
	- we also added some new metrics like cosine, euclidean and manhattan distance, all based on vectorized word membership

- clearly one of the most important parts of the methodology is the feature extraction
- here I show the various pathways that a sentence may be processed. any path from string to end is used, as long as a step is used at max 1 time
	- e.g. we might remove stopwords first, or chunk NEs first, which might lead to different results, such as including "the" in "the european union"
- to do this I built small functions with typed inputs and outputs and a function that validates that the possible permutation begins with a string and ends with valid output, like POS tags, characters or ngrams, and that the input/outputs agree at each step

- this leads to a massive quantity of computations, but I was able to make it run efficiently through the use of dynamic programming, leveraging the built-in functools.cache
- as you can imagine, some changes in ordering often result in similar or identical outputs, and this speeds up computation significantly
- to be specific, these 2080 features can be extracted for the 5000 sentence pairs across train and test in ~15 minutes

- for our model we chose a random forest
	- this was the right choice because we were particularly interested in feature importance analysis
	- but it's also worth noting that random forests' baggins strategy allows it to discover local patterns and surface importance even across lesser used features
	- it also handles high-dimensional feature space well
	- and it's relatively resistant to overfitting

- when it came time to perform feature selection, we inspected the feature importances of the model trained on all features, and from here we could determine which pipelines were the most prevalent, and we could retrain on subsets of features

- we evaluated the model in a few ways
- to start, we performed 5-fold cross validation to baseline the model and check that random forest was the right model to choose
- when performing feature selection, we did this on a validation set resulting from an 80-20 split
- and we reserved the test datasets for final evaluation
- we created a custom pearson correlation scorer which followed the sklearn style so we compare results directly with published results

results

- as you can see, across all datasets, the model with all features performed best
- to analyze the importance of syntactic and lexical features, we left these steps out for some models and observed the drop in performance.
- we saw that n-grams were used in many of the top feature pipelines, so we decided to analyze a without n-grams feature set too

- to make it easier to understand the importance of each step category, I've plotted the performance relative to the model with all features, for each dataset
- now it's clear that lexical features plays the most important role
- removing syntactic features actually often led to better performance, but for the full joined dataset, it was valuable to include

here's some key takeaways from the results
- the top 10 features account for ~50% of total importance
- 60% of features have non-zero importance
- and when we trained a model using just the top 500 features, we already saw a slight performance drop
- as we saw graphically before, the model trained on a combined feature set performed best


...

conclusions
- we performed comprehensive feature analysis by generating 2080 features across key categories
- we used random forest models to analyze which features were most important

## questions
- how did you **ensure that a method wasn't used where it shouldn't during the permutation pipeline**, like removing stopwords on pos tags or something?

- why did you choose to include **a model without n-grams**?

- **how did you exclude lexical features or semantic features** if the pipeline took a long time to run?

- why do you think **excluding syntactic features improved the score** on the SMTnews and OnWN?