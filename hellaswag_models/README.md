# hellaswag_models

This folder contains BERT for hellaswag, namely, by using `run_hellaswag.sh`. There are also instructions below for submitting models to the [HellaSwag leaderboard](https://rowanzellers.com/hellaswag):

# submitting to the leaderboard

Submission is easy! In this folder, I've attached `bertlarge-example-submission` which is an example of what your submission should look like. It should be the same size as the test set in `data/hellaswag_test.jsonl` with your probabilities for each ending `ending0,ending1,ending2,ending3`.

Here's a quick python script to make a submission file - just replace `test_probs` with your predictions.
```
import pandas as pd
import numpy as np
test_probs = np.random.randn(10003, 4)

test_probs_df = pd.DataFrame(test_probs, index=[f'test-{i}' for i in range(test_probs.shape[0])],
                           columns=['ending0', 'ending1', 'ending2', 'ending3'])
test_probs_df.index.name = 'annot_id'
test_probs_df.to_csv('examplesubmission.csv')
```

## Policies
Email Rowan (rowanz at cs.washington.edu) a CSV with your predictions, in the above format. Please include in your email 1) a name for your model, 2) your team name (including your affiliation), and optionally, 3) a github repo or paper link. I'll try to get back to you within a few days, usually sooner. 

* Teams can only submit results from a model once every 7 days by default, though I am happy to waive this restriction around paper deadlines. The reason I have this policy is that in the past, some teams made submissions for [VCR](https://visualcommonsense.com) once every day. 
* I reserve the right to not score any of your submissions if you cheat -- for instance, please don't make up a bunch of fake names / email addresses and send me multiple submissions under those names.