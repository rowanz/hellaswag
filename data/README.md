# data

This folder contains training, validation, and unlabeled test sets for HellaSwag, in `.jsonl` format. Here's what each dataset example contains:

* `ind`: dataset ID
* `activity_label`: The ActivityNet or WikiHow label for this example
* context: There are two formats. The full context is in `ctx`. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in `ctx_b`, and the context up until then is in `ctx_a`. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If `ctx_b` is nonempty, then `ctx` is the same thing as `ctx_a`, followed by a space, then `ctx_b`.
* `endings`: a list of 4 endings. The correct index is given by `label` (0,1,2, or 3)
* `split`: train, val, or test.
* `split_type`: `indomain` if the activity label is seen during training, else `zeroshot`
* `source_id`: Which video or WikiHow article this example came from
