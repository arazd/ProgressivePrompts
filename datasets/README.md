## Continual Learning benchmark data

Due to GitHub space limitations, we only uploaded one CL benchmark dataset - Amazon Reviews, since it's not available through HuggingFace. Please unzip the dataset file for usage.

To access the rest of CL datasets, you can either:
* Access them through their HuggingFace identifiers in our training script (AG: ag_news, Yahoo: yahoo_answers_topics, DbPedia: dbpedia_14, Yelp: yelp_review_full). Note that for Yahoo dataset we filtered rows with empty text fields following Zhang et al. (non-empty row idx are saved under "good_ids_yahoo").
* Download them from Zhang et. al., 2015 [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq) and put corresponding folders into ```datasets/src/data/```
