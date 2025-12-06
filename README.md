# Reddit Post Analysis and Moderation Classifier

📌***Project Overview***

Reddit communities receive large volumes of posts daily, including a significant portion of NSFW or policy-violating content that often goes undetected by automated filters or moderators.
This project builds an intelligent moderation classifier that identifies such risky posts using a hybrid approach:

Classical ML (Random Forest) for fast, low-cost, large-scale classification
LLM (Gemini 2.5-Flash) for targeted high-accuracy classification on ambiguous posts
The combination significantly improves recall and overall moderation accuracy while keeping computational cost low.

📊***Dataset Description***

[Link to Dataset](https://www.kaggle.com/datasets/unanimad/dataisbeautiful)

The dataset comprises 190,853 Reddit posts from the r/dataisbeautiful subreddit, with 12 columns.

**id** - Unique identifier for each Reddit post

**title** - Text content/title of the post

**score** -	Upvotes minus downvotes

**author** -	Username of the post creator

**author_flair_text	Flair** - associated with the author

**removed_by** -	Who removed the post (moderator, automod, admin), or NA if not removed

**total_awards_received** - 	Number of awards received

**awarders** -	Users who awarded the post

**created_utc** -	Post creation timestamp

**full_link**	- URL of the post

**num_comments** -	Number of comments

**over_18**	- Target label — 1 if NSFW, 0 otherwise

The dataset contained a large number of NSFW posts that were not removed by Reddit moderators, automod, or the creator indicating potential gaps in moderation.
This made the dataset ideal for building a post-classification moderation system.

📌***Process***

A complete data preparation pipeline was implemented, starting with cleaning operations such as handling missing values, imputing NaNs, normalizing timestamps, and performing outlier analysis on numerical fields like score, total_awards_received, and num_comments. Exploratory Data Analysis (EDA) included visualizing score distributions, comment activity patterns, temporal posting trends, correlation analysis among numerical features, and understanding how NSFW posts differ in behavior with respect to engagement metrics. User-related data was examined to identify patterns in moderation gaps. For text preprocessing, the post titles were cleaned by removing noise, converting to lowercase, eliminating stop words, performing stemming, and applying TF-IDF vectorization to transform the text into meaningful numerical representations suitable for machine learning. Together, these steps helped reveal hidden structure in the data, identify key sources of moderation inconsistencies, and prepare both numerical and textual features for downstream classification

We first trained a Random Forest model because it is fast, scalable, and handles TF-IDF text features well, but RF struggles with deeper semantic understanding something crucial for NSFW detection. Since our data is primarily text-based and many misclassified posts were context-heavy or subtle, an LLM was a natural next step for better interpretation. However, running an LLM on the entire 190k-post dataset would be computationally expensive and unnecessary. To balance accuracy and cost, we employed a hybrid strategy: only the posts misclassified by RF (i.e., its false negatives) were sent to Gemini 2.5-Flash for reevaluation. This allowed us to obtain the contextual intelligence of an LLM while maintaining a highly efficient pipeline.

🌲***Random Forest Baseline (RF)***

The Random Forest model performed extremely well in identifying safe posts but struggled to catch all NSFW ones. It achieved a Recall of 0.60, meaning it correctly detected only 60% of the actual NSFW posts. The confusion matrix:

37957 true negatives - safe posts correctly predicted

129 true positives - NSFW posts correctly predicted

85 false negatives - NSFW posts missed by RF

0 false positives - RF never marked a safe post as NSFW

This shows that RF is very strict and never over-flags content, but because of that strictness, it misses many NSFW posts, especially those with subtle or context-heavy wording.

🤖 ***LLM Enhancement (Gemini 2.5-Flash)***

To fix those misses, only the 85 RF false negatives were sent to the LLM for re-checking. The LLM correctly recovered many of them, improving Recall to 0.71. Updated confusion matrix:

37957 true negatives (unchanged)

153 true positives - more NSFW posts correctly caught

61 false negatives - fewer NSFW posts missed

0 false positives - still zero incorrect over-flagging

This hybrid approach substantially improves recall while maintaining perfect precision. It catches far more harmful/NSFW posts with almost no cost increase because the LLM only processes the ambiguous cases, not the full dataset.
