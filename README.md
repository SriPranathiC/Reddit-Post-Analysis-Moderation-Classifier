# Reddit Post Analysis and Moderation Classifier

📌***Project Overview***

Reddit communities receive large volumes of posts daily, including a significant portion of NSFW or policy-violating content that often goes undetected by automated filters or moderators.
This project builds an intelligent moderation classifier that identifies such risky posts using a hybrid approach:

Classical ML (Random Forest) for fast, low-cost, large-scale classification
LLM (Gemini 2.5-Flash) for targeted high-accuracy classification on ambiguous posts
The combination significantly improves recall and overall moderation accuracy while keeping computational cost low.

📊***Dataset Description***

[Link](https://www.kaggle.com/datasets/unanimad/dataisbeautiful) to the Dataset.

The dataset comprises 190,853 Reddit posts from the r/dataisbeautiful subreddit, with 12 columns.

**id** - Unique identifier for each Reddit post <br>
**title** - Text content/title of the post <br>
**score** -	Upvotes minus downvotes <br>
**author** -	Username of the post creator <br>
**author_flair_text	Flair** - associated with the author <br>
**removed_by** -	Who removed the post (moderator, automod, admin), or NA if not removed <br>
**total_awards_received** - 	Number of awards received <br>
**awarders** -	Users who awarded the post <br>
**created_utc** -	Post creation timestamp <br>
**full_link**	- URL of the post <br>
**num_comments** -	Number of comments <br>
**over_18**	- Target label — 1 if NSFW, 0 otherwise

The dataset contained a large number of NSFW posts that were not removed by Reddit moderators, automod, or the creator indicating potential gaps in moderation.
This made the dataset ideal for building a post-classification moderation system.

📌***Process***
* Implemented a complete data preparation pipeline, including handling missing values, imputing NaNs, and normalizing timestamps.
* Performed outlier analysis on key numerical fields such as score, total_awards_received, and num_comments.
* Conducted Exploratory Data Analysis (EDA) involving score distribution plots, comment activity patterns, temporal posting trends, and numerical feature correlations.
* Analyzed how NSFW posts differ in engagement behavior compared to non-NSFW posts.
* Examined user-related data to identify patterns that contribute to moderation gaps.
* Applied text preprocessing on post titles, including cleaning operations such as removing noise, special characters, and irrelevant text elements.
* Initially trained a Random Forest model due to its speed, scalability, and effectiveness with TF-IDF text representations.
* Identified that RF struggled with context-heavy or subtle NSFW patterns, as it lacks deeper semantic understanding.
* Since the dataset was largely text-driven, incorporating an LLM was the logical next step to capture nuanced meaning.
* Running an LLM on all 190k posts would be computationally expensive and inefficient.
* Implemented a hybrid pipeline, where only RF-misclassified posts (false negatives) were passed to Gemini 2.5-Flash for reevaluation, achieving high contextual accuracy with minimal computational cost.

🌲***Random Forest Baseline (RF)***

The Random Forest model performed extremely well in identifying safe posts but struggled to catch all NSFW ones. It achieved a Recall of 0.60, meaning it correctly detected only 60% of the actual NSFW posts. The confusion matrix: <br>

37957 true negatives - safe posts correctly predicted <br>
129 true positives - NSFW posts correctly predicted <br>
85 false negatives - NSFW posts missed by RF <br>
0 false positives - RF never marked a safe post as NSFW <br>

This shows that RF is very strict and never over-flags content, but because of that strictness, it misses many NSFW posts, especially those with subtle or context-heavy wording.

🤖 ***LLM Enhancement (Gemini 2.5-Flash)***

To fix those misses, only the 85 RF false negatives were sent to the LLM for re-checking. The LLM correctly recovered many of them, improving Recall to 0.71.  <br>
Updated confusion matrix:

37957 true negatives (unchanged)
153 true positives - more NSFW posts correctly caught <br>
61 false negatives - fewer NSFW posts missed <br>
0 false positives - still zero incorrect over-flagging <br>

This hybrid approach substantially improves recall while maintaining perfect precision. It catches far more harmful/NSFW posts with almost no cost increase because the LLM only processes the ambiguous cases, not the full dataset.
