# Customer Segmentation with K-Means

Splitting 3,900 retail customers into segments a marketing team can act on, and then testing
whether those segments are **real structure in the data or just a way of cutting it up**.

![Segment profiles](images/segment_profiles.png)

📄 **[Full report (PDF)](reports/customer_segmentation_report.pdf)** · 📊 **[Slide deck (PDF)](reports/customer_segmentation_slides.pdf)** · 📓 **[Notebook](customer_segmentation.ipynb)**

## TL;DR

- K-Means (k = 4) on **age, purchase amount, previous purchases and review rating** gives four
  evenly sized segments that are easy to describe, such as *"younger, higher spend, dissatisfied"*.
- A random-noise baseline shows the dataset has **no natural clusters**. The silhouette score
  on the real data matches uniform random data at every *k*, and rerunning with different seeds
  gives different segments (median ARI 0.47).
- The exploratory analysis agrees. Once raw counts are converted to shares, **gender,
  subscription status, age and spend level don't predict what or how often customers buy**
  (no chi-square test is significant at 5%).
- **Takeaway:** the segments are still usable as a simple age × spend × satisfaction targeting
  grid, but they are a convenient split of the customers, not customer types found in the
  data. On this dataset, a plain rule-based split would work just as well.

## Dataset

[Customer Shopping Trends](https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset)
(Kaggle, synthetic). It has 3,900 customers and 18 columns covering demographics, purchase
amount, product category, season, review rating, subscription status, discount use and
purchase history. There are no missing values or duplicates. A copy is in
[`data/shopping_trends.csv`](data/shopping_trends.csv).

## Approach

1. **Clean.** Convert column names to snake_case.
2. **Explore.** Compare purchase patterns across products, seasons, gender, subscription, age
   and spend level, using shares within each group and chi-square tests. Then check the numeric
   features, which are all close to uniform and nearly uncorrelated.
3. **Pick features.** Use the four numeric features, scaled with `StandardScaler`. I kept purchase
   amount (USD) and previous purchases (a count) as separate features rather than adding them
   together, since the sum would have no clear unit.
4. **Choose *k*.** Use the elbow method, the silhouette score, and a **uniform-noise baseline**
   (the same pipeline run on random data with the same shape).
5. **Profile.** Show the standardized centroids as a heatmap, plot the segments in PCA space,
   and compare segments on variables that were *not* used for clustering (subscription,
   discounts, category).
6. **Validate.** Check stability across 10 random seeds with the Adjusted Rand Index.

## Results

### Exploratory analysis: who buys what?

The customer base is uneven (68% men, 73% non-subscribers), so raw counts can mislead:

![Category by gender](images/eda_category_gender.png)

Men seem to dominate every category, but within each gender the category mix is identical
(p = 0.90). The same happens elsewhere:

| Comparison | Finding | p-value |
|---|---|---|
| Item × season | 25–54 purchases per cell (≈39 expected), no reliable seasonal pattern | 0.29 |
| Category × gender | Identical category mix once converted to shares | 0.90 |
| Purchase frequency × subscription | Subscribers don't buy more often | 0.69 |
| Category × age group | Borderline: 46–60 year-olds buy a bit more footwear (19% vs ~14%) | 0.053 |
| Purchase frequency × spend level | Identical at every spend level | 0.96 |

<details>
<summary>More EDA charts</summary>

![Top products](images/eda_top_products.png)
![Subscription and frequency](images/eda_subscription_frequency.png)
![Item by season](images/eda_season_items.png)
![Age groups](images/eda_age_groups.png)
![Spend and frequency](images/eda_spend_frequency.png)

</details>

### The segments

| # | Segment | Avg age | Avg purchase | Avg rating | Customers | Suggested action |
|---|---|---|---|---|---|---|
| 0 | Older, higher spend, satisfied | 56 | $82 | 4.0 | 966 | VIP perks, early access, referrals |
| 1 | Younger, lower spend, satisfied | 31 | $55 | 4.4 | 969 | Bundles and cross-sell to grow basket size |
| 2 | Younger, higher spend, dissatisfied | 33 | $65 | 3.1 | 986 | Service recovery: follow up on reviews |
| 3 | Older, lower spend, dissatisfied | 56 | $38 | 3.5 | 979 | Low-cost win-back offers and feedback surveys |

Previous purchases barely differs between segments (every centroid is within ±0.2 SD of the
average), so the split is driven by age, spend and rating.

### Are the segments real?

![Choosing k](images/choosing_k.png)

The elbow curve has no bend, and the silhouette score (around 0.19–0.23) follows the
uniform-noise baseline almost exactly. The PCA projection shows the same thing: one continuous
cloud of customers, divided into regions.

![Segments in PCA space](images/segments_pca.png)
![PCA loadings](images/pca_loadings.png)

Two more checks agree:
- **Stability:** the median Adjusted Rand Index across random seeds is **0.47**. The data can be
  split in several near-equal ways, and the random start decides which split you get.
- **External validity:** subscription rate, discount use, gender mix and category mix are
  almost identical across segments.

I think this is the most useful finding in the project. K-Means **always** returns *k*
clusters, even when the data has none, so comparing against a noise baseline is a quick way to
catch that before the segments reach a stakeholder.

## Limitations and next steps

- **The dataset is small.** It has 3,900 customers with a single transaction each and no dates,
  so it can't show how behaviour changes over time. Cross-tabs get thin quickly (around 39
  purchases per item-season cell), which limits the power to detect small effects, and the
  results may not generalise to a larger or real customer base.
- **The data is synthetic.** Its features are independent and uniformly distributed, so no
  clustering method would find natural groups in it. The pipeline is ready to run on real
  transaction data.
- **Categorical features are ignored.** K-Means needs numeric inputs. **K-Prototypes** or Gower
  distance with hierarchical clustering could include category, season and payment method.
- **RFM segmentation.** `frequency_of_purchases` could be turned into purchases per year to build
  Recency-Frequency-Monetary segments, which is the standard approach in retail.

## Run it yourself

```bash
git clone https://github.com/RatanaSovann/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
jupyter notebook customer_segmentation.ipynb
```

## Project structure

```
├── customer_segmentation.ipynb   # full analysis (executed, with outputs)
├── data/shopping_trends.csv      # dataset
├── images/                       # figures (generated by the notebook)
├── reports/
│   ├── customer_segmentation_report.pdf   # written report
│   └── customer_segmentation_slides.pdf   # slide deck
└── requirements.txt
```

## Tech

Python · pandas · NumPy · scikit-learn (KMeans, PCA, silhouette, ARI) · Matplotlib · SciPy (chi-square)
