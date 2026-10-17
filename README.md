# KNN From Scratch: Classifying Spotify Genres

**Can a machine hear the difference between EDM and rock?**
This project implements the **K-Nearest Neighbors (KNN)** classifier from scratch with NumPy and uses it to predict the genre of Spotify tracks (EDM, Latin, Pop, R&B, Rap, Rock) purely from their audio features, such as danceability, energy, tempo and loudness. The from-scratch model is then benchmarked against scikit-learn.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243)
![Notebook](https://img.shields.io/badge/format-Jupyter%20Notebook-F37626)

---

## Highlights

- **KNN built from scratch**: Euclidean and Manhattan distance functions plus a full `KNNClassifier` class (`fit`, `predict`, `score`) using only NumPy.
- **End-to-end workflow**: data loading, exploration, cleaning, balancing, splitting, scaling, tuning and evaluation.
- **6-class problem** on 900 balanced songs (150 per genre) described by 10 audio features.
- **Benchmarked against scikit-learn** with 10-fold cross-validation over K = 1 to 40.
- **Bonus experiments**: predict the genre of your own song, distance-weighted voting, and an analysis of the most confused genres.

## Results at a glance

| Experiment | Setting | Test accuracy |
|---|---|---|
| First run (scratch KNN) | K=5, Euclidean | **41.11%** |
| K sweep (scratch KNN) | best K=11, Euclidean | 46.67% |
| Metric comparison | K=5, Manhattan | 44.44% |
| Metric comparison | K=11, Manhattan | **48.33%** |
| Final head-to-head | K=37, scratch KNN | 45.00% |
| Final head-to-head | K=37, scikit-learn | 44.44% |

Random guessing across 6 genres would score about 16.7%, so the model is clearly learning something, but genre is far from cleanly separable using these features alone.

**Per-genre F1 (K=5, Euclidean):** rock 0.62, edm 0.55, rap 0.49, latin 0.38, r&b 0.33, **pop 0.13**. Pop is the hardest genre: it sits in the middle of the feature space, and the most confused pairs in the final comparison were rock to pop (10 songs), EDM to pop (8) and Latin to pop (8).

![Confusion matrix](images/05_confusion_matrix.png)

## Dataset

- **Source:** the [TidyTuesday Spotify songs dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21) (2020-01-21), around 32,800 tracks from six playlist genres.
- **Loaded automatically** from GitHub when the notebook runs, so no manual download is needed (an internet connection is required).
- **Features used (10):** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`.
- **Target:** `playlist_genre`.

### Preprocessing
1. Drop duplicate tracks by `track_id` (32,833 rows down to 28,356 unique tracks), so the same song cannot appear in both train and test.
2. Balance the classes by sampling 150 songs per genre (`random_state=42`), giving 900 songs.
3. Encode genre names with `LabelEncoder`.
4. Split 80/20 with stratification (720 train / 180 test).
5. Standardise with `StandardScaler`, fitted on the training set only to avoid data leakage.

## Exploring the data

| | |
|---|---|
| ![Feature distributions](images/02_feature_distributions.png) | ![Genre fingerprints](images/03_genre_fingerprints.png) |
| Feature distributions by genre | Average audio fingerprint per genre |
| ![Danceability vs energy](images/04_scatter_2d.png) | ![Genre distribution](images/01_genre_distribution.png) |
| Danceability vs energy, coloured by genre | Songs per genre in the raw data |

Danceability and energy separate genres best, but pop, Latin, rap and EDM overlap heavily in the high-energy, high-danceability region, which is where most of the model's mistakes occur.

## Implementation

The classifier follows the standard lazy-learning recipe:

```python
class KNNClassifier:
    def __init__(self, k=5, metric='euclidean'): ...
    def fit(self, X, y): ...           # stores the training data, nothing else
    def _predict_one(self, x): ...     # distances -> sort -> top K -> majority vote
    def predict(self, X): ...
    def score(self, X, y): ...         # accuracy
```

For each new song, the model computes its distance to every training song, takes the K closest, and returns the genre with the most votes. Distance functions:

- **Euclidean:** `sqrt(sum((a - b) ** 2))`
- **Manhattan:** `sum(abs(a - b))`

The notebook includes small auto-tests for both distance functions and the classifier. Prediction cost is O(n × d) per song, which is why the scratch version takes noticeably longer than scikit-learn's optimised implementation.

## Tuning and benchmarking

| K sweep (scratch) | 10-fold CV (scikit-learn) |
|---|---|
| ![K sweep](images/06_k_sweep.png) | ![CV sweep](images/08_cv_sweep.png) |

- **K sweep:** K=1 memorises the training data (train accuracy 1.00, test 0.37); larger K smooths the decision boundary. On the test set, K=11 scored best.
- **Cross-validation:** 10-fold CV over K=1 to 40 selected **K=37** (mean CV accuracy 44.72%).
- **Metric comparison:** Manhattan distance beat Euclidean at both K values tested.

![Metric comparison](images/07_metric_comparison.png)

- **Agreement with scikit-learn:** at K=37 the two models predict identically for 95.0% of test songs, with similar accuracy (45.00% vs 44.44%). The notebook's own check uses a 97% threshold, so it flags this result for review; small differences in how ties between neighbours are broken are one possible cause (not verified).

## Getting started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows (Git Bash): source venv/Scripts/activate
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebook
```bash
jupyter notebook Spotify_songs_using_KNN.ipynb
```
Run the cells from top to bottom. The scratch-KNN cells take roughly 15 to 40 seconds each, and the K sweep takes longer because it trains and evaluates the model several times.

## Repository structure
```
.
├── Spotify_songs_using_KNN.ipynb   # full walkthrough: data, model, experiments
├── images/                         # figures used in this README
└── README.md
```
Running the notebook also saves its figures (`01_genre_distribution.png` and so on) to the working directory.

## Limitations

- **Small sample:** only 900 songs and a single 80/20 split, so differences of a few percentage points are within noise.
- **K chosen on the test set:** the scratch K sweep (K=11) and the metric comparison use test accuracy to choose settings, which can make those numbers slightly optimistic. The cross-validation search is the more reliable estimate.
- **Features only describe audio:** playlist genre labels are noisy, and songs often blend genres, which caps what any feature-based classifier can achieve.
- **Scratch KNN is slow:** brute-force distance loops do not scale to the full dataset.

## Ideas for improvement

- Train on more than 150 songs per genre, using a vectorised distance computation to keep it fast.
- Choose K and the distance metric with cross-validation instead of the test set.
- Try feature selection or weighting, since not all 10 features help equally.
- Compare against other classifiers (random forest, gradient boosting) on the same split.

## Acknowledgements

- Dataset: [TidyTuesday](https://github.com/rfordatascience/tidytuesday) Spotify songs (2020-01-21), originally collected through the Spotify API.
- Audio features as defined by Spotify's audio analysis.

## License

Add a license of your choice (for example MIT) in a `LICENSE` file.

---

*Author: [Your Name]*
