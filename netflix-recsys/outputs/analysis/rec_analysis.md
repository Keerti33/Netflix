# Recommendation Analysis Report

Model: SVD | Users analysed: 1000 | AP@10 > 0: 1 | AP@10 = 0: 999

---

## Success Cases

Users with highest AP@10 (top 1):

### User 1822437 (AP@10 = 0.1111)

**Top-5 rated movies (training):**

| Movie ID | Title | Year | Rating |
|---|---|---|---|
| 14712 | Tomb Raider | 2001 | 3 |

**Top-10 recommendations:**

| Rank | Movie ID | Title | Year |
|---|---|---|---|
| 1 | 13501 | Rocky IV | 1985 |
| 2 | 15575 | Royal Wedding | 1951 |
| 3 | 747 | Saber Marionette J | 1996 |
| 4 | 3165 | Dirty Rotten Scoundrels | 1988 |
| 5 | 5793 | Underworld | 2003 |
| 6 | 17147 | She's the One | 1996 |
| 7 | 8846 | Election | 1999 |
| 8 | 17152 | Next Friday | 2000 |
| 9 | 16969 | Donnie Darko | 2001 |
| 10 | 7055 | Get Shorty | 1995 |

**Era overlap:** Shared decades: 2000s. The model captures this user's era preferences well.

## Failure Cases

Users with AP@10 = 0.0 (5 sampled):

### User 2052225
- **Type:** Near cold-start (1 training ratings)
- **Test items:** 1
- **Diagnosis:** Standard taste profile.

### User 1039295
- **Type:** Near cold-start (1 training ratings)
- **Test items:** 1
- **Diagnosis:** Standard taste profile.

### User 1738284
- **Type:** Near cold-start (3 training ratings)
- **Test items:** 1
- **Diagnosis:** Standard taste profile.

### User 205023
- **Type:** Near cold-start (2 training ratings)
- **Test items:** 2
- **Diagnosis:** Uniformly high rater (low variance) -- hard to distinguish preferences.

### User 1036389
- **Type:** Near cold-start (2 training ratings)
- **Test items:** 1
- **Diagnosis:** Standard taste profile.

## Era / Decade Analysis

Across 200 users, the average decade overlap between a user's top-10 rated movies and their top-10 recommendations is **87.9%**.

This means the model tends to recommend movies from similar eras as the user's favourites, suggesting it captures temporal taste patterns.
