# Amazon Return Risk Predictor

> **The insight:** A product can have 4.6 stars and still get returned 35% of the time.  
> Star ratings don't predict returns — but the *gap between what customers write and what they rate* does.

🔍 **[Live Demo](https://amazon-return-risk.streamlit.app/)** | Built with Python, XGBoost, VADER, Streamlit

---

## The Problem

Amazon loses billions annually to product returns. The current system relies heavily on star ratings as a quality signal — but star ratings are inflated. Customers consistently rate products higher than their review text suggests.

This project challenges that assumption.

---

## The Hypothesis

If a customer writes a lukewarm or negative review but gives 4-5 stars, that mismatch is a stronger return signal than the star rating alone.

**Finding:** 5-star reviews have an average sentiment-rating mismatch of +0.316, meaning customers systematically write less positively than they rate. 1-star reviews show the opposite: -1.033, meaning customers write even more negatively than their rating suggests.

---

## Approach

**1. Sentiment extraction**  
Used VADER to score the emotional tone of 34,660 Amazon Electronics reviews on a -1 to +1 scale.

**2. Mismatch score engineering**  
Computed the gap between normalized star rating and VADER sentiment score. This is the core feature.
