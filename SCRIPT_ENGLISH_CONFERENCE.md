# PRESENTATION SCRIPT — English Conference PPT
## "Performance Analysis of Stacking Ensemble and Explainable AI in Crop Recommendation Systems"

**Presenter:** Stiefanny Dwi Chandra
**Total Slides:** 11
**Estimated Duration:** ~10 minutes

---

## SLIDE 1 — TITLE (~20 seconds)

Good morning/afternoon, ladies and gentlemen. Thank you for the opportunity.

My name is Stiefanny Dwi Chandra. Today, I will be presenting my research titled **"Performance Analysis of Stacking Ensemble and Explainable AI in Crop Recommendation Systems."**

---

## SLIDE 2 — BACKGROUND (~1.5 minutes)

This research is motivated by three key challenges in agricultural AI:

**First, Habits vs Climate Change.** Farmers still often choose crops based on their gut feeling or generational habits. However, weather and soil conditions now change rapidly due to climate change, making traditional decision-making unreliable.

**Second, The Problem of AI's 'Black Box.'** Current AI systems are good at predicting — but they cannot explain the logical reasons behind their predictions. It is like a calculator that gives you an answer without showing the formula.

**Third, Farmers' Crisis of Trust.** Because the system cannot explain its reasoning, farmers become skeptical. Without transparency, AI technology is very difficult to be widely adopted in the field.

This is why we need a system that is not only accurate, but also **explainable and trustworthy**.

---

## SLIDE 3 — PROBLEM STATEMENT (~1 minute)

Based on those challenges, I formulated three research questions:

**First — Model Performance:** How does the Stacking Ensemble model compare to individual boosting models — XGBoost, LightGBM, and CatBoost — in a crop recommendation system?

**Second — Stability:** Does using Stacking Ensemble improve the stability and generalization ability of the model compared to the best single model?

**Third — System Transparency:** How can the SHAP method be used to improve the interpretability of model prediction results — so that farmers can understand and trust the AI's recommendations?

---

## SLIDE 4 — RESEARCH OBJECTIVES (~1 minute)

The objectives of this research are fourfold:

**First, Achieving Maximum Accuracy** — developing a crop recommendation system based on Stacking Ensemble with Logistic Regression as the meta-learner.

**Second, Avoiding Overfitting** — ensuring the AI truly understands natural patterns, not just memorizing training data.

**Third, Translating the Black Box** — transforming complex mathematical formulas into agronomic reasoning that makes sense to farmers.

**Fourth, Improving Stability** — evaluating model stability and generalization using 5-Fold Cross Validation and overfitting analysis.

---

## SLIDE 5 — MAIN COMPONENTS OF RESEARCH (~1 minute)

The research is built upon four main components:

**First, Balanced Dataset** — we use 2,200 land data points with exactly 100 samples for each of 22 crop types. This ensures the AI does not become biased toward any specific crop during training.

**Second, Soil Parameters** — the amounts of Nitrogen (N), Phosphorus (P), and Potassium (K) are used as key features for the AI's consideration.

**Third, Climate Physical Parameters** — we input temperature, humidity, rainfall, and soil pH conditions as additional features.

**Fourth, Stacking Ensemble** — we use a family of AI algorithms based on decision trees — XGBoost, LightGBM, and CatBoost as base learners — combined through Logistic Regression as meta-learner.

---

## SLIDE 6 — RESEARCH METHODOLOGY (~30 seconds)

This slide shows the overall research methodology. The pipeline flows from data collection, through Exploratory Data Analysis, preprocessing, train-test splitting, model development with Cross-Validation, final evaluation on the hold-out test set, and finally SHAP interpretability analysis — both at global and local levels.

---

## SLIDE 7 — DATA ANALYSIS: MODEL PERFORMANCE (~1.5 minutes)

Now let me share the key findings.

**First, Model Performance.** All models achieved very high performance on this dataset. CatBoost achieved the highest test accuracy at 99.77%. However, the Stacking Ensemble also delivers excellent and very stable performance at 99.55% test accuracy.

**Second, Cross-Validation vs Test.** The Stacking Ensemble achieved the highest CV accuracy at 99.60% with a standard deviation of only 0.0039 — meaning it is extremely consistent across different data splits. The gap between CV accuracy and test accuracy is minimal, confirming no overfitting. All models maintain a train-test gap below 2%, indicating strong generalization.

**Third, the Main Evaluation Insight.** Here is a critical finding: the best model in terms of raw accuracy (CatBoost) is NOT the same as the most stable model (Stacking Ensemble). This is because Stacking combines multiple models, reducing individual bias and producing more robust predictions overall.

---

## SLIDE 8 — DATA ANALYSIS: CONFUSION MATRIX & SHAP (~30 seconds)

This slide visualizes the Confusion Matrix and SHAP analysis results. As you can see, the confusion matrix shows near-perfect classification across all 22 crop types, with only 2 misclassifications out of 440 test samples.

---

## SLIDE 9 — KEY FINDINGS OF EXPLAINABLE AI (~1.5 minutes)

Now, the most important contribution of this research — Explainable AI findings:

**Error Rate:** Out of 440 test data points, the AI only made 2 mistakes. And even those mistakes are agronomically logical — it confused rice with jute because both crops require very high water conditions. This shows the AI truly learned meaningful patterns.

**Recommendation Driver:** The SHAP analysis reveals exactly WHY the AI makes each recommendation. For example, when recommending Oranges for a particular land, SHAP shows the decision was strongly driven by the matching Potassium values in the soil. This is the ideal case — transparent reasoning.

**System Transparency:** The black box has been successfully opened. Even when the AI makes errors, the system can explain the logical penalty — showing farmers which soil or climate factors caused the confusion. This builds trust.

---

## SLIDE 10 — CONCLUSION (~1 minute)

To conclude:

The combination of Stacking Ensemble and boosting models delivers **very high performance** in crop recommendation systems. CatBoost offers the highest accuracy at 99.77%, but **Stacking Ensemble is more stable and consistent** — which is more important for real-world deployment.

There is **no sign of overfitting** based on the comparison of train, CV, and test results.

SHAP successfully improves model interpretability by showing that **humidity and rainfall** are the most influential global factors in crop recommendations.

Overall, the combination of **Stacking Ensemble + SHAP** produces a model that is **accurate, stable, and explainable** — not just a calculator, but a smart advisor for farmers.

---

## SLIDE 11 — CLOSING (~15 seconds)

Thank you very much for your attention. I am happy to answer any questions.

---
---

# ANTICIPATED Q&A

## Q1: "Why Stacking Ensemble instead of just CatBoost which has higher accuracy?"
"CatBoost achieves 99.77% — slightly higher than Stacking's 99.55%. However, Stacking has the highest Cross-Validation accuracy (99.60% ± 0.0039) which shows superior consistency. In real-world deployment, stability matters more than peak accuracy on one specific test split. CatBoost might overfit to certain patterns while Stacking generalizes better."

## Q2: "Why Logistic Regression as meta-learner?"
"As shown in literature, a simple linear meta-learner prevents overfitting at the combination level. The base learners already capture non-linear patterns. The meta-learner's job is only to learn optimal weights — Logistic Regression is sufficient and interpretable."

## Q3: "Only 2 errors — is the dataset too easy?"
"The dataset is balanced (100 per class) and the features are scientifically meaningful soil/climate parameters. The high accuracy reflects that crop requirements are well-defined by these parameters. The 2 errors are between rice and jute — which genuinely have overlapping water requirements. This validates the model learned real agronomic logic."

## Q4: "How does SHAP help farmers in practice?"
"SHAP translates abstract probability scores into concrete explanations: 'This land is recommended for oranges because Potassium level is high.' Farmers can verify this against their experience. If SHAP shows a reason that contradicts their knowledge, they know to be cautious."
