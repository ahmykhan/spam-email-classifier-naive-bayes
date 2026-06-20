# spam-email-classifier-naive-bayes#
Email Spam Classification using Gaussian Naive Bayes

An end-to-end Machine Learning pipeline implemented in Python to detect and classify emails as either **Spam** or **Non-Spam (Ham)**. This project utilizes the widely recognized **UCI Spambase Dataset** to train a predictive Gaussian Naive Bayes classifier, leveraging statistical continuous feature patterns to identify unsolicited communications.

---

## 📊 Dataset Characteristics & Insights

The pipeline operates on **4,601 instances** featuring **57 unique continuous attributes** extracted from raw text email structures:

* **Total Records (Pre-cleaning):** 4,601 rows
* **Target Label Distribution:**
    * **Spam:** 1,813 emails (~39.4%)
    * **Non-Spam:** 2,788 emails (~60.6%)
* **Feature Distribution Framework:**
    * **Word Frequencies (48 columns):** Indicates the percentage of specific terms (e.g., *free*, *remove*, *money*) present within an email body on a scaled range from `0` to `100`.
    * **Character Frequencies (6 columns):** Measures percentage metrics for specific formatting trigger symbols (e.g., `$`, `!`).
    * **Capital Run-Length (3 columns):** Measures the average, longest string length, and absolute count sequence of successive consecutive capital letters.

---

## 🛠️ Data Preprocessing & Pipeline Rules

To ensure reliable model convergence and accurate general purpose prediction matrices, the dataset undergoes strict algorithmic cleaning phases:

1.  **Deduplication:** Dropping redundant recurring entries to eliminate statistical bias.
2.  **Null & Infinite Clean:** Replacing infinity floats and removing empty variables (`.dropna()`).
3.  **Data Leakage Prevention:** Outlining strict evaluation barriers, data features are normalized using `StandardScaler`. The scaling module is fitted exclusively on the training sets before cross-transforming testing variations.

---

## 🚀 Experimental Multi-Split Design

The classifier performance profile is evaluated across two distinct structural split partitions to measure accuracy vs model decay constraints:

* **Scenario A (80:20 Split):** 80% Training Data baseline matched against a 20% Testing Validation pool.
* **Scenario B (60:40 Split):** 60% Training Data configuration matched against a 40% Testing Validation pool.

Performance across both matrix pools is measured down to exact precision metrics, covering **Accuracy, Precision, Recall, and F1-Scores** along with visual **Confusion Matrix Maps**.

---

## 💻 Tech Stack & Dependencies

* **Language Environment:** Python 
* **Data Analysis Tools:** `numpy`, `pandas`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning Framework:** `scikit-learn` (`GaussianNB`, `train_test_split`, `StandardScaler`)

---
