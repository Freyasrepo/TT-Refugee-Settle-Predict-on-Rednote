# Efficient Deep Learning for Predicting User Retention Among TikTok Refugees on Xiaohongshu

## Project Overview
This project explores the retention of TikTok refugee users on Xiaohongshu, analyzing their behavioral engagement and content adaptation strategies. We utilize deep learning techniques with a focus on efficiency to predict user retention while minimizing computational costs.

## Problem Definition
The goal is to determine which factors contribute to users remaining active on Xiaohongshu after their migration from TikTok in January 2025. This is framed as a supervised classification problem, where we predict whether a user will stay engaged in February based on their behavioral and content-related features.

## Data
### 1. Data Source
The dataset (`tt_refugee.csv`) consists of user interactions and content metadata from January and February 2025. It is publicly available on Kaggle:
[Kaggle Dataset](https://www.kaggle.com/datasets/jinghuawu/tiktok-refugee-migration-data)

### 2. Data Features
- **Behavioral Features:** Likes, shares, comments, collections
- **Textual Content:** Posts made by users, processed for topic modeling
- **Temporal Trends:** Daily activity variations and retention indicators

## Methodology
### 1. Data Preprocessing
- **Text Processing:** Tokenization (Jieba for Chinese), stop-word removal, TF-IDF, FastText, and DistilBERT embeddings.
- **Feature Scaling:** Standardization of behavioral features.
- **Retention Labeling:** Users active in February are labeled as retained.

### 2. Exploratory Data Analysis (EDA)
- **Behavioral Analysis:** Distribution of interaction features, retention trends.
- **Correlation Study:** Identifying key engagement features affecting retention.
- **Topic Modeling (NMF):** Identifying content themes linked to user retention.

### 3. Predictive Modeling
We implemented a hybrid deep learning model combining:
- **DistilBERT:** For extracting rich semantic embeddings from textual data.
- **MLP (Multi-Layer Perceptron):** For processing behavioral engagement features.
- **LSTM (Optional):** To capture temporal trends in user activity.

#### Model Architecture
- **Dense (128 neurons, ReLU) + Batch Normalization + Dropout (0.2)**
- **Dense (64 neurons, ReLU) + Batch Normalization + Dropout (0.2)**
- **Final Output Layer:** Sigmoid activation for binary classification.

#### Training Strategy
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Binary cross-entropy
- **Regularization:** Dropout layers to mitigate overfitting
- **Early Stopping & Learning Rate Reduction:** Improve generalization

## Results & Discussion
- **Accuracy:** 99.38% (but misleading due to class imbalance)
- **AUC-ROC:** 0.7012 (indicating moderate predictive power)
- **Issue:** High class imbalance led to poor recall for retained users.
- **Future Work:** Addressing class imbalance through weighted loss functions or synthetic data augmentation.

## Requirements
- Python 3.8+
- TensorFlow 2.18.0
- NumPy 1.26.4
- Scikit-learn
- Transformers (Hugging Face)
- Jieba for Chinese tokenization

## Usage
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run data preprocessing and feature extraction:
   ```sh
   python preprocess.py
   ```
4. Train the model:
   ```sh
   python train.py
   ```
5. Evaluate model performance:
   ```sh
   python evaluate.py
   ```

## References
- Gershman, J., Bobrowsky, M., & Needleman, S. E. (2024, December 6). *Appeals court upholds U.S. ban of TikTok* [Video]. Wall Street Journal. https://www.wsj.com/politics/policy/tik-tok-congress-ban-court-ruling-1f0d6837
- Wu Jinghua. (2025). TikTok Refugee Migration Data [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/6808187
- Yuan, C. H., Cao, Y. D., Wei, T., & Pei, Y. T. (2025). *TikTok post-lockdown migration: Xiaohongshu comment analysis* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/10735086
- 21st Century Business Herald. (2025, January 20). Xiaohongshu’s rise: A new social media giant? *21st Century Business Herald.* https://www.21jingji.com/article/20250120/herald/9d80644e2ec7e853249b1d7b83d3e81b.html

## License
This project is open-source under the MIT License.


