# Техническое задание: ML-классификатор доменов gambling-тематики с DNS-валидацией

## 1. Цели и задачи проекта

### 1.1 Основная цель
Разработать систему классификации доменов на gambling/non-gambling тематику с использованием логистической регрессии, способную обрабатывать миллионы доменов из Certificate Transparency logs с минимальной стоимостью DNS-запросов.

### 1.2 Задачи MVP (Minimum Viable Product)
1. Собрать и подготовить обучающий датасет (2,000-5,000 gambling + 2,000-5,000 benign доменов)
2. Разработать feature engineering pipeline для доменных имен
3. Обучить модель логистической регрессии с метриками quality
4. Протестировать на тестовом наборе 10,000-20,000 доменов
5. Провести DNS-валидацию отобранных доменов (NS records для детекции паркингов)
6. Оценить false positive/false negative rates и итоговую cost-effectiveness

### 1.3 Критерии успеха MVP
- **Accuracy модели:** ≥92% на holdout set
- **Precision (gambling class):** ≥90% (минимум false positives)
- **Recall (gambling class):** ≥85% (допустимо пропустить некоторые gambling sites)
- **Inference speed:** ≥50,000 доменов/секунду на обычном CPU
- **DNS queries reduction:** фильтрация ≥60% нерелевантных доменов до DNS-запросов
- **Parking detection rate:** ≥90% парковок через NS records

---

## 2. Архитектура решения

### 2.1 Общая схема pipeline

```
[Data Collection]
    ↓
[Feature Engineering]
    ↓
[Model Training: Logistic Regression]
    ↓
[Model Evaluation & Validation]
    ↓
[Inference on Test Set: 10-20k domains]
    ↓
[DNS Validation: NS records]
    ↓
[Results Analysis & Metrics]
```

### 2.2 Компоненты системы

#### Компонент 1: Data Collection Module
**Назначение:** Сбор и preprocessing обучающих данных

**Источники данных:**
- **Gambling domains (positive class):**
  - BlockList Project Gambling List (~2,500 доменов): https://github.com/blocklistproject/Lists/blob/master/gambling.txt
  - Hagezi DNS Blocklists Gambling: https://github.com/hagezi/dns-blocklists
  - Дополнительно: ручная выборка из известных gambling операторов (bet365, 888casino, pokerstars, etc.)

- **Benign domains (negative class):**
  - Cloudflare Radar Top Domains: https://radar.cloudflare.com/domains
  - Tranco Top 10,000: https://tranco-list.eu/
  - Исключить: домены с gambling keywords для чистоты данных

**Функции:**
- `fetch_gambling_domains()` → list of gambling domains
- `fetch_benign_domains()` → list of benign domains
- `deduplicate_domains()` → удаление дубликатов
- `validate_domain_format()` → проверка валидности доменных имен
- `split_train_test()` → 80/20 split с stratification

**Output:** 
- `train.csv`: 3,200-4,000 gambling + 3,200-4,000 benign
- `test.csv`: 800-1,000 gambling + 800-1,000 benign
- `unlabeled_test.csv`: 10,000-20,000 доменов для inference

#### Компонент 2: Feature Engineering Module
**Назначение:** Извлечение признаков из доменных имен

**Features для extraction:**

1. **Keyword-based features (binary/count):**
   - Gambling keywords: casino, bet, betting, poker, slots, roulette, jackpot, bingo, wager, gamble, lottery, spin, sportsbook, odds
   - Multilingual: kasino, apuesta, apostas, казино, wetten
   - L33t speak variants: cas1no, p0ker, b3t
   - Count occurrences в domain name

2. **TLD features:**
   - Gambling-specific TLDs: .bet, .casino, .poker, .game (binary flag)
   - Common TLDs distribution: .com, .net, .org, .io, .co
   - Country-code TLDs (ccTLDs): .uk, .de, .es, .ru, .cn

3. **Structural features:**
   - Domain length (character count)
   - Number of subdomains (count dots)
   - Presence of numbers (binary + count)
   - Presence of hyphens (binary + count)
   - Ratio of digits to letters
   - Consecutive repeated characters (count)

4. **Character n-grams (TF-IDF):**
   - Character bigrams (2-grams): 'ca', 'as', 'si', 'in', 'no'
   - Character trigrams (3-grams): 'cas', 'asi', 'sin', 'ino'
   - Top 100-500 most frequent n-grams
   - TF-IDF vectorization для weight adjustment

**Функции:**
- `extract_features(domain: str) → dict` → все признаки для одного домена
- `create_feature_matrix(domains: list) → pandas.DataFrame` → feature matrix для batch
- `save_vectorizer(path: str)` → сохранение TF-IDF vectorizer для inference

**Output:**
- Feature matrix (pandas DataFrame)
- Fitted TF-IDF vectorizer (pickle)
- Feature names list

#### Компонент 3: Model Training Module
**Назначение:** Обучение Logistic Regression модели

**Спецификация модели:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,                    # Regularization strength (tune via CV)
    penalty='l2',             # L2 regularization
    solver='lbfgs',           # Optimizer
    max_iter=1000,            # Max iterations
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
```

**Training процесс:**
1. Hyperparameter tuning via GridSearchCV или RandomizedSearchCV
   - `C`: [0.01, 0.1, 1.0, 10.0]
   - `penalty`: ['l1', 'l2']
   - `solver`: ['lbfgs', 'liblinear', 'saga']

2. Cross-validation (5-fold stratified)
   - Метрики: accuracy, precision, recall, F1-score
   - ROC-AUC score

3. Feature importance analysis
   - Top 20 most important features (по coefficients)
   - Visualization через matplotlib

**Функции:**
- `train_model(X_train, y_train) → model` → обучение модели
- `tune_hyperparameters(X_train, y_train) → best_params` → поиск оптимальных гиперпараметров
- `evaluate_model(model, X_test, y_test) → metrics_dict` → evaluation метрики
- `save_model(model, path: str)` → сохранение модели (joblib)

**Output:**
- Trained model (joblib file)
- Training metrics report (JSON/CSV)
- Feature importance visualization (PNG)
- Confusion matrix (PNG)

#### Компонент 4: Inference Module
**Назначение:** Классификация unlabeled доменов

**Спецификация:**
- Batch inference на 10,000-20,000 доменов
- Probability scores для каждого домена
- Confidence thresholds для decision making

**Функции:**
- `load_model(path: str) → model` → загрузка обученной модели
- `predict_batch(domains: list) → predictions` → batch prediction
- `predict_proba_batch(domains: list) → probabilities` → вероятности классов
- `filter_by_confidence(predictions, threshold=0.8) → filtered_domains` → фильтрация по уверенности

**Output:**
- Predictions CSV с колонками:
  - `domain`
  - `prediction` (0=benign, 1=gambling)
  - `probability_benign`
  - `probability_gambling`
  - `confidence` (max probability)

#### Компонент 5: DNS Validation Module
**Назначение:** Проверка NS records для детекции паркингов

**Parking nameservers list (MISP warninglists):**
```python
# Используем только верифицированные parking nameservers
PARKING_NAMESERVERS_2025 = [
    'sedoparking.com',
    'bodis.com',
    'parkingcrew.net',
    'parklogic.com',
    'above.com',
    'afternic.com',
    'namebrightdns.com',
    'dns-parking.com',
    'ztomy.com',  # 14th по популярности
]

# Known parking IPs (обновлять ежеквартально)
PARKING_IPS_2025 = {
    '3.33.130.190',      # GoDaddy AWS
    '15.197.148.33',     # GoDaddy AWS
    '50.63.202.32',      # GoDaddy legacy
    '199.59.242.150',    # Bodis
}
```

**Спецификация:**
- Async DNS queries (aiodns или dnspython)
- Concurrency: 100-500 одновременных запросов
- Timeout: 3-5 секунд на запрос
- Retry logic: 2 попытки при failures
- Используем одновременно разные публичные DNS-серверы (Google, Cloudflare), чередуя и делая не более 10 запросов в секунду.

**Функции:**
- `query_ns_records(domain: str) → list[str]` → получение NS records
- `is_parked(ns_records: list) → bool` → проверка на parking
- `batch_dns_check(domains: list) → results_dict` → batch DNS checking
- `calculate_parking_stats(results: dict) → stats` → статистика по паркингам

**Output:**
- DNS results CSV с колонками:
  - `domain`
  - `ns_records` (JSON list)
  - `is_parked` (boolean)
  - `parking_provider` (если detected)
  - `query_status` (success/timeout/nxdomain)

---

## 3. Технический стек

### 3.1 Язык программирования
**Python 3.9+** — обязательно

**Обоснование:**
- Лучшие ML библиотеки (scikit-learn)
- Отличная экосистема для data science
- Простота прототипирования
- Async поддержка для DNS queries

### 3.2 Основные библиотеки

**Machine Learning & Data Processing:**
```
scikit-learn==1.5.0      # Logistic Regression, metrics
pandas==2.2.0            # Data manipulation
numpy==1.26.0            # Numerical operations
matplotlib==3.8.0        # Visualization
seaborn==0.13.0          # Statistical plotting
joblib==1.3.0            # Model serialization
```

**DNS Operations:**
```
aiodns==3.2.0            # Async DNS resolver
dnspython==2.6.0         # DNS queries (альтернатива)
aiohttp==3.9.0           # Async HTTP (для fetch данных)
```

**Utilities:**
```
requests==2.31.0         # HTTP requests
tqdm==4.66.0             # Progress bars
python-dotenv==1.0.0     # Environment variables
```

### 3.3 Опциональные библиотеки для расширения
```
imbalanced-learn==0.12.0  # SMOTE для балансировки классов
shap==0.44.0              # Feature importance analysis
mlflow==2.10.0            # Experiment tracking
```

---

## 4. Структура проекта

```
gambling-classifier/
├── README.md
├── requirements.txt
├── .env.example
├── config.py              # Конфигурация (пути, параметры)
│
├── data/
│   ├── raw/               # Исходные данные
│   │   ├── gambling_domains.txt
│   │   ├── benign_domains.txt
│   │   └── unlabeled_test.txt
│   ├── processed/         # Обработанные датасеты
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── features.csv
│   └── results/           # Результаты inference
│       ├── predictions.csv
│       └── dns_validation.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metrics.json
│
├── notebooks/             # Jupyter notebooks для анализа
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py    # Модуль 1: Сбор данных
│   ├── feature_engineering.py # Модуль 2: Feature extraction
│   ├── model_training.py      # Модуль 3: Обучение модели
│   ├── inference.py           # Модуль 4: Inference
│   ├── dns_validation.py      # Модуль 5: DNS проверка
│   └── utils.py               # Вспомогательные функции
│
├── scripts/
│   ├── 01_collect_data.py
│   ├── 02_train_model.py
│   ├── 03_run_inference.py
│   └── 04_validate_dns.py
│
└── tests/
    ├── test_features.py
    ├── test_model.py
    └── test_dns.py
```

---

## 5. Этапы разработки (Timeline: 2-3 недели)

### Неделя 1: Data Collection & Feature Engineering

#### День 1-2: Сбор данных
- [ ] Скачать gambling домены из BlockList Project, Hagezi
- [ ] Скачать benign домены из Cloudflare Radar, Tranco
- [ ] Очистка данных: удаление дубликатов, валидация форматов
- [ ] Создание train/test split (80/20)
- [ ] Подготовка unlabeled test set (10-20k доменов)

**Deliverable:** `data/raw/` заполнена, `data/processed/train.csv` и `test.csv` готовы

#### День 3-4: Feature Engineering
- [ ] Реализация extraction функций для всех feature типов
- [ ] Создание feature matrix для train/test
- [ ] TF-IDF vectorizer для character n-grams
- [ ] Feature visualization и EDA (Exploratory Data Analysis)
- [ ] Сохранение vectorizer для последующего inference

**Deliverable:** `src/feature_engineering.py` готов, feature matrix создана

#### День 5: Анализ данных
- [ ] Jupyter notebook для EDA
- [ ] Visualizations: feature distributions, correlations
- [ ] Идентификация наиболее важных признаков
- [ ] Проверка на class imbalance

**Deliverable:** `notebooks/01_eda.ipynb` с insights

### Неделя 2: Model Training & Evaluation

#### День 6-7: Model Training
- [ ] Baseline Logistic Regression модель
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Cross-validation (5-fold)
- [ ] Feature importance analysis
- [ ] Сохранение лучшей модели

**Deliverable:** `models/logistic_regression.joblib`, training metrics

#### День 8-9: Model Evaluation
- [ ] Evaluation на holdout test set
- [ ] Confusion matrix, classification report
- [ ] ROC curve, precision-recall curve
- [ ] Error analysis: false positives/negatives
- [ ] Threshold tuning для optimal precision/recall balance

**Deliverable:** Полный evaluation report с визуализациями

#### День 10: Inference на unlabeled данных
- [ ] Загрузка 10-20k unlabeled доменов
- [ ] Batch inference с probability scores
- [ ] Фильтрация по confidence threshold (например, >0.8)
- [ ] Анализ распределения predictions
- [ ] Выборка доменов для DNS validation

**Deliverable:** `data/results/predictions.csv`

### Неделя 3: DNS Validation & Final Analysis

#### День 11-12: DNS Validation
- [ ] Реализация async DNS resolver
- [ ] Загрузка parking nameservers list (MISP)
- [ ] Batch NS record queries для predicted gambling доменов
- [ ] Детекция паркингов
- [ ] Статистика по parking rate

**Deliverable:** `data/results/dns_validation.csv`

#### День 13-14: Анализ результатов
- [ ] Расчет реальных false positive/negative rates
- [ ] Оценка cost-effectiveness (DNS queries saved)
- [ ] Рекомендации по threshold adjustment
- [ ] Final report с метриками и визуализациями
- [ ] Документация API и usage examples

**Deliverable:** Финальный отчет, готовый к production код

---

## 6. Метрики и критерии приемки

### 6.1 Model Performance Metrics

**Обязательные метрики:**
- **Accuracy:** ≥92% на test set
- **Precision (gambling):** ≥90%
- **Recall (gambling):** ≥85%
- **F1-Score (gambling):** ≥87%
- **ROC-AUC:** ≥0.95

**Дополнительные метрики:**
- Confusion matrix analysis
- Per-class precision/recall
- False positive rate: ≤10%
- False negative rate: ≤15%

### 6.2 System Performance Metrics

**Скорость inference:**
- Single prediction: <1ms
- Batch 10k domains: <10 seconds
- Throughput: ≥50,000 domains/second

**DNS validation:**
- Success rate: ≥95% (queries не таймаутятся)
- Parking detection accuracy: ≥90%
- Average query time: <2 seconds per domain

### 6.3 Business Metrics

**Cost-effectiveness:**
- DNS queries reduction: ≥60% доменов отфильтровано до DNS
- Если на 20k test set:
  - Без ML: 20,000 DNS queries
  - С ML: ≤8,000 DNS queries (40% экономия минимум)

**Scalability:**
- Код готов к масштабированию до 1M+ доменов
- Модель размер: <50MB
- Memory footprint: <500MB при inference

---

## 7. Риски и митигация

### Риск 1: Class Imbalance
**Описание:** Возможен дисбаланс классов в реальных CT logs

**Митигация:**
- `class_weight='balanced'` в Logistic Regression
- Опционально: SMOTE для oversampling minority class
- Stratified sampling при train/test split

### Риск 2: Domain Name Variations
**Описание:** L33t speak, internationalized domains, новые паттерны

**Митигация:**
- Character n-grams захватывают variations
- Multilingual keyword list
- Регулярное обновление training data

### Риск 3: Parking Detection False Positives
**Описание:** Некоторые legitimate сайты могут использовать те же nameservers

**Митигация:**
- Multiple validation signals (NS + IP ranges + ASN)
- Manual review sample для calibration
- Confidence scoring вместо binary classification

### Риск 4: DNS Query Failures
**Описание:** Timeouts, rate limits, NXDOMAIN

**Митигация:**
- Async queries с retry logic
- Fallback на alternative DNS resolvers
- Graceful degradation при failures

---

## 8. Deliverables (Что должно быть готово)

### 8.1 Code Deliverables
- [ ] Полностью функциональный Python package
- [ ] Все модули (`data_collection`, `feature_engineering`, `model_training`, `inference`, `dns_validation`)
- [ ] CLI scripts для каждого этапа
- [ ] Unit tests (≥70% code coverage)
- [ ] Requirements.txt с pinned versions

### 8.2 Model Artifacts
- [ ] Trained Logistic Regression model (`.joblib`)
- [ ] TF-IDF vectorizer (`.joblib`)
- [ ] Model metrics report (JSON/CSV)
- [ ] Feature importance visualization

### 8.3 Data Artifacts
- [ ] Training dataset (CSV)
- [ ] Test dataset (CSV)
- [ ] Predictions на unlabeled data (CSV)
- [ ] DNS validation results (CSV)

### 8.4 Documentation
- [ ] README.md с installation и usage instructions
- [ ] API documentation для каждого модуля
- [ ] Jupyter notebooks с анализом
- [ ] Final report с результатами и рекомендациями

### 8.5 Reports
- [ ] Model evaluation report:
  - Training/test metrics
  - Confusion matrix
  - ROC curve, precision-recall curve
  - Feature importance
  - Error analysis
  
- [ ] DNS validation report:
  - Parking detection rate
  - False positive/negative analysis
  - Cost-effectiveness analysis

- [ ] Recommendations:
  - Optimal confidence threshold
  - Production deployment strategy
  - Suggestions для improvement

---

## 9. Команды для запуска (Quick Start)

### 9.1 Setup
```bash
# Clone repository
git clone <repo-url>
cd gambling-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Execution Pipeline
```bash
# Step 1: Collect data
python scripts/01_collect_data.py

# Step 2: Train model
python scripts/02_train_model.py --tune-hyperparams

# Step 3: Run inference
python scripts/03_run_inference.py --input data/raw/unlabeled_test.txt

# Step 4: DNS validation
python scripts/04_validate_dns.py --input data/results/predictions.csv
```

### 9.3 Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_eda.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_evaluation.ipynb
```

---

## 10. Дальнейшее развитие (Post-MVP)

### Возможные улучшения:
1. **Ensemble модели:** Комбинация Logistic Regression + Random Forest для uncertain cases
2. **Active learning:** Ручная разметка uncertain predictions для retraining
3. **Real-time inference API:** REST API для on-demand classification
4. **Database integration:** PostgreSQL для кэширования результатов
5. **Monitoring dashboard:** Streamlit app для визуализации метрик
6. **Automated retraining:** CI/CD pipeline для регулярного обновления модели
7. **Multi-label classification:** Детекция других категорий (phishing, malware, adult)

---

## 11. Контакты и поддержка

**Вопросы по ТЗ:** [указать контакт]

**Репозиторий:** [указать URL после создания]

**Timeline:** 2-3 недели от start до MVP

**Estimated Effort:** 60-80 часов development time