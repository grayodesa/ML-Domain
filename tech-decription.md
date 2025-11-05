# ТЕХНИЧЕСКОЕ ОПИСАНИЕ МЕХАНИЗМА ОБРАБОТКИ ДОМЕНОВ

## 1. АРХИТЕКТУРА СИСТЕМЫ

Система представляет собой ML-конвейер для классификации доменов с поддержкой gambling-тематики. Архитектура состоит из 5 основных модулей, работающих последовательно:

```
Сбор данных → Извлечение признаков → Обучение модели → Инференс → DNS-валидация
```

## 2. МОДУЛЬ СБОРА ДАННЫХ (data_collection.py)

### 2.1 Основной функционал
Модуль собирает обучающие данные из двух классов: gambling и benign (безопасные) домены.

### 2.2 Источники данных

**Gambling-домены:**
- Hagezi DNS blocklist (`gambling.medium-onlydomains.txt`)
- Загрузка через `fetch_from_url()` с HTTP timeout 30 сек

**Benign-домены:**
- Tranco Top 1M список (`top-1m.csv`)
- Локальный файл `benign_domains.txt` (если доступен)
- Fallback: 16 hardcoded доменов (`google.com`, `amazon.com`, etc.)

### 2.3 Процесс нормализации доменов

**Функция `clean_domain(line)`:**
- Удаляет комментарии (`#`)
- Убирает префиксы: `http://`, `https://`, `www.`
- Удаляет IP-адреса в формате hosts-файла
- Удаляет номера портов (`:8080`)
- Убирает пути (`/path/to/resource`)
- Приводит к lowercase

**Функция `validate_domain(domain)`:**
- Минимальная длина: 4 символа
- Обязательное наличие точки
- Regex валидация: `^[a-z0-9.-]+$`
- Проверка через `tldextract` на наличие domain + suffix

**Функция `normalize_domain(domain)` (utils.py):**

```python
# Использует tldextract для извлечения registered domain
# Примеры:
www.example.com → example.com
subdomain.example.co.uk → example.co.uk
a.b.c.example.org → example.org
```

### 2.4 Фильтрация данных

Для benign-доменов применяется функция `has_gambling_keyword()`:
- Проверяет 18 ключевых слов: `casino`, `bet`, `betting`, `poker`, `slots`, `kasino`, `apuesta`, `wetten`, etc.
- Домены с gambling-keywords исключаются из benign-набора

### 2.5 Создание датасета

**Функция `create_dataset()`:**
- Создает DataFrame с колонками: `domain`, `label` (0=benign, 1=gambling)
- Объединяет gambling и benign домены
- Перемешивает данные (shuffle с `random_state=42`)
- Разделяет на train/test в соотношении 80/20
- Сохраняет в `data/processed/train.csv` и `test.csv`

**Формат CSV:**
```csv
domain,label
example.com,0
casino365.bet,1
```

## 3. МОДУЛЬ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (feature_engineering.py)

### 3.1 Класс DomainFeatureExtractor

Центральный класс для извлечения features. Хранит состояние:
- `tfidf_vectorizer`: TF-IDF векторизатор для n-грамм
- `feature_names`: Список имен всех признаков
- `manual_feature_names`: Список имен мануальных признаков (не n-граммы)

### 3.2 Типы извлекаемых признаков

#### A. Ключевые слова (Keyword Features)

**Функция:** `extract_keyword_features(domain)`

**Признаки:**
- `keyword_{keyword}`: Счетчик вхождений для каждого из 23 ключевых слов
- `total_keyword_count`: Общее количество ключевых слов в домене
- `has_gambling_keyword`: Бинарный флаг (0/1)

**Список ключевых слов:**
```python
# English
'casino', 'bet', 'betting', 'poker', 'slots', 'roulette', 'jackpot',
'bingo', 'wager', 'gamble', 'gambling', 'lottery', 'spin', 'sportsbook',
'odds', 'blackjack', 'craps', 'dice', 'vegas', 'stake', 'win', 'bonus'

# Multilingual
'kasino', 'apuesta', 'apostas', 'wetten', 'juego', 'jeu'

# L33t speak
'cas1no', 'c4sino', 'b3t', 'p0ker', 'sl0ts', 'gambl3'
```

#### B. TLD Features (Top-Level Domain)

**Функция:** `extract_tld_features(domain)`

**Gambling TLDs (6 штук):**
- `.bet`, `.casino`, `.poker`, `.game`, `.win`, `.games`

**Common TLDs (9 штук):**
- `.com`, `.net`, `.org`, `.io`, `.co`, `.uk`, `.de`, `.fr`, `.es`

Каждый TLD создает бинарный признак `tld_{tld}` (0/1)

#### C. Структурные признаки (Structural Features)

**Функция:** `extract_structural_features(domain)`

**Признаки:**
- `domain_length`: Длина домена в символах
- `dot_count`: Количество точек (уровень поддоменов)
- `digit_count`: Количество цифр
- `has_digits`: Бинарный флаг наличия цифр
- `hyphen_count`: Количество дефисов
- `has_hyphen`: Бинарный флаг наличия дефисов
- `digit_letter_ratio`: Отношение цифр к буквам
- `max_consecutive_chars`: Максимум повторяющихся символов подряд
- `vowel_ratio`: Отношение гласных к согласным

#### D. Character N-grams (TF-IDF)

**Функция:** `fit_tfidf(domains, max_features=500)`

**Параметры TfidfVectorizer:**
```python
analyzer='char'          # Посимвольный анализ
ngram_range=(2, 3)      # Биграммы и триграммы
max_features=500        # Топ-500 n-грамм
lowercase=True
min_df=2                # Игнорировать редкие n-граммы
```

Примеры n-грамм: `ca`, `cas`, `asi`, `sin`, `ino` (для "casino")

### 3.3 Создание feature matrix

**Функция `create_feature_matrix(domains, is_training=False)`:**

**Процесс:**
1. Извлекает manual features для каждого домена (with progress bar via tqdm)
2. Заполняет отсутствующие колонки нулями (`fillna(0)`)
3. Выравнивает колонки для test данных по training schema
4. Трансформирует домены через TF-IDF vectorizer
5. Объединяет manual features + TF-IDF features
6. Возвращает DataFrame с ~550 колонками (50 manual + 500 n-grams)

**Оптимизация памяти: `create_feature_matrix_sparse()`**

Для больших датасетов (5M+ доменов):
- Использует `scipy.sparse.csr_matrix` для manual features
- TF-IDF features уже sparse
- Объединяет через `scipy.sparse.hstack()`
- Снижает потребление памяти в 10-20 раз

### 3.4 Сохранение/загрузка

**Функция `save(path)`:**
- Сохраняет `tfidf_vectorizer.joblib` (TF-IDF векторизатор)
- Сохраняет `feature_metadata.joblib` (имена features)

**Функция `load(path)`:**
- Загружает векторизатор и metadata для инференса

## 4. МОДУЛЬ ОБУЧЕНИЯ МОДЕЛЕЙ (model_training.py)

### 4.1 Поддерживаемые модели

Система поддерживает два классификатора:
1. `GamblingDomainClassifier` (Logistic Regression)
2. `RandomForestDomainClassifier` (Random Forest)

### 4.2 Logistic Regression (GamblingDomainClassifier)

**Базовая конфигурация:**
```python
LogisticRegression(
    C=1.0,                    # Регуляризация
    penalty='l2',             # L2 регуляризация
    solver='lbfgs',           # Оптимизатор
    max_iter=2000,            # Макс итераций
    class_weight='balanced',  # Балансировка классов
    random_state=42
)
```

**Hyperparameter tuning (`tune_hyperparameters()`):**

Grid search параметров:
```python
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],                    # 4 значения
    'penalty': ['l2'],                               # L2 only for lbfgs
    'solver': ['lbfgs', 'liblinear', 'saga'],       # 3 солвера
    'class_weight': ['balanced'],
    'max_iter': [2000]
}
```

Дополнительно тестируется L1 penalty для liblinear и saga.

**GridSearchCV конфигурация:**
- 5-fold cross-validation
- Метрика: F1 score
- Параллелизация: `n_jobs=-1` (все CPU cores)

**Лучшие параметры (по данным из CLAUDE.md):**
- `C=10.0`
- `solver='liblinear'`
- `penalty='l2'`

**Производительность:**
- Training time: 2-5 минут (на 159k samples)
- Inference speed: <1 мс/домен
- Размер модели: ~5 MB
- Accuracy: 92.29%
- Precision: 95.36%
- Recall: 88.86%

### 4.3 Random Forest (RandomForestDomainClassifier)

**Оптимизированная конфигурация для Apple Silicon:**
```python
RandomForestClassifier(
    n_estimators=200,          # 200 деревьев
    max_depth=30,              # Макс глубина дерева
    min_samples_split=5,       # Мин сэмплов для split
    min_samples_leaf=2,        # Мин сэмплов в листе
    max_features='sqrt',       # sqrt(n_features) для split
    n_jobs=8,                  # M1 Max: 8 performance cores
    class_weight='balanced',
    random_state=42,
    verbose=1
)
```

**Hyperparameter tuning:**

Фокусированная сетка (27 комбинаций):
```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt'],
    'min_samples_leaf': [2],
    'class_weight': ['balanced'],
    'n_jobs': [8]
}
```

**GridSearchCV конфигурация:**
- 3-fold CV (сокращено с 5 для скорости)
- Метрика: F1 score
- Outer parallelization: `n_jobs=1` (RF делает внутреннюю параллелизацию)

**Производительность:**
- Training time: 3-8 минут
- Inference speed: 2-5 мс/домен
- Размер модели: ~50-100 MB
- Accuracy: ~93-95% (оценочно)
- Precision: ~94-96%
- Recall: ~90-93%

### 4.4 Evaluation и метрики

**Функция `evaluate(X_test, y_test, output_dir)`:**

**Вычисляет метрики:**
- Accuracy
- ROC-AUC
- Precision (для класса gambling)
- Recall (для класса gambling)
- F1 score
- Confusion matrix (TN, FP, FN, TP)

**Генерирует визуализации:**
- Confusion Matrix - тепловая карта (seaborn)
- ROC Curve - с AUC score
- Precision-Recall Curve
- Feature Importance - топ-20 features

**Feature Importance:**

Для Logistic Regression:
- Использует коэффициенты (`model.coef_[0]`)
- Сортирует по абсолютному значению
- Зеленые bars = положительная корреляция с gambling
- Красные bars = отрицательная корреляция

Для Random Forest:
- Использует Gini importance (`model.feature_importances_`)
- Сортирует по убыванию
- Сохраняет в `top_features_rf.csv`

### 4.5 Сохранение модели

**Функция `save(path)`:**

Для Logistic Regression:
- `logistic_regression.joblib` - модель

Для Random Forest:
- `random_forest.joblib` - модель
- `rf_metadata.json` - конфигурация (n_estimators, max_depth, best_params)

Общие файлы:
- `tfidf_vectorizer.joblib` (из feature extractor)
- `feature_metadata.joblib` (имена features)
- `model_metrics.json` или `model_metrics_rf.json`

## 5. МОДУЛЬ ИНФЕРЕНСА (inference.py)

### 5.1 Класс DomainClassifierInference

Центральный класс для классификации unlabeled доменов.

**Инициализация:**
```python
DomainClassifierInference(model_path, model_type='auto')
```

**Auto-detection логика:**
- Проверяет наличие `random_forest.joblib`
- Проверяет наличие `logistic_regression.joblib`
- Если оба существуют → предпочитает Random Forest
- Если ни одного → выбрасывает FileNotFoundError

**Загружает:**
- Модель (через `classifier.load()`)
- Feature extractor (через `feature_extractor.load()`)

### 5.2 Batch prediction

**Функция `predict_batch(domains)`:**

**Процесс:**
1. Нормализует домены через `normalize_domain()` (utils.py)
2. Извлекает features через `create_feature_matrix()`
3. Делает prediction: `model.predict(X)`
4. Вычисляет вероятности: `model.predict_proba(X)`
5. Возвращает DataFrame с колонками:
   - `original_domain`: Оригинальный домен
   - `domain`: Нормализованный домен
   - `prediction`: 0 (benign) или 1 (gambling)
   - `probability_benign`: Вероятность benign класса
   - `probability_gambling`: Вероятность gambling класса
   - `confidence`: max(probability_benign, probability_gambling)
   - `label`: 'benign' или 'gambling' (строковое значение)

**Статистика:**
- Время выполнения (seconds)
- Throughput (domains/second)
- Распределение предсказаний (gambling %, benign %)

**Целевой throughput:** ≥50,000 domains/second

### 5.3 Chunked processing (для больших датасетов)

**Функция `predict_batch_chunked(domains, batch_size=50000, output_path=None)`:**

Для датасетов 5M+ доменов с ограничением памяти.

**Процесс:**
1. Разбивает домены на чанки по `batch_size` доменов
2. Для каждого чанка:
   - Нормализует домены
   - Извлекает features через `create_feature_matrix_sparse()` (sparse matrices)
   - Делает prediction
   - Если `output_path` указан → пишет chunk incrementally в CSV
   - Если `output_path` None → аккумулирует в памяти
3. Возвращает DataFrame или пустой DataFrame (если данные на диске)

**Memory optimization:**
- Использует sparse matrices для features (экономия 10-20x памяти)
- Инкрементальная запись в CSV (chunk by chunk)
- Не аккумулирует результаты в памяти при `output_path != None`

**Auto-detection:**
- Используется автоматически для датасетов >100k доменов

### 5.4 Filtering и анализ

**Функция `filter_by_confidence(results, threshold=0.8)`:**
- Фильтрует предсказания по confidence threshold
- Оставляет только домены с confidence >= threshold
- Выводит статистику: сколько доменов осталось/удалено

**Функция `analyze_predictions(results, output_dir)`:**

Генерирует 4 графика:
1. Confidence Distribution - гистограмма confidence
2. Probability Distribution by Class - распределение вероятностей
3. Prediction Counts - bar chart (gambling vs benign)
4. Confidence by Prediction - boxplot confidence по классам

Сохраняет в `prediction_analysis.png`

### 5.5 Функция run_inference (главный entry point)

**Сигнатура:**
```python
run_inference(
    input_file,           # Path к файлу с доменами (one per line)
    model_path,           # Path к обученной модели
    output_dir,           # Output директория
    confidence_threshold=0.8,
    model_type='auto',
    batch_size=None,      # Default 50000
    use_chunked=None      # Auto-detect по размеру датасета
)
```

**Workflow:**
1. Загружает домены из `input_file`
2. Инициализирует inference engine
3. Auto-detect chunked mode (если >100k доменов)
4. Выполняет prediction (chunked или standard)
5. Сохраняет результаты:
   - `predictions.csv` - все предсказания
   - `predictions_confident_80.csv` - с confidence ≥80%
   - `predicted_gambling.csv` - только gambling домены (sorted by probability)
6. Генерирует analysis plots (если ≤500k доменов)

**Chunked mode особенности:**
- Incremental CSV writing во время prediction
- Фильтрация и анализ также выполняются по чанкам
- Пропускает visualizations для >500k доменов (memory intensive)

## 6. МОДУЛЬ DNS ВАЛИДАЦИИ (dns_validation.py)

### 6.1 Назначение модуля

Валидирует gambling-домены через DNS queries для определения:
- **Active domains** - реально работающие gambling сайты
- **Parked domains** - припаркованные домены (не используются)

### 6.2 Класс DNSValidator

**Инициализация:**
```python
DNSValidator(
    timeout=10.0,            # Таймаут DNS запроса (сек)
    max_retries=2,           # Количество retry попыток
    concurrency=1500,        # Макс параллельных запросов
    resolver_pool_size=50    # Размер пула resolvers
)
```

**Оптимизация для локального Unbound resolver:**
- DNS server: 127.0.0.1 (local Unbound)
- Высокая concurrency: 1500 параллельных запросов
- Пул из 50 DNS resolvers для round-robin балансировки
- Короткие retry delays: 0.1 сек

### 6.3 Parking Detection механизм

#### A. Parking Nameservers (NS records)

**Список parking nameservers:**
```python
PARKING_NAMESERVERS_2025 = {
    'sedoparking.com',
    'bodis.com',
    'parkingcrew.net',
    'parklogic.com',
    'above.com',
    'afternic.com',
    'namebrightdns.com',
    'dns-parking.com',
    'ztomy.com'
}
```

**Функция `_check_parking_ns(ns_records)`:**
- Проверяет каждый NS record на вхождение parking nameserver
- Возвращает `(is_parked, parking_provider)`

#### B. Parking IPs (A records)

**Индивидуальные IPs (33 штуки):**
```python
PARKING_IPS_2025 = {
    '3.33.130.190',      # AWS, 28,700 domains
    '15.197.148.33',     # AWS, 28,852 domains
    '199.59.243.228',    # Bodis, 37,512 domains
    '91.195.240.12',     # Sedo, 21,231 domains
    '208.91.197.27',     # 24,637 domains
    # ... и т.д.
}
```

**CIDR ranges:**
```python
PARKING_IP_RANGES_2025 = [
    '185.53.176.0/22',   # 185.53.176.0 - 185.53.179.255
]
```

**Функция `_check_parking_ip(a_records)`:**
- Проверяет exact match в `PARKING_IPS_2025` (fast set lookup)
- Проверяет вхождение в CIDR ranges через `ipaddress.ip_network`
- Возвращает `(is_parked, parking_provider)`

**Формат `parking_provider`:**
- `"sedoparking.com"` - для NS records
- `"IP:3.33.130.190"` - для individual IPs
- `"CIDR:185.53.176.0/22"` - для CIDR ranges

### 6.4 DNS Query процесс

**Функция `_validate_domain(domain)`:**

Асинхронный процесс с semaphore контролем:
1. Получает resolver из пула (round-robin через `_get_resolver()`)
2. Запрашивает 3 типа records параллельно (`asyncio.gather`):
   - NS records - nameservers
   - A records - IPv4 адреса
   - CNAME records - canonical names
3. При ошибке делает до 2 retry с delay 0.1 сек
4. Определяет `query_status`:
   - `"success"` - хотя бы один record получен
   - `"nxdomain"` - домен не существует
   - `"timeout"` - превышен таймаут
   - `"error: {message}"` - другая ошибка
5. Проверяет parking (сначала NS, потом IPs)
6. Обновляет статистику через `_update_stats()`
7. Возвращает DNSResult dataclass

**DNSResult dataclass:**
```python
@dataclass
class DNSResult:
    domain: str
    ns_records: List[str]
    a_records: List[str]
    cname_records: List[str]
    is_parked: bool
    parking_provider: Optional[str]
    query_status: str
    query_time: float
```

### 6.5 Batch validation

**Функция `validate_batch(domains)`:**

**Процесс:**
1. Создает асинхронные tasks для всех доменов
2. Использует `asyncio.as_completed()` для получения results по мере готовности
3. Выводит progress каждые 5 секунд:
   - Progress: X/Y (%)
   - QPS (queries per second)
   - ETA (estimated time)
   - Success/Errors/Timeouts counters
4. Возвращает список DNSResult объектов

**Финальная статистика:**
- Total time
- Average QPS
- Success rate (%)

**Целевой QPS:** 100-500 queries/second (для local Unbound)

### 6.6 Chunked validation (для больших датасетов)

**Функция `validate_batch_chunked(domains, chunk_size=10000, output_path=None)`:**

Для датасетов 100k+ доменов с управлением памятью.

**Процесс:**
1. Разбивает домены на чанки по `chunk_size` (default 10k)
2. Для каждого чанка:
   - Валидирует через `validate_batch()`
   - Конвертирует results в DataFrame
   - Преобразует lists (ns_records, a_records) в JSON strings
   - Пишет chunk incrementally в CSV (если `output_path` указан)
   - **CRITICAL:** Освобождает память (`del`, `gc.collect()`)
   - **CRITICAL:** Очищает DNS resolver pool для release connections
3. Возвращает список results или пустой список (если на диске)

**Memory safeguards:**
- Предупреждение если `chunk_size` >50k
- Рекомендуемый размер: 5k-15k доменов
- Force garbage collection между чанками
- Очистка resolver pool

**Auto-detection:**
- Используется автоматически для >50k доменов

### 6.7 DNSValidationPipeline (главный workflow)

**Класс DNSValidationPipeline:**

Обертка над DNSValidator с полным reporting и output management.

**Функция `run()`:**

**Параметры:**
```python
run(
    predictions_file,       # CSV с predictions
    output_dir,             # Output директория
    confidence_threshold=0.8,
    filter_gambling=True,   # Фильтровать только gambling
    chunk_size=None,        # Default 10000
    use_chunked=None        # Auto-detect
)
```

**Workflow:**
1. Загружает predictions из CSV
2. Фильтрует:
   - Если `filter_gambling=True`: `prediction==1` AND `confidence>=threshold`
   - Если `filter_gambling=False`: только `confidence>=threshold`
3. Auto-detect chunked mode (если >50k доменов)
4. Запускает validation (chunked или standard)
5. Вычисляет статистику:
   - Total domains
   - Active domains (not parked, successful query)
   - Parked domains
   - Successful queries
   - Timeout/NXDOMAIN/Error queries
   - Success rate %
   - Parking rate %
   - Active rate %
   - Average query time
   - Parking providers breakdown
6. Сохраняет outputs:
   - `dns_validation.csv` - полные results
   - `active_gambling_domains.txt` - список active доменов
   - `parked_gambling_domains.txt` - список parked доменов
   - `dns_validation_stats.json` - статистика
7. Выводит summary в консоль

**Chunked mode особенности:**
- Статистика вычисляется из CSV файла (streaming, без загрузки в память)
- Domain lists извлекаются из CSV (streaming, chunk by chunk)
- Функции `_calculate_stats_from_file()` и `_save_domain_lists_from_file()`

### 6.8 Производительность DNS модуля

**Целевые метрики:**
- Success rate: ≥95%
- DNS queries reduction: ≥60% доменов отфильтровано через ML перед DNS
- Parking detection accuracy: ≥90%
- Timeout: 10 секунд (увеличен для reliability)

**Фактическая производительность с Unbound:**
- Concurrency: 1500 параллельных запросов
- QPS: 500-1500 queries/second (зависит от сети)
- Resolver pool: 50 instances для балансировки

## 7. ИНТЕГРАЦИЯ МОДУЛЕЙ И ПОЛНЫЙ PIPELINE

### 7.1 Скрипт run_full_pipeline.py

Выполняет полный цикл обучения:

```
1. Data Collection:
   - fetch_gambling_domains()
   - fetch_benign_domains()
   - create_dataset() → train.csv, test.csv

2. Feature Engineering:
   - DomainFeatureExtractor.fit_tfidf(train_domains)
   - create_feature_matrix() → X_train, X_test

3. Model Training:
   - GamblingDomainClassifier.train(X_train, y_train)
   - tune_hyperparameters() (GridSearchCV)

4. Model Evaluation:
   - evaluate(X_test, y_test)
   - plot_confusion_matrix(), plot_roc_curve()
   - plot_feature_importance()

5. Model Saving:
   - save() → models/logistic_regression.joblib
   - save feature extractor → models/tfidf_vectorizer.joblib
```

### 7.2 Inference workflow (scripts/03_run_inference.py)

```
1. Load domains from file (data/raw/unlabeled_test.txt)

2. Initialize DomainClassifierInference:
   - Load model
   - Load feature extractor

3. Predict:
   - predict_batch_chunked() для больших датасетов
   - predict_batch() для маленьких

4. Save results:
   - predictions.csv
   - predictions_confident_80.csv
   - predicted_gambling.csv

5. Analyze:
   - analyze_predictions() → prediction_analysis.png
```

### 7.3 DNS validation workflow (scripts/04_validate_dns.py)

```
1. Initialize DNSValidator:
   - timeout=10.0
   - concurrency=1500
   - resolver_pool_size=50

2. Initialize DNSValidationPipeline(validator)

3. Run validation:
   - Load predictions.csv
   - Filter gambling domains (confidence ≥0.8)
   - validate_batch_chunked() для больших датасетов
   - validate_batch() для маленьких

4. Save results:
   - dns_validation.csv
   - active_gambling_domains.txt
   - parked_gambling_domains.txt
   - dns_validation_stats.json

5. Print summary
```

## 8. ТЕХНИЧЕСКИЕ ДЕТАЛИ ОБРАБОТКИ ДОМЕНОВ

### 8.1 Domain Normalization Pipeline

```
Raw input → clean_domain() → validate_domain() → normalize_domain() → Feature extraction
```

**Пример обработки:**
```
Input:  "https://www.subdomain.example.co.uk:8080/path"
↓ clean_domain()
"subdomain.example.co.uk"
↓ validate_domain()
True (valid domain)
↓ normalize_domain()
"example.co.uk"
↓ Feature extraction
{domain_length: 13, has_digits: 0, tld_uk: 1, ...}
```

### 8.2 Feature Matrix Structure

Итоговая структура (примерно 550 колонок):

```
Manual Features (45-50 колонок):
├── Keyword features (25 колонок)
│   ├── keyword_casino, keyword_bet, ...
│   ├── total_keyword_count
│   └── has_gambling_keyword
├── TLD features (15 колонок)
│   ├── tld_bet, tld_casino, tld_com, ...
└── Structural features (10 колонок)
    ├── domain_length, dot_count, digit_count, ...
    └── vowel_ratio, max_consecutive_chars, ...

TF-IDF N-gram Features (500 колонок):
├── ngram_0: "ca" (TF-IDF score)
├── ngram_1: "cas" (TF-IDF score)
├── ...
└── ngram_499: "xyz" (TF-IDF score)
```

**Размерность:**
- Training set: (127k samples, 550 features)
- Test set: (32k samples, 550 features)

**Memory footprint:**
- Dense matrix: ~250 MB (для 159k samples)
- Sparse matrix: ~15-25 MB (10x reduction)

### 8.3 Inference Processing Modes

**Standard mode (датасеты ≤100k доменов):**
```
Domains → normalize_domain() → create_feature_matrix()
   ↓
Dense DataFrame (all in memory)
   ↓
model.predict() → predictions DataFrame
   ↓
Save to CSV
```

**Chunked mode (датасеты >100k доменов):**
```
Domains → split into chunks (50k each)
   ↓
For each chunk:
   ├── normalize_domain()
   ├── create_feature_matrix_sparse() → sparse matrix
   ├── model.predict()
   ├── Write chunk to CSV (incremental)
   └── Free memory (gc.collect)
   ↓
Final CSV on disk
```

### 8.4 DNS Validation Processing Modes

**Standard mode (датасеты ≤50k доменов):**
```
Domains → validate_batch() (all async tasks)
   ↓
List[DNSResult] in memory
   ↓
Convert to DataFrame → Save to CSV
   ↓
Calculate stats, extract lists
```

**Chunked mode (датасеты >50k доменов):**
```
Domains → split into chunks (10k each)
   ↓
For each chunk:
   ├── validate_batch() → List[DNSResult]
   ├── Convert to DataFrame
   ├── Write chunk to CSV (incremental)
   ├── Free memory (del, gc.collect)
   └── Clear resolver pool
   ↓
Calculate stats from CSV (streaming)
   ↓
Extract lists from CSV (streaming)
```

## 9. ОПТИМИЗАЦИИ И BEST PRACTICES

### 9.1 Memory Management

**Sparse matrices для features:**
- Используются в `create_feature_matrix_sparse()`
- Экономия: 10-20x для TF-IDF features
- Важно для датасетов >1M доменов

**Chunked processing:**
- Inference: `chunk_size=50k` domains
- DNS validation: `chunk_size=10k` domains
- Incremental CSV writing
- Explicit memory cleanup: `del`, `gc.collect()`

**DNS resolver pool cleanup:**
- Очистка пула между чанками
- Предотвращает утечки connections
- Критично для >100k доменов

### 9.2 Performance Optimizations

**Parallel processing:**
- GridSearchCV: `n_jobs=-1` (all CPU cores)
- Random Forest: `n_jobs=8` (M1 Max optimization)
- DNS queries: `concurrency=1500`
- DNS resolver pool: 50 instances

**Async operations:**
- DNS queries: полностью async через aiodns
- Параллельные NS/A/CNAME queries через `asyncio.gather()`
- Round-robin resolver balancing

**Caching:**
- TF-IDF vectorizer: fit once, reuse for inference
- Feature extractor: save/load для consistency

### 9.3 Error Handling

**DNS validation:**
- Retry logic: до 2 попыток с 0.1 сек delay
- Timeout handling: configurable (default 10 сек)
- Exception handling в `asyncio.gather(return_exceptions=True)`
- Статистика errors/timeouts/nxdomain

**Inference:**
- Feature alignment: test features aligned to train schema
- Missing columns: filled with 0
- Sparse matrix warnings: suppressed

### 9.4 Warnings and Safeguards

**DNS validation:**
- Warning если `chunk_size` >50k
- Рекомендация: 5k-15k domains per chunk
- Memory critical safeguards в chunked mode

**Inference:**
- Auto-detection chunked mode при >100k domains
- Skip visualization при >500k domains

## 10. ФАЙЛОВАЯ СТРУКТУРА И ФОРМАТЫ

### 10.1 Структура директорий

```
ML-Domain/
├── data/
│   ├── raw/
│   │   ├── gambling_domains.txt
│   │   ├── benign_domains.txt
│   │   └── unlabeled_test.txt
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
│   └── results/
│       ├── predictions.csv
│       ├── predictions_confident_80.csv
│       ├── predicted_gambling.csv
│       ├── dns_validation.csv
│       ├── active_gambling_domains.txt
│       └── parked_gambling_domains.txt
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib (optional)
│   ├── tfidf_vectorizer.joblib
│   ├── feature_metadata.joblib
│   ├── model_metrics.json
│   ├── model_metrics_rf.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   └── top_features.csv
├── src/
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── inference.py
│   ├── dns_validation.py
│   └── utils.py
└── scripts/
    ├── 01_collect_data.py
    ├── 02_train_model.py
    ├── 03_run_inference.py
    ├── 04_validate_dns.py
    └── run_full_pipeline.py
```

### 10.2 CSV форматы

**train.csv / test.csv:**
```csv
domain,label
example.com,0
casino365.bet,1
```

**predictions.csv:**
```csv
original_domain,domain,prediction,probability_benign,probability_gambling,confidence,label
www.bet365.com,bet365.com,1,0.05,0.95,0.95,gambling
google.com,google.com,0,0.98,0.02,0.98,benign
```

**dns_validation.csv:**
```csv
domain,ns_records,a_records,cname_records,is_parked,parking_provider,query_status,query_time
bet365.com,"[""ns1.bet365.com""]","[""104.16.1.1""]","[]",false,,success,0.123
parked-domain.com,"[""ns1.sedoparking.com""]","[""91.195.240.12""]","[]",true,sedoparking.com,success,0.089
```

**active_gambling_domains.txt / parked_gambling_domains.txt:**
```
bet365.com
pokerstars.net
...
```

### 10.3 JSON форматы

**model_metrics.json:**
```json
{
  "accuracy": 0.9229,
  "roc_auc": 0.9645,
  "precision_gambling": 0.9536,
  "recall_gambling": 0.8886,
  "f1_gambling": 0.9199,
  "true_negatives": 15234,
  "false_positives": 723,
  "false_negatives": 1754,
  "true_positives": 13942
}
```

**dns_validation_stats.json:**
```json
{
  "total_domains": 10000,
  "active_domains": 6500,
  "parked_domains": 2800,
  "successful_queries": 9300,
  "timeout_queries": 500,
  "nxdomain_queries": 150,
  "error_queries": 50,
  "success_rate_pct": 93.0,
  "parking_rate_pct": 30.1,
  "active_rate_pct": 69.9,
  "avg_query_time": 0.123,
  "parking_providers": {
    "sedoparking.com": 1200,
    "bodis.com": 800,
    "IP:91.195.240.12": 600
  }
}
```

## 11. КЛЮЧЕВЫЕ МЕТРИКИ И ТРЕБОВАНИЯ

### 11.1 Model Performance Requirements

- ✓ Accuracy ≥ 92%
- ✓ Precision (gambling) ≥ 90%
- ✓ Recall (gambling) ≥ 85%
- ✓ Inference speed ≥ 50,000 domains/second
- ✓ Model size < 50 MB (Logistic Regression: ~5 MB)
- ✓ Memory footprint < 500 MB

### 11.2 DNS Validation Requirements

- ✓ Success rate ≥ 95%
- ✓ Parking detection accuracy ≥ 90%
- ✓ DNS queries reduction ≥ 60% (ML filtering)
- ✓ Timeout: 3-10 seconds
- ✓ Concurrency: 100-1500 queries

### 11.3 Dataset Requirements

- ✓ Training: 2,000-5,000 domains per class
- ✓ Test: 10,000-20,000 unlabeled domains
- ✓ Train/test split: 80/20
- ✓ Class balance: handled via `class_weight='balanced'`