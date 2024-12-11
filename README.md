# 1. Projekt Klasyfikacja Emocji

## Opis Projektu
Projekt dotyczył analizy sentymentu za pomocą modelu BERT (Bidirectional Encoder Representations from Transformers). Zbiór danych składał się z tekstów przypisanych do sześciu klas emocji: *surprise*, *sadness*, *love*, *joy*, *fear* oraz *anger*. 

Dane wstępnie przetworzono przy użyciu narzędzi takich jak **BertTokenizer** do tokenizacji tekstu oraz **LabelEncoder** do kodowania etykiet. Model BERT został wytrenowany przez 10 epok, osiągając następujące wyniki:
- **Dokładność na zbiorze treningowym**: 98.94%
- **Dokładność na zbiorze testowym**: 92.55%
- **Wartość funkcji straty na zbiorze testowym**: 0.288

## Dane wejściowe
Dane wejściowe to krótkie zdania przypisane do odpowiednich kategorii emocji. Zbiór danych podzielono na dwie części:
- **Zbiór treningowy**: 16,000 przykładów
- **Zbiór testowy**: 2,000 przykładów

## Proces uczenia
### Tokenizacja
- Teksty zostały przetworzone przy użyciu tokenizer’a BERT-a, co umożliwiło konwersję danych tekstowych na wektory wejściowe dla modelu.

### Funkcja straty
- W projekcie wykorzystano funkcję **CrossEntropyLoss**, odpowiednią dla klasyfikacji wieloklasowej.

### Optymalizacja
- Użyto optymalizatora **AdamW** z małym współczynnikiem uczenia (**2e-5**), aby zapewnić stabilność treningu.

### Uruchomienie
- Model był trenowany przez **10 epok** z użyciem GPU, co przyspieszyło proces treningu.

## Wyniki
Model osiągnął wysoką dokładność zarówno na danych treningowych, jak i testowych:
- **Dokładność na zbiorze treningowym**: 98.94%
- **Dokładność na zbiorze testowym**: 92.55%
- **Wartość funkcji straty na zbiorze testowym**: 0.2588

## Wykorzystane technologie
- **Język programowania**: Python
- **Biblioteki**: Pandas, Scikit-learn, PyTorch
- **Model**: BERT

## Wnioski
Projekt demonstruje skuteczne zastosowanie zaawansowanego modelu językowego do rozpoznawania emocji w tekście. Osiągnięte wyniki sugerują, że model BERT jest odpowiedni do rzeczywistych zastosowań, zachowując dobrą równowagę między dokładnością a uniwersalnością. Może być używany w aplikacjach wymagających analizy sentymentu i rozpoznawania emocji w tekstach.

## Link: [Classification Emotion Notebook](https://github.com/Gabsonenene/PyTorch-and-OpenCV/blob/main/Classification_Emotion.ipynb)


# 2. Projekt Klasyfikacja Znaków Drogowych

## Opis Projektu
Projekt polegał na detekcji i klasyfikacji znaków drogowych przy użyciu biblioteki **OpenCV** oraz głębokich sieci neuronowych. Wykorzystano zbiór danych **German Traffic Sign Recognition Benchmark (GTSRB)**, zawierający zdjęcia znaków drogowych przypisanych do **43 klas**. Celem projektu było stworzenie modelu, który poprawnie rozpoznaje znaki drogowe po wstępnym przetworzeniu obrazów.

## Etapy projektu
### Przygotowanie danych
1. **Załadowanie danych**:
   - Wczytanie obrazów i ich etykiet z zestawu danych GTSRB.
2. **Przetwarzanie obrazów**:
   - Zmiana rozmiaru obrazów.
   - Konwersja obrazów na skalę szarości.
   - Normalizacja pikseli.
   - Augmentacja danych: rotacja, odbicia lustrzane, rozmycie Gaussowskie.
3. **Podział danych**:
   - Dane podzielono na zbiory treningowy i testowy w proporcji **80:20**.

### Implementacja modelu
1. **Architektura sieci neuronowej**:
   - Model zbudowano jako **konwolucyjną sieć neuronową (CNN)**.
   - Architektura:
     - **4 warstwy konwolucyjne**.
     - **2 w pełni połączone warstwy**.
2. **Techniki wspomagające**:
   - Funkcja aktywacji **Leaky ReLU**.
   - **Batch Normalization** dla stabilizacji uczenia.
   - **Dropout** w celu zapobiegania przeuczeniu.
3. **Funkcja kosztu**:
   - **CrossEntropyLoss**, odpowiednia dla klasyfikacji wieloklasowej.
4. **Optymalizacja**:
   - Użyto optymalizatora **Adam** z dynamiczną regulacją szybkości uczenia.

### Trening i testowanie
- Model był trenowany przez **20 epok**.
- Osiągnięte wyniki:
  - **Dokładność na zbiorze treningowym**: 98,81%.
  - **Strata na zbiorze treningowym**: 0,0404.
  - **Dokładność na zbiorze testowym**: 87,22%.

## Wykorzystane technologie
- **Język programowania**: Python
- **Biblioteki**: OpenCV, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib
- **Model**: Konwolucyjna sieć neuronowa (CNN)

## Wnioski
Model wykazał wysoką skuteczność w klasyfikacji znaków drogowych, osiągając **98,81% dokładności** na zbiorze treningowym i **87,22% dokładności** na zbiorze testowym. Wyniki te pokazują potencjał zastosowania CNN w systemach wspomagania kierowcy (ADAS). Projekt demonstruje, że głębokie sieci neuronowe są efektywnym narzędziem w rozpoznawaniu obrazów w rzeczywistych zastosowaniach.

## Link: [GTSRB OpenCV Notebook](https://github.com/Gabsonenene/PyTorch-and-OpenCV/blob/main/GTSRB%20OpenCV.ipynb)


# 3. Projekt: Model Predykcji Kolejnego Słowa  

## Opis projektu  
Celem projektu było stworzenie modelu do przewidywania kolejnego słowa w zdaniu, co jest podstawowym zadaniem w dziedzinie przetwarzania języka naturalnego (NLP). Wykorzystano sieci neuronowe z warstwami LSTM (Long Short-Term Memory), które są efektywne w analizie sekwencji. Model został wytrenowany na przetworzonym zbiorze danych tekstowych, a następnie wykorzystany do generowania przewidywań kolejnych słów.  

## Etapy projektu  

### 1. Przygotowanie danych
- Wczytanie tekstu z dostępnego zbioru danych.
- Wstępne przetwarzanie: usuwanie znaków specjalnych i nieistotnych danych.
- Tokenizacja: zamiana tekstu na sekwencje liczb.
- Tworzenie zestawów danych treningowych w formie par: sekwencja wejściowa i docelowe słowo.

### 2. Implementacja modelu
- Wykorzystano warstwy `Embedding` do reprezentacji wyrazów w formie wektorowej.
- Sieć zawiera warstwy LSTM dla przetwarzania sekwencji oraz warstwę Dense do klasyfikacji następnego słowa.
- Zastosowano funkcję straty CrossEntropyLoss i optymalizator Adam.

### 3. Proces uczenia
- Model był trenowany przez 20 epok z zastosowaniem GPU, aby przyspieszyć proces uczenia.
- Przygotowano wykresy strat i dokładności w celu monitorowania efektywności treningu.

### 4. Testowanie i generowanie wyników
- Model został przetestowany na zestawie testowym, gdzie wygenerowano przewidywania kolejnych słów dla różnych przykładów.
- Analiza wyników wskazuje na wysoką skuteczność w generowaniu logicznych kontynuacji zdań.

## Wykorzystane technologie
- **Język programowania**: Python  
- **Biblioteki**: PyTorch, NumPy, Pandas, Matplotlib  
- **Model**: LSTM z warstwami osadzania (embedding).  

## Wnioski  
Projekt pokazuje skuteczność LSTM w przetwarzaniu sekwencji i przewidywaniu następnego słowa. Wyniki modelu wskazują na możliwość zastosowania w rzeczywistych scenariuszach, takich jak autouzupełnianie tekstu czy generowanie treści.

## Link: [Next Word Prediction Notebook](https://github.com/Gabsonenene/PyTorch-and-OpenCV/blob/main/Next%20Word%20Prediction%20Model.ipynb)
