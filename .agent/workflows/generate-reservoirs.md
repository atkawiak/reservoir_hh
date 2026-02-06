---
description: Jak wygenerować Triplety Rezerwuaru (Stable, Edge, Chaotic)
---

Ten potok (pipeline) służy do automatycznego generowania i testowania rezerwuarów Hodgkin-Huxleya w trzech kluczowych reżimach dynamicznych.

### 1. Przygotowanie Konfiguracji
Upewnij się, że plik `stage_D_rigorous/task_config.yaml` zawiera poprawne parametry bazowe (N=100, dt=0.05ms, itp.).

### 2. Generowanie Tripletów
Upewnij się, że używasz środowiska conda `shelf-recognition-llm-hub`. Skrypt automatycznie znajdzie punkt krytyczny dla każdego ziarna (seed) i przetestuje trzy stany:
- **STABLE**: Silna inhibicja (λ < 0)
- **EDGE**: Granica chaosu (λ ≈ 0)
- **CHAOTIC**: Słaba inhibicja (λ > 0)

// turbo
```bash
conda run -n shelf-recognition-llm-hub python3 stage_D_rigorous/generate_reservoir_triplets.py
```

### 3. Analiza Wyników
Wyniki zostaną zapisane w pliku `reservoir_triplets_results.csv`. Skrypt wyświetli również podsumowanie średnich metryk dla każdego reżimu:
- **NARMA NRMSE**: Niższe jest lepsze.
- **XOR Accuracy**: Wyższe jest lepsze.
- **MC bits**: Pojemność pamięci (wyższa jest lepsza).

### 4. Lokalizacja punktu EDGE
Skrypt używa metody `brentq` do precyzyjnego znalezienia wartości `inh_scaling`, przy której wykładnik Lapunowa λ przecina zero. Jest to najbardziej rygorystyczna metoda wyznaczania Granicy Chaosu.
