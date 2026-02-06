# 📚 Dokumentacja Projektu: Edge of Chaos (HH Reservoir)
**Ostatnia aktualizacja:** 2026-02-06

Ten folder zawiera rygorystyczną metodologię i wyniki badań nad hipotezą "Edge of Chaos" w rezerwuarach spikingowych opartych na modelu Hodgkin-Huxley.

## 🗂️ Struktura i Pliki

### 1. Strategia i Design
- **[PUBLICATION_STRATEGY_V2.md](./PUBLICATION_STRATEGY_V2.md)**: Główne źródło prawdy. Zawiera 8-tygodniowy plan publikacji i analizę wyników.
- **[RESEARCH_DESIGN.md](./RESEARCH_DESIGN.md)**: Opis hipotez badawczych i mapy eksperymentalnej.
- **[EXPERIMENTAL_SYNOPSIS.md](./EXPERIMENTAL_SYNOPSIS.md)**: Krótkie podsumowanie techniczne parametrów.

### 2. Wyniki (Kluczowe Figury)
- **[STORYTELLER_RESULTS.png](./STORYTELLER_RESULTS.png)**: Figura główna. Łączy wydajność MC/NARMA z wykładnikiem Lapunowa. **(AKTUALNY DOWÓD)**
- **[REGIME_COMPARISON.png](./REGIME_COMPARISON.png)**: Porównanie tripletów (Stable, Edge, Chaotic). Pokazuje szczyt wydajności na granicy chaosu.

### 3. Pipeline Obliczeniowy
- **`benchmark_mc.py`**: Nowa, poprawiona wersja obliczeń Memory Capacity (bez błędu `np.roll`).
- **`compare_regimes.py`**: Skrypt do porównywania tripletów stanów.
- **`generate_storyteller_plot.py`**: Skrypt generujący figurę główną do artykułu.
- **`run_batch_protocol.py`**: Pipeline do masowego sprawdzania wielu ziaren (seeds).
- **`task_config.yaml`**: Centralna konfiguracja parametrów biologicznych i symulacyjnych.

## �️ Jak odtworzyć wyniki?

1.  **Instalacja:** Upewnij się, że masz środowisko z `numpy`, `scipy`, `scikit-learn` i `matplotlib` (np. środowisko `base`).
2.  **Kolejność uruchamiania:**
    - `python benchmark_mc.py` -> weryfikacja stabilności pamięci.
    - `python compare_regimes.py` -> wygenerowanie porównania Tripletu.
    - `python generate_storyteller_plot.py` -> stworzenie figury do publikacji.

## 🔬 Metodologia (Rygor)
Wszystkie obliczenia w tym folderze zostały zweryfikowane pod kątem:
- **Poprawności matematycznej MC**: Użycie korelacji bez naruszania przyczynowości.
- **Stabilności Lapunowa**: Synchronizacja wszystkich 6 bramek/zmiennych HH ($V, m, h, n, a, b$).
- **Statystyki**: Multi-seed validation.

---
**Status:** Gotowe do etapu pisania manuskryptu.
