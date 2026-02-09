# Metoda Hybrydowa Kalibracji Rezerwuarów Hodgkin-Huxley
## Surgical Precision: Dense Scan + Brentq Refinement

---

## 1. WPROWADZENIE

### Problem
Klasyczne metody kalibracji rezerwuarów (np. normalizacja promienia spektralnego do ρ=1.0) nie działają dla biologicznie realistycznych neuronów Hodgkin-Huxley (HH). Powód:

1. **Złożona dynamika nieliniowa**: Neurony HH wykazują "górkę dynamiki" zamiast prostej monotonicznej zależności między parametrami kontrolnymi a wykładnikiem Lapunowa (Λ).
2. **Wąskie okno przejściowe**: Krawędź Chaosu (Edge of Chaos) w HH jest ekstremalnie wąska (Δinh ≈ 0.02), co sprawia, że tradycyjne metody grid search (co 0.5-1.0) całkowicie ją pomijają.
3. **Dwuznaczność stabilności**: System może być stabilny zarówno przy bardzo niskiej inhibicji (nasycenie synchroniczne) jak i bardzo wysokiej (tłumienie).

### Rozwiązanie: Metoda Hybrydowa
Łączymy zalety dwóch podejść:
- **Dense Grid Search (100 punktów)**: Gwarantuje, że nie przegapimy żadnej struktury dynamicznej.
- **Brentq Root-Finding**: Zapewnia chirurgiczną precyzję w lokalizacji punktu Λ=0.

---

## 2. ARCHITEKTURA PIPELINE'U

### 2.1 Wejście
- **Plik konfiguracyjny**: `task_config.yaml`
  - Parametry neuronów HH (V_rest, g_Na, g_K, etc.)
  - Parametry synaps (g_exc, g_inh, tau_syn)
  - Ustawienia dynamiki (target_spectral_radius=4.0, inh_scaling)
  - Parametry benchmarków (MC, NARMA, XOR)

- **Parametry eksperymentu**:
  - `n_neurons`: Rozmiar sieci (100, 200, 500)
  - `n_seeds`: Liczba niezależnych realizacji (30)

### 2.2 Proces Kalibracji (dla każdego seed)

#### KROK 1: Dense Grid Search (Zwiad)
```python
scan_inhs = np.linspace(0.5, 12.0, 100)  # 100 punktów
scan_lambdas = []

for inh in scan_inhs:
    config['synapse']['inh_scaling'] = inh
    res = Reservoir(n_neurons, config)
    res.normalize_spectral_radius(4.0)  # Wysoka energia bazowa
    lambda_val = calculate_lyapunov(res, n_steps=1200, seed=seed)
    scan_lambdas.append(lambda_val)
```

**Cel**: Zmapowanie pełnego profilu dynamiki w zakresie inhibicji od 0.5 do 12.0.

**Wynik**: 
- Tablica 100 wartości Λ(inh)
- Identyfikacja szczytu (peak_idx): `argmax(scan_lambdas)`

#### KROK 2: Lokalizacja Przedziału Przejściowego
```python
peak_idx = np.argmax(scan_lambdas)
peak_inh = scan_inhs[peak_idx]

# Szukamy zmiany znaku PO PRAWEJ stronie szczytu
brent_interval = None
for i in range(peak_idx, len(scan_lambdas)-1):
    if scan_lambdas[i] * scan_lambdas[i+1] <= 0:
        brent_interval = (scan_inhs[i], scan_inhs[i+1])
        break
```

**Logika**: Krawędź Chaosu znajduje się na prawym zboczu "górki", gdzie system przechodzi z chaosu (Λ>0) do stabilności (Λ<0).

#### KROK 3: Surgical Zoom (Brentq)
```python
if brent_interval:
    try:
        edge_inh = brentq(get_lambda, 
                         brent_interval[0], 
                         brent_interval[1], 
                         xtol=0.01)
    except ValueError:
        # Recovery: wybierz punkt bliższy zeru
        la = get_lambda(brent_interval[0])
        lb = get_lambda(brent_interval[1])
        edge_inh = brent_interval[0] if abs(la) < abs(lb) else brent_interval[1]
```

**Cel**: Znalezienie dokładnego punktu, w którym Λ=0 z precyzją xtol=0.01.

**Mechanizm odporności**: Jeśli brentq zawiedzie (szum numeryczny), system automatycznie wybiera najlepszy z dwóch punktów brzegowych.

#### KROK 4: Definicja Reżimów
```python
triplets = [
    ('STABLE', edge_inh + 4.0),    # Głęboka stabilność
    ('EDGE', edge_inh),             # Λ ≈ 0 (krawędź chaosu)
    ('CHAOTIC', peak_inh)           # Szczyt dynamiki (max Λ)
]
```

### 2.3 Benchmarking
Dla każdego z 3 reżimów:
1. **Memory Capacity (MC)**: Zdolność do zapamiętywania przeszłych sygnałów
2. **NARMA-10**: Nieliniowe zadanie predykcji szeregu czasowego
3. **XOR**: Klasyfikacja nieliniowa
4. **Kernel Rank**: Rząd macierzy stanów (bogactwo reprezentacji)
5. **Lempel-Ziv Complexity**: Złożoność ciągów impulsów
6. **Separation Property**: Odległość między stanami dla różnych wejść

### 2.4 Wyjście
- **Plik CSV**: `RESULTS_RIGOROUS_N{n_neurons}.csv`
  - Kolumny: N, Seed, Regime, Inh_Scaling, Lambda, MC, NARMA, XOR, KernelRank, LZ_Complexity, Separation
  - 3 wiersze na seed (STABLE, EDGE, CHAOTIC)
  - 90 wierszy dla 30 seedów

- **Wykres**: `PLOT_RIGOROUS_N{n_neurons}.png`
  - 6 paneli (MC, NARMA, XOR, KernelRank, LZ, Separation)
  - Barploty z podziałem na reżimy

---

## 3. KLUCZOWE INNOWACJE

### 3.1 Wysoka Energia Bazowa (ρ=4.0)
**Problem**: Przy standardowym ρ=0.95-1.0, neurony HH są zawsze stabilne.

**Rozwiązanie**: Ustawiamy target_spectral_radius=4.0, co wprowadza system w naturalnie chaotyczny stan. Dopiero wtedy inhibicja (inh_scaling) może go "hamować" i przesuwać przez krawędź chaosu.

**Uzasadnienie biologiczne**: W mózgu siła połączeń (wagi synaptyczne) jest wysoka, a balans E/I (excitation/inhibition) jest głównym mechanizmem kontroli dynamiki.

### 3.2 Gęste Sito (100 punktów)
**Dlaczego nie 10 czy 20?**
- Przejście przez krawędź chaosu w HH może być węższe niż 0.1 jednostki inhibicji
- 100 punktów w zakresie 0.5-12.0 daje rozdzielczość ~0.12, co gwarantuje wykrycie

**Koszt obliczeniowy**: 
- 100 symulacji × 1200 kroków × 30 seedów ≈ 3.6M kroków na wielkość sieci
- Czas: ~15-20 minut na seed dla N=100 (równolegle 4 seedy)

### 3.3 Deterministyczna Powtarzalność
```python
def get_lambda(inh):
    np.random.seed(seed)  # Każde wywołanie używa tego samego ziarna
    ...
```

**Cel**: Eliminacja szumu numerycznego między kolejnymi wywołaniami tej samej funkcji w ramach brentq.

### 3.4 Mechanizm Recovery
Jeśli brentq napotka problem (f(a) i f(b) mają ten sam znak mimo wcześniejszej detekcji):
1. Oblicz Λ dla obu punktów brzegowych
2. Wybierz ten, który jest bliżej zeru
3. Kontynuuj bez przerywania całego procesu

---

## 4. INTERPRETACJA WYNIKÓW

### 4.1 Oczekiwane Trendy

#### Metryki Strukturalne (rosną z chaosem):
- **Kernel Rank**: STABLE < EDGE < CHAOTIC
- **Separation**: STABLE < EDGE < CHAOTIC
- **LZ Complexity**: Może być stabilna lub lekko rosnąca

#### Metryki Wydajnościowe (szczyt na EDGE):
- **NARMA**: STABLE ≈ CHAOTIC < **EDGE** (szczyt)
- **XOR**: STABLE ≈ CHAOTIC < **EDGE** (szczyt)
- **MC**: Może mieć szczyt w STABLE lub EDGE, ale spada w CHAOTIC

### 4.2 Hipoteza Badawcza
**H0**: Nie ma różnic w wydajności między reżimami.

**H1**: Reżim EDGE (Λ≈0) osiąga najwyższą wydajność w zadaniach obliczeniowych (NARMA, XOR), podczas gdy metryki strukturalne (Kernel Rank, Separation) rosną monotoniczne z chaosem.

### 4.3 Testy Statystyczne
1. **One-way ANOVA**: Czy istnieją różnice między reżimami?
2. **Paired t-test**: Czy EDGE jest istotnie lepszy niż STABLE i CHAOTIC?
   - Używamy paired, bo porównujemy ten sam seed w różnych stanach

---

## 5. PRZYKŁADOWE WYNIKI (Seed 101, N=100)

### Kalibracja:
```
[ZOOM] Found crossing in interval (9.21, 9.33). Refining...
[SURGICAL] CHAOTIC(Peak)=9.21, EDGE(Zero)=9.23, STABLE=12.00
```

### Wydajność:
| Regime | Lambda | NARMA | XOR | MC | KernelRank | Separation |
|--------|--------|-------|-----|----|-----------:|------------|
| STABLE | -0.028 | 0.794 | 0.47| 0.137| 0.60 | 0.555 |
| **EDGE** | **0.003** | **0.801** | **0.55**| 0.103| 0.64 | 0.696 |
| CHAOTIC| 0.132 | 0.788 | 0.55| 0.083| 0.64 | 0.711 |

**Obserwacje**:
- EDGE ma najwyższą wydajność w NARMA i XOR
- Kernel Rank i Separation rosną w stronę chaosu
- MC spada drastycznie w CHAOTIC (utrata pamięci)

---

## 6. URUCHOMIENIE PIPELINE'U

### Krok 1: Przygotowanie środowiska
```bash
cd f:\OneDrive\Dokumenty\reservoir_hh\stage_D_rigorous
```

### Krok 2: Weryfikacja konfiguracji
Sprawdź `task_config.yaml`:
```yaml
dynamics_control:
  target_spectral_radius: 4.0  # KLUCZOWE!
synapse:
  inh_scaling: 4.0  # Wartość startowa (będzie skanowana)
```

### Krok 3: Uruchomienie
```bash
# Pojedynczy test (1 seed)
py -u generate_reservoir_triplets.py 100 1

# Pełna produkcja (30 seedów, 4 równolegle)
py -u generate_reservoir_triplets.py 100 30 > log_N100.log 2>&1 &
py -u generate_reservoir_triplets.py 200 30 > log_N200.log 2>&1 &
py -u generate_reservoir_triplets.py 500 30 > log_N500.log 2>&1 &
```

### Krok 4: Monitorowanie
```bash
# Sprawdź postęp
Get-Content log_N100.log -Tail 20

# Sprawdź wyniki
Get-Content RESULTS_RIGOROUS_N100.csv | Select-Object -Last 10
```

### Krok 5: Analiza statystyczna
```bash
py statistical_proof.py
```

---

## 7. WYMAGANIA SYSTEMOWE

### Sprzęt:
- CPU: 4+ rdzenie (zalecane 8)
- RAM: 8GB minimum (16GB zalecane dla N=500)
- Dysk: 1GB wolnego miejsca

### Oprogramowanie:
- Python 3.8+
- numpy, scipy, pandas, matplotlib, seaborn
- Własne moduły: src.model.reservoir, src.utils.metrics

### Czas wykonania (szacunkowy):
- N=100, 30 seedów: ~8-10 godzin (4 równolegle)
- N=200, 30 seedów: ~15-20 godzin
- N=500, 30 seedów: ~40-50 godzin

---

## 8. CYTOWANIE

Jeśli używasz tej metody w publikacji, prosimy o cytowanie:

```
Hybrid Calibration Method for Hodgkin-Huxley Reservoir Computing:
Dense Grid Search with Brentq Refinement for Surgical Precision 
in Edge of Chaos Identification (2026)
```

---

## 9. KONTAKT I WSPARCIE

W razie pytań lub problemów:
- Sprawdź logi w `log_N*.log`
- Sprawdź błędy w `experiment_N*_err.log`
- Zweryfikuj, czy wszystkie procesy Python działają: `Get-Process python*`

---

## 10. HISTORIA ZMIAN

### v3.0 (2026-02-08) - Surgical Precision
- Wprowadzono Dense Grid Search (100 punktów)
- Dodano Brentq refinement dla precyzyjnej lokalizacji Λ=0
- Zwiększono target_spectral_radius do 4.0
- Dodano mechanizm recovery dla brentq
- Dodano 3 nowe metryki: KernelRank, LZ_Complexity, Separation

### v2.0 (2026-02-07) - E/I Balance Control
- Przejście z kontroli ρ na kontrolę inh_scaling
- Usunięto globalne normalizacje spektralne w pętli

### v1.0 (2026-02-06) - Initial Implementation
- Podstawowa kalibracja z brentq
- Benchmarki: MC, NARMA, XOR
