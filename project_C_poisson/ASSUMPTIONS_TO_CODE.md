# Założenia Teoretyczne a Implementacja (Project C)

## GŁÓWNA TEZA BADAWCZA (Core Thesis)
**Jakość działania sieci neuronowej (XOR, NARMA, MC) jest najwyższa, gdy dynamika znajduje się na granicy chaosu i stabilności ($\lambda \approx 0$). Wraz z oddalaniem się od tej granicy (zarówno w stronę głębokiej stabilności $\lambda \ll 0$, jak i silnego chaosu $\lambda \gg 0$), jakość ta spada.**

---

## 1. Koncepcja Badania

Badanie ma na celu empiryczne potwierdzenie powyższej tezy poprzez:
1.  Zidentyfikowanie parametrów sterujących, które przesuwają sieć przez reżimy: Stabilny $\to$ Krawędź $\to$ Chaotyczny.
2.  Zmierzenie wydajności (Performance) dla zadań:
    *   **XOR** (Nieliniowa integracja)
    *   **NARMA** (Pamięć + Nieliniowość)
    *   **MC** (Pojemność pamięci)
3.  Wykazanie korelacji między odległością od krawędzi ($|\lambda|$) a spadkiem wydajności.

### Hipoteza 1: Performance Peak at the Edge

**Stwierdzenie:** Istnieje optymalny punkt pracy systemu (Critical Point), w którym $\lambda \approx 0$. W tym punkcie:
*   Maksymalizowana jest pamięć (MC).
*   Maksymalizowana jest zdolność rozwiązywania zadań nieliniowych (XOR, NARMA).

**Weryfikacja:**
*   Wykres $Performance(\lambda)$ powinien mieć kształt odwróconego "U" (lub "Góry") z wierzchołkiem przy $\lambda \approx 0$.
*   Punkty oddalone od zera (np. $\lambda = -0.05$ lub $\lambda = +0.05$) powinny mieć istotnie niższe wyniki.

---

## LOGICZNY PROTOKÓŁ BADAWCZY (Staged Approach)

Zgodnie z klasyczną literaturą (Bertschinger & Natschläger, 2004; Legenstein & Maass, 2007), badanie jest podzielone na trzy etapy:

### Stage A: Budowa Ensemble "Trójek" (Targeted Triplets)
**Cel:** Stworzenie 1000 zestawów sieci (tripletów), gdzie każda trójka współdzieli architekturę, ale różni się dynamiką: Stable ↔ Edge ↔ Chaotic.
*   **Stage A.1: Critical Points** – Znalezienie 1000 bazowych konfiguracji na krawędzi chaosu ($\lambda \approx 0$).
*   **Stage A.2: Bifurcation Tracking** – Znalezienie sąsiadów poprzez systematyczne "pchanie" parametrów ($\rho, g_L, g_A$) w stronę stabilności i chaosu, używając szukania binarnego.
*   **Metrologia**: Pomiar metryk wewnętrznych (Intrinsic Metrics) takich jak **Kernel Rank (KR)** oraz **Memory Capacity (MC)** bez zadania, aby wyjaśnić fizyczną przyczynę wydajności.

### Stage B: Ocena Funkcjonalna (Functional Evaluation)
**Cel:** Pomiar wydajności zadań (XOR, NARMA, MC) na zidentyfikowanych tripletach.
*   **Analiza:** Porównanie wydajności wewnątrz trójek (testy sparowane) oraz analiza trendów w zależności od odległości $d_{4D}$ od krawędzi.

### Stage C: Próba Syntezy (Validation of Thesis)
**Cel:** Wykazanie korelacji między "Stage A" a "Stage B".
*   **Weryfikacja:** Udowodnienie, że peak wydajności ze Stage B pokrywa się topologicznie z granicą stabilności ze Stage A.

---

## 2. BIOLOGICZNY REALIZM MODELU

---

### Założenie 1.4: Biologicznie wiarygodne częstotliwości igłowania

**Teoria:** Neurony korowe strzelają z częstotliwością 1-50 Hz w stanie czuwania.

**Implementacja:**
```python
# hh_model.py, wynik simulate()
'mean_rate': (np.sum(trimmed_spikes) / (valid_steps * dt * 1e-3)) / N
```

**Weryfikacja:**
- Filtr: `1 < firing_rate < 50 Hz` → trial biologicznie wiarygodny
- Trial poza tym zakresem → odrzucony z analizy

---

### Założenie 1.5: Nieregularne igłowanie (CV ISI ≈ 1.0)

**Teoria:** Neurony korowe wykazują nieregularne igłowanie (Coefficient of Variation interspike intervals ≈ 1.0), NIE regularne jak zegar.

**Implementacja:**
```python
# hh_model.py, bio-metrics section
for neuron_isis in isi_list:
    if len(neuron_isis) > 1:
        cv = np.std(neuron_isis) / (np.mean(neuron_isis) + 1e-9)
        cv_values.append(cv)
mean_cv = np.mean(cv_values)
```

**Weryfikacja:**
- CV ∈ [0.5, 1.5] → biologicznie wiarygodne
- CV < 0.5 → zbyt regularne (patologiczne)
- CV > 1.5 → zbyt chaotyczne (burst-like)

---

## 2. KRAWĘDŹ CHAOSU

### Założenie 2.1: Wykładnik Lyapunowa jako miara chaosu

**Teoria:** 
- λ < 0 → dynamika stabilna (ordered)
- λ = 0 → krawędź chaosu (critical)
- λ > 0 → dynamika chaotyczna

**Implementacja:**
```python
# tasks/lyapunov_task.py + run_experiment.py
# 1. Rozgrzewka do atraktora
res_wu = hh.simulate(..., trim_steps=0)
start_state = res_wu['final_state']

# 2. Trajektoria referencyjna
state_l1 = hh.simulate(..., full_state=start_state)

# 3. Trajektoria zaburzona (ε = 1e-6 mV)
perturbed_state['V'][0] += cfg.task.lyap_eps
state_l2 = hh.simulate(..., full_state=perturbed_state)

# 4. Obliczenie dywergencji
slope = lyap.compute_lambda(phi1, phi2, window_range=[50, 250])
lambda_sec = slope / step_duration_s  # Jednostka: s⁻¹
```

**Weryfikacja:**
- λ_sec ∈ [-0.01, +0.01] → krawędź chaosu ✓
- Szukamy peak performance w tym zakresie

---

### Założenie 2.2: Spektralny promień (ρ) kontroluje dynamikę

**Teoria:** 
- ρ < 1 → zanikające echa (słaba pamięć)
- ρ ≈ 1 → długie echa (dobra pamięć)
- ρ > 1 → eksplodujące echa (chaos)

**Implementacja:**
```python
# hh_model.py
radius = np.max(np.abs(np.linalg.eigvals(W)))
if radius > 1e-10: W /= radius  # Normalizacja do ρ=1

# run_experiment.py
W_eff = self.W_rec * rho  # Skalowanie do zadanego ρ
```

**Weryfikacja:**
- Sweep ρ ∈ [0.1, 50.0] → znajdź gdzie λ przechodzi przez 0
- W HH z prądem A spodziewamy się, że potrzeba WYŻSZEGO ρ niż w ESN

---

### Założenie 2.3: Konduktancja wycieku (gL) wpływa na stabilność

**Teoria:** 
- Wysoki gL → szybszy powrót do potencjału spoczynkowego → stabilniejszy
- Niski gL → wolniejsza dynamika membranowa → łatwiej wejść w chaos

**Implementacja:**
```python
# hh_model.py, simulate()
gL_eff = gL if gL is not None else p.gL  # Override dla Phase 1

# Równanie dV:
dV = (... - gL_eff*(V-p.EL) ...) / p.C
```

**Weryfikacja:**
- Sweep gL ∈ [0.3, 0.15, 0.075] (100%, 50%, 25% baseline)
- Niższy gL → λ powinno rosnąć przy tym samym ρ

---

## 3. WYDAJNOŚĆ OBLICZENIOWA

### Założenie 3.1: XOR jako test nieliniowej integracji

**Teoria:** Zadanie XOR wymaga SYNERGII - odpowiedź zależy od kombinacji dwóch wejść, nie od każdego z osobna. Tylko sieci z nieliniową dynamiką mogą je rozwiązać.

**Parametry:**
- `delay = 5` symboli = 5 × 20ms = **100ms pamięci**
- Trudniejsze niż typowe delay=2 (wymaga dłuższej pamięci)

**Implementacja:**
```python
# tasks/xor.py
y[t] = u[t] XOR u[t-5]  # delay = 5 symboli

# Readout: Ridge Regression na stanach filtrowanych
met_xor = readout.train_ridge_cv(phi_xor, y_xor, task_type='classification')
```

**Weryfikacja:**
- Accuracy > 0.6 → sieć funkcjonalna
- Accuracy ≈ 0.5 → losowe zgadywanie (brak funkcjonalności)
- **Peak przy λ ≈ 0** → potwierdza hipotezę krawędzi chaosu

---

### Założenie 3.2: NARMA jako test pamięci i nieliniowości

**Teoria:** NARMA-10 wymaga zarówno pamięci (10 kroków wstecz) jak i nieliniowej transformacji.

**Implementacja:**
```python
# tasks/narma.py
y[t] = 0.3*y[t-1] + 0.05*y[t-1]*Σy[t-10:t] + 1.5*u[t-1]*u[t-10] + 0.1
```

**Weryfikacja:**
- NRMSE < baseline AR(10) → sieć lepsza niż prosty model liniowy
- **Minimum NRMSE przy λ ≈ 0** → potwierdza hipotezę

---

### Założenie 3.3: Memory Capacity jako miara pojemności pamięci

**Teoria:** MC mierzy ile bitów informacji sieć może zapamiętać z przeszłości.

**Implementacja:**
```python
# tasks/mc.py
# Dla każdego lag k: regresja phi(t) → u(t-k)
# MC = Σ r²(k) dla k = 1..max_lag
```

**Weryfikacja:**
- MC > baseline (shuffled) → sieć ma pamięć
- **Peak MC przy λ ≈ 0** → potwierdza hipotezę

---

## 4. READOUT (WARSTWA WYJŚCIOWA)

### Założenie 4.1: Ridge Regression jako standard

**Teoria:** W Reservoir Computing tylko warstwa wyjściowa jest trenowana. Ridge zapobiega overfitting w wysokowymiarowej przestrzeni stanów.

**Implementacja:**
```python
# readout.py
from sklearn.linear_model import RidgeCV
alphas = [1e-6, 1e-4, 1e-2, 1, 10, 100]
model = RidgeCV(alphas=alphas, cv=5)
```

**Weryfikacja:**
- Porównanie z baseline (AR, random classifier)
- Improvement = (baseline - model) / baseline

---

### Założenie 4.2: Walidacja krzyżowa z przerwą czasową

**Teoria:** Dane czasowe wymagają CV z przerwą, żeby zapobiec wyciekowi informacji z przyszłości.

**Implementacja:**
```python
# cv.py
# BlockedTimeSeriesSplit z gap=10 kroków
```

**Weryfikacja:**
- Train/Test gap zapobiega leakage
- Wyniki uogólniają na niewidziane dane

---

## 5. PROTOKÓŁ EKSPERYMENTALNY

### Faza 1: Generowanie Ensemble (Triplets)

**Cel:** Stworzenie 1000 tripletów z kompletem metryk.

**Protokół:**
1.  **Stage A.1:** Random search dla 1000 punktów $\lambda \in [-0.05, 0.05]$.
2.  **Stage A.2:** Bifurcation Tracking dla sąsiadów.
3.  **Metrics:** Kernel Rank, Intrinsic MC, Distances ($d_{\lambda}, d_{4D}$).

---

### Faza 2: Walidacja Hipotezy (Stage B)

**Cel:** Potwierdzić, że performance jest MAKSYMALNE przy krawędzi chaosu i skorelowane z metrykami wewnętrznymi.

**Analiza:**
1.  **Testy sparowane:** Wilcoxon wewnątrz trójek (Edge vs Stable, Edge vs Chaotic).
2.  **Korelacje:** Pearson/Spearman między (Kernel Rank, MC) a (XOR, NARMA).
3.  **Analiza odległości:** Spadek wydajności jako funkcja $d_{4D}$.

---

## PODSUMOWANIE: ŁAŃCUCH LOGICZNY

```
1. BIOLOGIA
   ├── HH z gNa, gK, gL, gA → realistyczna dynamika jonowa
   ├── Prąd A → adaptacja spike-frequency
   ├── Dale's principle → E/I balance
   └── Firing rate 1-50 Hz, CV ISI ≈ 1.0 → biologiczne markery

2. DYNAMIKA
   ├── ρ (spectral radius) → kontrola siły recurrence
   ├── gL (leak) → kontrola stabilności membranowej
   └── λ (Lyapunov) → MIARA krawędzi chaosu

3. WYDAJNOŚĆ
   ├── XOR → nieliniowa integracja
   ├── NARMA → pamięć + nieliniowość
   └── MC → pojemność pamięci

4. HIPOTEZA
   └── Peak(XOR, NARMA, MC) występuje przy λ ≈ 0

5. WERYFIKACJA
   ├── Faza 1: Znajdź λ ≈ 0
   └── Faza 2: Potwierdź peak statystycznie
```

---

## MAPOWANIE PARAMETRÓW: TEORIA → YAML → KOD

| Założenie | Parametr YAML | Zmienna w kodzie | Gdzie sprawdzić |
|-----------|---------------|------------------|-----------------|
| Konduktancja Na | `hh.gNa` | `p.gNa` | `hh_model.py:dV` |
| Konduktancja K | `hh.gK` | `p.gK` | `hh_model.py:dV` |
| Konduktancja wycieku | `gL_grid` | `gL_eff` | `hh_model.py:dV` |
| Prąd A | `hh.gA` | `p.gA` | `hh_model.py:dV` |
| Spektralny promień | `rho_grid_phase1` | `rho` | `run_experiment.py` |
| Rozmiar sieci | `N_grid` | `N` | `run_experiment.py` |
| Dale's principle | `hh.conn_type: dale` | `ei_vec` | `hh_model.py` |
| Firing rate filter | wynik | `mean_rate` | output parquet |
| CV ISI | wynik | `cv_isi` | output parquet |
| E/I balance | wynik | `ei_balance` | output parquet |
| Lyapunov λ | wynik | `lambda_sec` | output parquet |

---

## OUTPUT SCHEMA: Co Zapisywać

### CORE (zawsze zapisywane - niezbędne do analizy)
| Kolumna | Typ | Opis |
|---------|-----|------|
| `N` | int | Rozmiar sieci |
| `gL` | float | Konduktancja wycieku |
| `rho` | float | Spektralny promień |
| `bias` | float | Prąd zewnętrzny |
| `difficulty` | str | Poziom trudności (easy/medium/hard) |
| `task` | str | Nazwa zadania (NARMA/XOR/MC/Lyapunov) |
| `metric` | str | Metryka (nrmse/accuracy/capacity/lambda_sec) |
| `value` | float | Wartość metryki |

### BIO (bio-plausibility - do filtrowania i walidacji)
| Kolumna | Typ | Cel |
|---------|-----|-----|
| `firing_rate` | float | Filtr: 1-50 Hz = biologiczne |
| `cv_isi` | float | Filtr: 0.5-1.5 = nieregularne |
| `ei_balance` | float | Walidacja: ~1.0 = zbalansowane |
| `mean_voltage` | float | Walidacja: ~-65 mV = spoczynkowe |
| `saturation_flag` | bool | Filtr: False = brak saturacji |

### NIE ZAPISYWANE (oszczędność dysku)
| Kolumna | Powód pominięcia |
|---------|------------------|
| `seed_rec, seed_inmask...` | Reprodukowalność przez `seed_tuple_id` |
| `I_syn_mean, I_syn_var` | Diagnostyka, nie do analizy głównej |
| `baseline, improvement` | Można obliczyć post-hoc |
| `timestamp, git_hash` | Raz na plik, nie per-row |

### Szacowane zużycie dysku

**Phase 1 (full):**
- 3 N × 3 gL × 8 rho × 4 bias × 3 difficulty × 3 seeds = **2592 trials**
- ~6 rows per trial (4 tasks, 2 metrics lyapunov) = **15552 rows**
- ~10 kolumn × 8 bytes = **~1.2 MB** (parquet compressed)

**Akceptowalne** ✓

---

## WYKRESY I ANALIZY DO PUBLIKACJI

### 1. GŁÓWNY WYKRES: Performance vs Lyapunov (λ)

**Cel:** Potwierdzić peak performance przy λ ≈ 0

**Dane wymagane:**
| Kolumna | Użycie |
|---------|--------|
| `lambda_sec` | Oś X (binning: [-0.1, -0.05, -0.01, 0, +0.01, +0.05, +0.1]) |
| `value` (XOR accuracy) | Oś Y |
| `value` (NARMA nrmse) | Oś Y (osobny panel) |
| `value` (MC capacity) | Oś Y (osobny panel) |
| `difficulty` | Osobne krzywe/panele |

**Typ wykresu:** 
- Line plot z error bars (mean ± std per bin)
- Lub violin plot per bin

**Kod analizy:**
```python
df_xor = df[(df['task'] == 'XOR') & (df['metric'] == 'accuracy')]
df_lyap = df[(df['task'] == 'Lyapunov') & (df['metric'] == 'lambda_sec')]

# Merge by (N, gL, rho, bias, difficulty, seed_tuple_id)
merged = df_xor.merge(df_lyap, on=['N', 'gL', 'rho', 'bias', 'difficulty', 'seed_tuple_id'], 
                      suffixes=('_xor', '_lyap'))

# Bin by lambda
bins = [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf]
labels = ['Subcrit Deep', 'Subcrit Near', 'Critical', 'Supercrit Near', 'Chaotic']
merged['regime'] = pd.cut(merged['value_lyap'], bins, labels=labels)

# Plot
sns.boxplot(data=merged, x='regime', y='value_xor', hue='difficulty')
```

---

### 2. HEATMAP: Performance w przestrzeni (ρ, gL)

**Cel:** Znaleźć optymalny region parametrów

**Dane wymagane:**
| Kolumna | Użycie |
|---------|--------|
| `rho` | Oś X |
| `gL` | Oś Y |
| `value` (XOR) | Kolor (mean per cell) |
| `N` | Osobne panele |

**Typ wykresu:** Heatmap (imshow/pcolormesh)

**Kod analizy:**
```python
pivot = df_xor.groupby(['rho', 'gL'])['value'].mean().unstack()
sns.heatmap(pivot, cmap='viridis', annot=True)
plt.xlabel('ρ (spectral radius)')
plt.ylabel('gL (leak conductance)')
```

---

### 3. SCALING: Performance vs N (rozmiar sieci)

**Cel:** Pokazać jak performance skaluje się z rozmiarem sieci

**Dane wymagane:**
| Kolumna | Użycie |
|---------|--------|
| `N` | Oś X (100, 300, 1000) |
| `value` | Oś Y |
| `difficulty` | Osobne krzywe |
| `task` | Osobne panele |

**Oczekiwany wynik:** 
- Większe N → wyższa accuracy dla trudniejszych zadań
- Ale przy tym samym ρ może wejść w inny regime (λ się zmieni)

---

### 4. BIO-VALIDATION: Firing Rate vs λ

**Cel:** Pokazać, że biologiczny regime (1-50 Hz) pokrywa się z Edge of Chaos

**Dane wymagane:**
| Kolumna | Użycie |
|---------|--------|
| `firing_rate` | Oś Y |
| `lambda_sec` | Oś X |
| `saturation_flag` | Kolor/marker (saturated = czerwony) |

**Oczekiwany wynik:**
- Punky przy λ ≈ 0 mają firing_rate ∈ [1, 50] Hz
- Chaos (λ > 0.05) → saturation lub firing > 50 Hz

---

### 5. BIO-VALIDATION: CV ISI Distribution

**Cel:** Potwierdzić nieprawialność igłowania (CV ≈ 1.0)

**Dane wymagane:**
| Kolumna | Użycie |
|---------|--------|
| `cv_isi` | Histogram |
| `regime` (binned λ) | Osobne histogramy |

**Oczekiwany wynik:**
- CV ∈ [0.5, 1.5] dla większości punktów
- Wartości poza tym zakresem → niefizjologiczne

---

### 6. DIFFICULTY SCALING

**Cel:** Pokazać, że Edge of Chaos jest szczególnie korzystny dla TRUDNYCH zadań

**Dane wymagane:**
| Kolumna | Użycie |
|---------|--------|
| `difficulty` | Grupowanie (easy/medium/hard) |
| `regime` | Oś X |
| `value` | Oś Y |

**Hipoteza:** 
- Easy: performance podobna w subcrit i critical
- Hard: performance ZNACZĄCO wyższa w critical

---

### 7. TESTY STATYSTYCZNE

**Test 1: Critical vs Neighbors (główna hipoteza)**
```python
from scipy.stats import wilcoxon

critical = merged[merged['regime'] == 'Critical']['value_xor']
subcrit = merged[merged['regime'] == 'Subcrit Near']['value_xor']
supercrit = merged[merged['regime'] == 'Supercrit Near']['value_xor']

stat, p_sub = wilcoxon(critical, subcrit)
stat, p_sup = wilcoxon(critical, supercrit)

# Bonferroni correction
alpha_corrected = 0.05 / 2
significant = (p_sub < alpha_corrected) and (p_sup < alpha_corrected)
```

**Test 2: Effect Size (Cohen's d)**
```python
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)

d_sub = cohens_d(critical, subcrit)
d_sup = cohens_d(critical, supercrit)

# Interpretacja: |d| > 0.8 = large effect
```

**Test 3: Difficulty × Regime Interaction (ANOVA)**
```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('value_xor ~ C(regime) * C(difficulty)', data=merged).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
# Szukamy: significant interaction effect
```

---

## PODSUMOWANIE: MINIMALNE DANE DO PUBLIKACJI

| Kolumna | Wymagana? | Używana w |
|---------|-----------|-----------|
| `N` | ✅ | Heatmap, Scaling |
| `gL` | ✅ | Heatmap |
| `rho` | ✅ | Heatmap |
| `bias` | ✅ | (grupowanie) |
| `difficulty` | ✅ | Difficulty Scaling, ANOVA |
| `task` | ✅ | Filtrowanie |
| `metric` | ✅ | Filtrowanie |
| `value` | ✅ | **WSZYSTKIE WYKRESY** |
| `firing_rate` | ✅ | Bio-validation |
| `cv_isi` | ✅ | Bio-validation |
| `ei_balance` | ⚪ | Opcjonalne (można pominąć) |
| `mean_voltage` | ⚪ | Opcjonalne |
| `saturation_flag` | ✅ | Bio-validation, filtrowanie |
| `seed_tuple_id` | ✅ | Merge między taskami |

**KLUCZOWE:** `seed_tuple_id` jest WYMAGANE do łączenia wyników XOR z Lyapunov dla tego samego trialu!
