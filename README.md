# HH+STP Liquid State Machine — Regime Calibration Pipeline

## Cel projektu

Zbadanie jak **reżim dynamiczny** (od super-stabilnego po chaotyczny) wpływa na
zdolność obliczeniową rezerwuaru opartego na biologicznie realistycznych neuronach
Hodgkina-Huxleya z synapsami STP (Short-Term Plasticity).

Hipoteza: reservoir computing działa najlepiej na **krawędzi chaosu** ("edge of chaos"),
gdzie układ wykazuje wystarczającą wrażliwość na sygnał wejściowy, ale nie traci
informacji przez niestabilność dynamiczną.

## Po co to robimy

1. **Weryfikacja hipotezy edge-of-chaos** — Maass, Natschlaeger & Markram (2002)
   pokazali, że LSM osiąga najlepszą separowalność stanów w pobliżu krytyczności.
   Testujemy to na modelu HH+STP (bardziej biologicznie realistycznym niż typowe LIF).

2. **Kontrolowane porównanie reżimów** — dzięki temu, że topologia i wagi bazowe
   są zamrożone (ETAP A), a jedynie skalujemy je współczynnikiem `alpha`, uzyskujemy
   5 reżimów dynamicznych NA TYM SAMYM rezerwuarze. To eliminuje zmienność
   topologiczną i izoluje wpływ dynamiki.

3. **Rzetelna kalibracja** — zamiast ad-hoc parametrów, używamy prawdziwego
   wykładnika Lyapunova (metoda Benettina) do ilościowej klasyfikacji reżimów.
   Bisection z guardrailami (firing rate, CV ISI, synchronizacja) gwarantuje,
   że każdy reżim jest fizjologicznie sensowny.

## Pipeline — 3 etapy

### ETAP A: `regime_builder.py` — Budowa rezerwuaru (bez Brian2)

Generuje JEDNĄ zamrożoną topologię E/I (Erdos-Renyi) z wagami i parametrami STP,
a następnie znajduje 5 reżimów operacyjnych przez skalowanie `alpha`:

| Reżim | Nazwa | Proxy λ (log ρ) |
|-------|-------|-----------------|
| R1 | super_stable | najniższy |
| R2 | stable | niski |
| R3 | near_critical | umiarkowany |
| R4 | edge_of_chaos | wysoki |
| R5 | chaotic | najwyższy |

**Wyjście:** 5 plików `.npz` z pełną łącznością + parametry STP + metadane.

### ETAP B: `regime_calibrator.py` — Kalibracja dynamiczna (Brian2 HH+STP)

Ładuje `.npz` z ETAP A, buduje sieć Brian2 w standardowych warunkach operacyjnych
(prąd tła I_b + niezależne wejście Poissona 20 Hz na neuron) i dostraja `alpha`
za pomocą bracketed bisection, aż prawdziwy wykładnik Lyapunova (λ) każdego reżimu
mieści się w wyznaczonym oknie:

| Reżim | Okno λ [1/s] | Target λ |
|-------|-------------|----------|
| R1 super_stable | [0, 8) | 5.0 |
| R2 stable | [8, 18) | 12.0 |
| R3 near_critical | [18, 30) | 22.0 |
| R4 edge_of_chaos | [30, 45) | 35.0 |
| R5 chaotic | [45, 80) | 55.0 |

Guardrails: firing rate E/I, CV ISI, sync index, spontaneous activity check.

**Metoda Lyapunova (Benettin):**
- Dwie trajektorie z małą perturbacją stanów (v, m, h, n)
- Deterministyczny replay wejścia Poissona via `SpikeGeneratorGroup`
- Periodyczna renormalizacja co 10 ms, mediana z K powtórzeń

**Wyjście:** 5 skalibrowanych `.npz` + CSV z metrykami.

**Wyniki kalibracji v15:**
```
R2  alpha=0.003669  λ=15.30  [OK]
R3  alpha=0.011695  λ=27.37  [OK]
R4  alpha=0.009724  λ=34.08  [OK]
R5  alpha=0.073494  λ=47.30  [OK]
Monotonic: 15.30 < 27.37 < 34.08 < 47.30  PASS
```

### ETAP C: Benchmark (TODO)

Testy separowalności (Schmitt/Maass) i pojemności pamięci na skalibrowanych
rezerwuarach, porównanie wydajności RC między reżimami.

## Pliki projektu

| Plik | Opis |
|------|------|
| `regime_builder.py` | ETAP A — budowa rezerwuaru, 5 reżimów via alpha-scaling |
| `regime_calibrator.py` | ETAP B — kalibracja Lyapunova, bisection, Brian2 |
| `lambda_scan.py` | Diagnostyka — skan λ(alpha) + test zbieżności dt |
| `reservoir_hh_stp.py` | Referencyjny benchmark HH+STP (separation test) |
| `reservoir_lif_stp.py` | Referencyjny benchmark LIF+STP (szybszy, mniej realistyczny) |
| `liquid_state_machine.py` | Baza LSM (Schmitt 2022) |
| `narma10.py` | Benchmark NARMA-10 |
| `hh_sweep.py` | Sweep parametrów HH (alpha, I_b) |
| `_concat.ps1` | Helper do konkatenacji kodu pipeline |

## Uruchomienie

```bash
# ETAP A — budowa rezerwuaru (lokalne, bez Brian2)
python regime_builder.py

# ETAP B — kalibracja (wymaga Brian2, najlepiej na serwerze)
python regime_calibrator.py                           # wszystkie reżimy sekwencyjnie
python regime_calibrator.py --regime R3_near_critical  # pojedynczy reżim (do równoległego)
```

## Wymagania

- Python 3.9+
- NumPy, SciPy
- Brian2 (ETAP B)
- scikit-learn (benchmarki RC)

## Literatura

- Maass, Natschlaeger, Markram (2002) — "Real-time computing without stable states"
- Schmitt (2022) — konstrukcja rezerwuaru LSM
- Benettin et al. (1980) — metoda obliczania wykładników Lyapunova
