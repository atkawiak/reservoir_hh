# Projekt Badawczy: Granica Chaosu w Rezerwuarach HH

## 🎯 Główna Hipoteza
Biologiczne sieci neuronowe (Hodgkin-Huxley) maksymalizują swoją wydajność obliczeniową (pamięć i przetwarzanie nieliniowe) podczas pracy na **„Granicy Chaosu” (Edge of Chaos)** — krytycznym stanie przejścia, w którym wykładnik Lapunowa λ wynosi około 0.

---

## 🔬 Pytania Badawcze

### **RQ1: Lokalizacja Granicy Chaosu**
*Przy jakim promieniu spektralnym (ρ) w sieci rezerwuaru HH (N=100, 80/20 E/I) występuje stan krytyczny, definiowany jako pseudo-Lyapunov exponent λ ∈ [-0.05, +0.05], przy kodowaniu Poisson (40 Hz, tło 2 Hz) i stałym biasie wejściowym?*

**Operacyjna Definicja:**
- **Metoda pomiaru λ**: Pseudo-Lyapunov dla zmiennych ciągłych (V, n, m, h) z perturbacją ΔV = 0.1 mV
- **Zakres sweepingu**: ρ ∈ [0.1, 20.0], krok 0.1
- **Kryteria**: ρ_critical = wartość ρ, przy której |λ| jest minimalne (najbliżej zeru)

---

### **RQ2: Maksymalizacja Wydajności Obliczeniowej przy λ ≈ 0**
*Czy wydajność obliczeniowa (Memory Capacity, Delayed XOR, NARMA-10) osiąga maksimum w regionie krytycznym λ ∈ [-0.05, +0.05], w porównaniu do reżimów stabilnego (λ < -0.1) i chaotycznego (λ > +0.1)?*

**Testowane Hipotezy:**
- **H1**: MC jest maksymalne przy λ ≈ 0 (expected: 15-30 bitów)
- **H2**: Delayed XOR accuracy jest najwyższe przy λ ≈ 0 (expected: >85%)
- **H3**: NARMA-10 NRMSE jest minimalne przy λ ≈ 0 (expected: <0.4)

**Validacja Statystyczna:**
- Test Wilcoxon Signed-Rank (n=20 seeds) porównujący wydajność w 3 reżimach
- Krzywa wydajności vs λ z analizą korelacji Pearsona

---

### **RQ3: Wpływ Parametrów Biologicznych na Region Krytyczny**
*Jak parametry sieci HH (stosunek E/I, siła połączeń synaptycznych, parametry kanałów jonowych) wpływają na lokalizację i szerokość regionu krytycznego?*

**Analiza:**
- Identyfikacja zakresu ρ gdzie λ ∈ [-0.05, +0.05] (szerokość regionu krytycznego: Δρ)
- Wpływ zmienności parametrów biologicznych na stabilność ρ_critical (multi-seed)
- Podstawowa charakterystyka dynamiki w regionie krytycznym

---

### **RQ4: Branching Ratio i Propagacja Aktywności** *(opcjonalne)*
*Czy w stanie krytycznym (λ ≈ 0) sieć HH wykazuje branching ratio σ ≈ 1, zgodnie z teorią krytyczności w spiking networks? Jak σ zmienia się w funkcji ρ?*

**Metryka:**
- σ = średnia liczba spike'ów wywołanych przez pojedynczy spike (analiza spike-triggered average)
- Oczekiwanie: σ ≈ 1 przy ρ_critical, σ < 1 w stable, σ > 1 w chaotic

**Uwaga:** To pytanie jest **opcjonalne** - stanowi rozszerzenie analizy, ale nie jest krytyczne dla potwierdzenia głównej hipotezy.

---

## 🛠️ Zadania Benchmarkowe i Metryki

| Zadanie | Opis | Cel | Metryka |
| :--- | :--- | :--- | :--- |
| **Lapunow (λ)** | Dywergencja trajektorii po perturbacji 0.1mV. | Ilościowe określenie chaosu/stabilności. | λ (s⁻¹) |
| **NARMA-10** | Nieliniowa regresja szeregów czasowych 10. rzędu. | Złożona pamięć nieliniowa. | NRMSE |
| **Delayed XOR** | Operacja XOR na bitach z opóźnieniem $d \in \{1, 2, 3\}$. | Nieliniowa separowalność klas. | Accuracy |
| **Pojemność Pamięci (MC)** | Liniowa rekonstrukcja wejść do opóźnienia $k=60$. | Liniowa pamięć zanikająca. | Bity ($\sum R^2$) |

---

## 📈 Memory Capacity (MC) – Scenariusz i Oczekiwane Wyniki

### Scenariusz Testowy
*   **Sieć**: N=100 neuronów HH.
*   **Wejście**: Sygnał losowy Poisson (Rate Coding).
*   **Analiza**: Regresja Ridge dla opóźnień (lags) od 1 do 60 kroków.
*   **Walidacja**: Porównanie z „shuffled input” (baseline) oraz rezerwuarem ESN.

### Oczekiwane Wyniki
1.  **Stan Stabilny (λ < 0)**: MC jest niskie (~5-10 bitów). Pamięć szybko zanika, sieć jest zbyt tłumiona, by zachować historię sygnału.
2.  **Granica Chaosu (λ ≈ 0)**: **Szczyt MC**. Oczekujemy wartości w zakresie 15-30 bitów (15-30% rozmiaru sieci). Jest to punkt optymalny, gdzie informacja jest podtrzymywana przez dynamikę rekurencyjną bez utraty stabilności.
3.  **Stan Chaotyczny (λ > 0)**: Gwałtowny spadek MC. Choć sieć jest aktywna, „efekt motyla” (wrażliwość na warunki początkowe) niszczy korelację między stanem rezerwuaru a przeszłym sygnałem wejściowym.

---

## 🗺️ Mapa Drogowa Implementacji

1.  **Faza 1 (Search)**: Wykonanie gęstego sweepu $\rho \in [0.1, 20.0]$ dla λ. Wyznaczenie $\rho_{krytyczne}$.
2.  **Faza 2 (Benchmark)**: Mapowanie wydajności MC, XOR i NARMA na osi λ.
3.  **Faza 3 (Proof)**: Statystyczne potwierdzenie (Wilcoxon, n=20) przewagi stanu Edge of Chaos nad innymi reżimami.
