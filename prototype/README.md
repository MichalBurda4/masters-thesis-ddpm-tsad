# Prototyp magisterki: DDPM + anomalie w szeregach czasowych

Minimalny pipeline do tematu *„Wykrywanie anomalii w szeregach czasowych przy użyciu DDPM z elementami wyjaśnialności”*.

## Co robi prototyp

1. **Generuje** syntetyczne szeregi (sinus + trend + szum) i wstrzykuje anomalie (skoki, tłumienie, plateau).
2. **Trenuje** DDPM **tylko na normalnych** danych — model uczy się „zdrowego” sygnału.
3. **Wykrywa anomalie** — częściowo zaszumia sygnał, odszumia go; duży błąd rekonstrukcji = anomalia.
4. **Wyjaśnialność (prosta)** — mapa błędu `|x - x_recon|` w każdym punkcie czasu (heatmapa na wykresie).

## Szybki start

```bash
cd prototype
pip install -r requirements.txt
python run.py
```

Albo krok po kroku:

```bash
python train.py --epochs 30
python detect.py
python visualize.py
```

Wynik: `checkpoints/ddpm_prototype.pt` oraz `outputs/example.png`.

## Struktura

| Plik | Rola |
|------|------|
| `data.py` | Dane syntetyczne, dataset treningowy |
| `model.py` | U-Net 1D (predykcja szumu) |
| `ddpm.py` | Harmonogram β, szumienie, odszumianie |
| `train.py` | Trening |
| `detect.py` | Scoring + proste metryki |
| `visualize.py` | Wykres sygnał / rekonstrukcja / błąd vs maska |
| `run.py` | Wszystko naraz (15 epok) |

## Kolejne kroki (pod pracę magisterską)

- [ ] Publiczny benchmark (SMD, Yahoo, UCR)
- [ ] Baseline: LSTM-AE, Isolation Forest
- [ ] Porównanie kroku dyfuzji `start_t`
- [ ] Lepsze XAI: SHAP / Integrated Gradients na wejściu
- [ ] Multivariate (wiele kanałów czujników)

## Uwaga

To **prototyp edukacyjny**, nie SOTA. Threshold w `detect.py` to mediana score — w pracy zastąpisz to walidacją na zbiorze dev.
