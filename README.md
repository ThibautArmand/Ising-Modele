# Modèle d’Ising, transition de phase, méthode de Monte Carlo

![CI Status](https://github.com/ThibautArmand/Ising-Modele/actions/workflows/ci.yml/badge.svg)

Une implémentation en Python du modèle d’Ising utilisant des méthodes de Monte Carlo pour le cours de Physique Numérique LU3PY124 2025-26.

## Installation

### Sur Spyder

1. Installer les dépendances requises :

```bash
pip install -r requirements.txt
```

2. Lancer la simulation :

```bash
python IsingBase.py
```

## Exécution des tests

```bash
python -m pytest tests/ -v
```

## Structure du projet

- `utils/` - Fonctions utilitaires pour les simulations du modèle d’Ising
- `tests/` - Suite de tests
- `Transition_Phase_tailles` - Des analyses d'énergie, aimantation, susceptibilité, et chaleur et son évolution pour différents températures.
- `Exposants_Critiques` - Des analyses à `T_c` pour différentes tailles
- `IsingBase.py` - Implémentation de base du modèle d’Ising

## Contributeurs

- ARGUELLO Camilo
- ARMAND Thibaut
