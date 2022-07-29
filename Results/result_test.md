# Model tuning best results
Results file for best modelling results for trial `result_test`.'

---

The tuning parameter grid:

```json
{
  "learning_rate": [0.1,0.2],
  "max_depth": [3,6,9],
  "min_child_weight": [1,3,5],
  "colsample_bytree": [0.75],
  "gamma": [0]
}
```

The best_params_ grid from function is 
```json
{
  "colsample_bytree": 0.75,
  "gamma": 0,
  "learning_rate": 0.1,
  "max_depth": 9,
  "min_child_weight": 1
}
```

The max mean train score is -0.0013171555215647265.

The max mean test score is -0.02879445430072753.

The model predicted neg log loss on TEST is -0.025029141546079815.

---

The reliabilty diagram:
![reliability graphic](./result_test.jpg)

---

Top 20 features by importance:

```python
              feature name  importance
62            D19_SOZIALES       0.146
52       D19_KONSUMTYP_MAX       0.048
140           KBA05_KRSZUL       0.037
63             D19_TECHNIK       0.035
36             D19_BUCH_CD       0.034
106             HEALTH_TYP       0.032
55               D19_LOTTO       0.030
33     D19_BEKLEIDUNG_REST       0.023
101     GEBAEUDETYP_RASTER       0.022
98               FINANZTYP       0.021
105       GREEN_AVANTGARDE       0.019
16               CJT_TYP_1       0.016
51           D19_KONSUMTYP       0.016
137         KBA05_KRSKLEIN       0.016
50       D19_KINDERARTIKEL       0.015
43       D19_GESAMT_ANZ_24       0.014
132        KBA05_HERSTTEMP       0.014
58              D19_REISEN       0.013
96   FINANZ_UNAUFFAELLIGER       0.013
138          KBA05_KRSOBER       0.012
```

