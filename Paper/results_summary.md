# Results summary

Comparison of raw RandomForest and calibrated classifier.

## Metrics table

|model|accuracy|precision_macro|recall_macro|f1_macro|brier|
|-|-|-|-|-|-|
|raw|1.0000|1.0000|1.0000|1.0000|0.022355|
|orig|1.0000|1.0000|1.0000|1.0000|0.022355|
|calibrated|1.0000|1.0000|1.0000|1.0000|0.003215|


## Calibration plots

### Class 0
![](figures/calibration_compare_class_0.png)

### Class 1
![](figures/calibration_compare_class_1.png)

### Class 2
![](figures/calibration_compare_class_2.png)
