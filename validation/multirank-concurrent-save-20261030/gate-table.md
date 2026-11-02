# Concurrent save gates

Δ = median of five paired Meta − CPU samples; percent denominator = median CPU. Wall includes fsync; shape removes the global union of measured fsync intervals. strict=±3%; floor=max(3%×CPU median,100 ms).

|Cell|Metric|Meta s|CPU s|Paired Δ ms|Paired Δ %|Strict|Floor|
|---|---|---:|---:|---:|---:|---|---|
|1b-4r-dcp|wall|13.1371|13.1005|75.84|0.58|pass|pass|
|1b-4r-dcp|P10|1.1669|1.1573|13.36|1.15|pass|pass|
|1b-4r-dcp|P50|1.6593|1.6484|10.83|0.66|pass|pass|
|1b-4r-dcp|P90|2.1779|2.1646|10.38|0.48|pass|pass|
|1b-4r-dcp|P100|2.3000|2.3000|0.00|0.00|pass|pass|
|1b-4r-safetensors|wall|12.4218|12.4995|-70.38|-0.56|pass|pass|
|1b-4r-safetensors|P10|0.6273|0.6262|2.13|0.34|pass|pass|
|1b-4r-safetensors|P50|0.9633|0.9710|-7.71|-0.79|pass|pass|
|1b-4r-safetensors|P90|1.3351|1.3381|-3.05|-0.23|pass|pass|
|1b-4r-safetensors|P100|1.4000|1.4000|0.00|0.00|pass|pass|
|1b-8r-dcp|wall|13.2918|13.2939|82.07|0.62|pass|pass|
|1b-8r-dcp|P10|1.8591|1.8594|0.13|0.01|pass|pass|
|1b-8r-dcp|P50|2.3047|2.3040|5.97|0.26|pass|pass|
|1b-8r-dcp|P90|2.4864|2.4379|48.48|1.99|pass|pass|
|1b-8r-dcp|P100|2.5000|2.4500|50.00|2.04|pass|pass|
|1b-8r-safetensors|wall|12.6423|12.4819|227.91|1.83|pass|pass|
|1b-8r-safetensors|P10|1.0583|1.0636|-10.30|-0.97|pass|pass|
|1b-8r-safetensors|P50|1.3767|1.3642|2.62|0.19|pass|pass|
|1b-8r-safetensors|P90|1.5803|1.5342|46.24|3.01|fail|pass|
|1b-8r-safetensors|P100|1.6000|1.5500|50.00|3.23|fail|pass|
|3b-4r-dcp|wall|40.1475|40.2246|-122.28|-0.30|pass|pass|
|3b-4r-dcp|P10|3.0411|3.0721|-71.67|-2.33|pass|pass|
|3b-4r-dcp|P50|4.5984|4.6487|-96.03|-2.07|pass|pass|
|3b-4r-dcp|P90|16.0347|16.1962|-106.61|-0.66|pass|pass|
|3b-4r-dcp|P100|18.2500|18.5500|-150.00|-0.81|pass|pass|
|3b-4r-safetensors|wall|38.6488|38.6774|-2.77|-0.01|pass|pass|
|3b-4r-safetensors|P10|1.8005|1.8342|-34.32|-1.87|pass|pass|
|3b-4r-safetensors|P50|2.9532|2.9493|18.92|0.64|pass|pass|
|3b-4r-safetensors|P90|14.3824|14.4219|-48.19|-0.33|pass|pass|
|3b-4r-safetensors|P100|16.7500|16.6000|0.00|0.00|pass|pass|
|3b-8r-dcp|wall|42.3802|42.3887|-21.97|-0.05|pass|pass|
|3b-8r-dcp|P10|4.9018|4.8889|32.55|0.67|pass|pass|
|3b-8r-dcp|P50|6.2671|6.3461|-93.70|-1.48|pass|pass|
|3b-8r-dcp|P90|11.8800|11.2824|1195.94|10.60|fail|fail|
|3b-8r-dcp|P100|12.3000|11.3000|1200.00|10.62|fail|fail|
|3b-8r-safetensors|wall|40.0816|40.1317|-42.91|-0.11|pass|pass|
|3b-8r-safetensors|P10|3.0360|3.0330|-28.24|-0.93|pass|pass|
|3b-8r-safetensors|P50|4.0738|4.0984|-24.55|-0.60|pass|pass|
|3b-8r-safetensors|P90|14.4612|12.6725|1098.96|8.67|fail|fail|
|3b-8r-safetensors|P100|14.5000|12.7000|1100.00|8.66|fail|fail|
