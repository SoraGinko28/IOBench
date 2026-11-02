# gdtensor trace analysis

Times are rank-local seconds. Save excludes the union of fsync spans; loads use the complete checkpoint call. Payload/header requests are compared separately. No cross-rank scheduling order is compared.

|Model / format / rank|Config|Phase|Total|Save excl. fsync|D2H|H2D|pwrite|Read|fsync|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|1b / dcp / 0|cpu r1|cold-load|11.4736|11.4736|0.0000|0.0000|0.0000|8.6563|0.0000|
|1b / dcp / 0|cpu r1|save|15.0809|4.1467|0.0000|0.0000|1.2816|0.0000|10.9342|
|1b / dcp / 0|cpu r1|warm-load|3.0244|3.0244|0.0000|0.0000|0.0000|0.6231|0.0000|
|1b / dcp / 0|gpu r1|cold-load|11.4295|11.4295|0.0000|0.5125|0.0000|7.3022|0.0000|
|1b / dcp / 0|gpu r1|save|16.6734|5.5446|1.4250|0.0000|1.1695|0.0000|11.1287|
|1b / dcp / 0|gpu r1|warm-load|4.4785|4.4785|0.0000|0.5106|0.0000|0.6747|0.0000|
|1b / dcp / 0|meta r1|cold-load|11.4089|11.4089|0.0000|0.0000|0.0000|7.8246|0.0000|
|1b / dcp / 0|meta r1|save|14.6047|3.9248|0.0000|0.0000|1.2023|0.0000|10.6799|
|1b / dcp / 0|meta r1|warm-load|3.8890|3.8890|0.0000|0.0000|0.0000|0.6654|0.0000|
|1b / dcp / 0|meta-sim r1|cold-load|11.5504|11.5504|0.0000|0.5134|0.0000|7.4290|0.0000|
|1b / dcp / 0|meta-sim r1|save|16.7906|5.6733|1.4226|0.0000|1.2092|0.0000|11.1174|
|1b / dcp / 0|meta-sim r1|warm-load|4.5650|4.5650|0.0000|0.5132|0.0000|0.7001|0.0000|
|1b / safetensors / 0|cpu r1|cold-load|11.4008|11.4008|0.0000|0.0000|0.0000|9.1852|0.0000|
|1b / safetensors / 0|cpu r1|save|14.3528|3.3189|0.0000|0.0000|1.5363|0.0000|11.0339|
|1b / safetensors / 0|cpu r1|warm-load|2.6484|2.6484|0.0000|0.0000|0.0000|0.6411|0.0000|
|1b / safetensors / 0|gpu r1|cold-load|11.4314|11.4314|0.0000|0.5113|0.0000|7.7324|0.0000|
|1b / safetensors / 0|gpu r1|save|15.6514|4.6196|1.4154|0.0000|1.3719|0.0000|11.0318|
|1b / safetensors / 0|gpu r1|warm-load|4.1516|4.1516|0.0000|0.5106|0.0000|0.6793|0.0000|
|1b / safetensors / 0|meta r1|cold-load|11.3091|11.3091|0.0000|0.0000|0.0000|8.0416|0.0000|
|1b / safetensors / 0|meta r1|save|13.9105|3.2306|0.0000|0.0000|1.5491|0.0000|10.6799|
|1b / safetensors / 0|meta r1|warm-load|3.6550|3.6550|0.0000|0.0000|0.0000|0.6598|0.0000|
|1b / safetensors / 0|meta-sim r1|cold-load|11.5149|11.5149|0.0000|0.5136|0.0000|7.7342|0.0000|
|1b / safetensors / 0|meta-sim r1|save|15.5901|4.5498|1.4242|0.0000|1.3463|0.0000|11.0403|
|1b / safetensors / 0|meta-sim r1|warm-load|4.1726|4.1726|0.0000|0.5132|0.0000|0.6709|0.0000|
|1b / dcp / 0|cpu r1|cold-load|12.0712|12.0712|0.0000|0.0000|0.0000|9.8964|0.0000|
|1b / dcp / 1|cpu r1|cold-load|12.0712|12.0712|0.0000|0.0000|0.0000|8.9335|0.0000|
|1b / dcp / 2|cpu r1|cold-load|12.0713|12.0713|0.0000|0.0000|0.0000|9.9672|0.0000|
|1b / dcp / 3|cpu r1|cold-load|12.0712|12.0712|0.0000|0.0000|0.0000|10.0912|0.0000|
|1b / dcp / 0|cpu r1|save|14.3483|12.4235|0.0000|0.0000|0.3061|0.0000|1.9248|
|1b / dcp / 1|cpu r1|save|14.3636|12.4435|0.0000|0.0000|0.2977|0.0000|1.9200|
|1b / dcp / 2|cpu r1|save|14.3620|12.2237|0.0000|0.0000|0.3029|0.0000|2.1383|
|1b / dcp / 3|cpu r1|save|14.3619|12.4401|0.0000|0.0000|0.3692|0.0000|1.9218|
|1b / dcp / 0|cpu r1|warm-load|1.9658|1.9658|0.0000|0.0000|0.0000|0.1901|0.0000|
|1b / dcp / 1|cpu r1|warm-load|1.9659|1.9659|0.0000|0.0000|0.0000|0.1873|0.0000|
|1b / dcp / 2|cpu r1|warm-load|1.9657|1.9657|0.0000|0.0000|0.0000|0.1892|0.0000|
|1b / dcp / 3|cpu r1|warm-load|1.9659|1.9659|0.0000|0.0000|0.0000|0.1825|0.0000|
|1b / dcp / 0|meta r1|cold-load|11.9784|11.9784|0.0000|0.0000|0.0000|6.4639|0.0000|
|1b / dcp / 1|meta r1|cold-load|11.9784|11.9784|0.0000|0.0000|0.0000|8.4511|0.0000|
|1b / dcp / 2|meta r1|cold-load|11.9784|11.9784|0.0000|0.0000|0.0000|9.7140|0.0000|
|1b / dcp / 3|meta r1|cold-load|11.9784|11.9784|0.0000|0.0000|0.0000|9.7627|0.0000|
|1b / dcp / 0|meta r1|save|14.1672|12.2345|0.0000|0.0000|0.3473|0.0000|1.9327|
|1b / dcp / 1|meta r1|save|14.1776|12.2523|0.0000|0.0000|0.3109|0.0000|1.9253|
|1b / dcp / 2|meta r1|save|14.1776|12.2600|0.0000|0.0000|0.3048|0.0000|1.9176|
|1b / dcp / 3|meta r1|save|14.1776|12.0877|0.0000|0.0000|0.3010|0.0000|2.0899|
|1b / dcp / 0|meta r1|warm-load|2.2538|2.2538|0.0000|0.0000|0.0000|0.1960|0.0000|
|1b / dcp / 1|meta r1|warm-load|2.2539|2.2539|0.0000|0.0000|0.0000|0.1931|0.0000|
|1b / dcp / 2|meta r1|warm-load|2.2540|2.2540|0.0000|0.0000|0.0000|0.1757|0.0000|
|1b / dcp / 3|meta r1|warm-load|2.2540|2.2540|0.0000|0.0000|0.0000|0.1767|0.0000|
|1b / safetensors / 0|cpu r1|cold-load|12.1461|12.1461|0.0000|0.0000|0.0000|10.1131|0.0000|
|1b / safetensors / 1|cpu r1|cold-load|12.1461|12.1461|0.0000|0.0000|0.0000|10.0523|0.0000|
|1b / safetensors / 2|cpu r1|cold-load|12.1461|12.1461|0.0000|0.0000|0.0000|9.9694|0.0000|
|1b / safetensors / 3|cpu r1|cold-load|12.1460|12.1460|0.0000|0.0000|0.0000|10.3763|0.0000|
|1b / safetensors / 0|cpu r1|save|12.1927|10.2607|0.0000|0.0000|0.3660|0.0000|1.9320|
|1b / safetensors / 1|cpu r1|save|12.1981|10.2833|0.0000|0.0000|0.3454|0.0000|1.9149|
|1b / safetensors / 2|cpu r1|save|12.1974|10.2800|0.0000|0.0000|0.3487|0.0000|1.9175|
|1b / safetensors / 3|cpu r1|save|12.1982|10.2788|0.0000|0.0000|0.5115|0.0000|1.9194|
|1b / safetensors / 0|cpu r1|warm-load|1.9027|1.9027|0.0000|0.0000|0.0000|0.1841|0.0000|
|1b / safetensors / 1|cpu r1|warm-load|1.9027|1.9027|0.0000|0.0000|0.0000|0.1581|0.0000|
|1b / safetensors / 2|cpu r1|warm-load|1.9028|1.9028|0.0000|0.0000|0.0000|0.1834|0.0000|
|1b / safetensors / 3|cpu r1|warm-load|1.9027|1.9027|0.0000|0.0000|0.0000|0.1755|0.0000|
|1b / safetensors / 0|meta r1|cold-load|12.2707|12.2707|0.0000|0.0000|0.0000|10.1603|0.0000|
|1b / safetensors / 1|meta r1|cold-load|12.2707|12.2707|0.0000|0.0000|0.0000|10.2741|0.0000|
|1b / safetensors / 2|meta r1|cold-load|12.2708|12.2708|0.0000|0.0000|0.0000|10.3457|0.0000|
|1b / safetensors / 3|meta r1|cold-load|12.2709|12.2709|0.0000|0.0000|0.0000|9.9886|0.0000|
|1b / safetensors / 0|meta r1|save|12.2365|10.3141|0.0000|0.0000|0.3488|0.0000|1.9224|
|1b / safetensors / 1|meta r1|save|12.2469|10.3219|0.0000|0.0000|0.3367|0.0000|1.9250|
|1b / safetensors / 2|meta r1|save|12.2434|9.9897|0.0000|0.0000|0.3676|0.0000|2.2537|
|1b / safetensors / 3|meta r1|save|12.2468|10.2877|0.0000|0.0000|0.3603|0.0000|1.9590|
|1b / safetensors / 0|meta r1|warm-load|2.2089|2.2089|0.0000|0.0000|0.0000|0.1849|0.0000|
|1b / safetensors / 1|meta r1|warm-load|2.2090|2.2090|0.0000|0.0000|0.0000|0.1955|0.0000|
|1b / safetensors / 2|meta r1|warm-load|2.2091|2.2091|0.0000|0.0000|0.0000|0.1984|0.0000|
|1b / safetensors / 3|meta r1|warm-load|2.2091|2.2091|0.0000|0.0000|0.0000|0.1786|0.0000|

|Model / format / ranks|Phase|Full sequence|Payload sequence|
|---|---|---|---|
|1b / dcp / 1|cold-load|identical|identical|
|1b / dcp / 1|save|identical|identical|
|1b / dcp / 1|warm-load|identical|identical|
|1b / dcp / 4|cold-load|identical|identical|
|1b / dcp / 4|save|identical|identical|
|1b / dcp / 4|warm-load|identical|identical|
|1b / safetensors / 1|cold-load|identical|identical|
|1b / safetensors / 1|save|identical|identical|
|1b / safetensors / 1|warm-load|identical|identical|
|1b / safetensors / 4|cold-load|identical|identical|
|1b / safetensors / 4|save|identical|identical|
|1b / safetensors / 4|warm-load|identical|identical|

Meta+sim minus GPU (paired repeats, per rank):

|Model / format / rank|Phase|Median seconds|Median percent|
|---|---|---:|---:|
|1b / dcp / 0|cold-load|+0.1209|+1.06%|
|1b / dcp / 0|save|+0.1286|+2.32%|
|1b / dcp / 0|warm-load|+0.0864|+1.93%|
|1b / safetensors / 0|cold-load|+0.0835|+0.73%|
|1b / safetensors / 0|save|-0.0698|-1.51%|
|1b / safetensors / 0|warm-load|+0.0210|+0.50%|
