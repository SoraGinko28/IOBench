# gdtensor trace analysis

Times are rank-local seconds. Save excludes the union of fsync spans; loads use the complete checkpoint call. Payload/header requests are compared separately. No cross-rank scheduling order is compared.

|Model / format / rank|Config|Phase|Total|Save excl. fsync|D2H|H2D|pwrite|Read|fsync|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|1b / dcp / 0|gpu r1|cold-load|11.4837|11.4837|0.0000|0.5182|0.0000|7.0887|0.0000|
|1b / dcp / 0|gpu r1|save|16.7552|6.1512|1.6015|0.0000|1.3733|0.0000|10.6040|
|1b / dcp / 0|gpu r1|warm-load|4.8548|4.8548|0.0000|0.5156|0.0000|0.8080|0.0000|
|1b / dcp / 0|meta-sim r1|cold-load|11.4728|11.4728|0.0000|0.5135|0.0000|7.0453|0.0000|
|1b / dcp / 0|meta-sim r1|save|16.2036|5.5330|1.4211|0.0000|1.1777|0.0000|10.6706|
|1b / dcp / 0|meta-sim r1|warm-load|4.7451|4.7451|0.0000|0.5135|0.0000|0.7752|0.0000|
|1b / safetensors / 0|cpu r1|cold-load|11.5244|11.5244|0.0000|0.0000|0.0000|9.1680|0.0000|
|1b / safetensors / 0|cpu r1|save|15.1145|4.0922|0.0000|0.0000|2.2355|0.0000|11.0223|
|1b / safetensors / 0|cpu r1|warm-load|2.6668|2.6668|0.0000|0.0000|0.0000|0.6244|0.0000|
|1b / safetensors / 0|gpu r1|cold-load|11.6646|11.6646|0.0000|0.5113|0.0000|7.8628|0.0000|
|1b / safetensors / 0|gpu r1|save|15.7384|4.7001|1.4572|0.0000|1.3708|0.0000|11.0384|
|1b / safetensors / 0|gpu r1|warm-load|4.3333|4.3333|0.0000|0.5099|0.0000|0.7281|0.0000|
|1b / safetensors / 0|meta r1|cold-load|11.3294|11.3294|0.0000|0.0000|0.0000|8.0438|0.0000|
|1b / safetensors / 0|meta r1|save|14.7848|4.1333|0.0000|0.0000|2.3315|0.0000|10.6515|
|1b / safetensors / 0|meta r1|warm-load|3.6770|3.6770|0.0000|0.0000|0.0000|0.6795|0.0000|
|1b / safetensors / 0|meta-sim r1|cold-load|11.7042|11.7042|0.0000|0.5133|0.0000|7.9341|0.0000|
|1b / safetensors / 0|meta-sim r1|save|15.4508|4.5454|1.4216|0.0000|1.3034|0.0000|10.9054|
|1b / safetensors / 0|meta-sim r1|warm-load|4.2201|4.2201|0.0000|0.5133|0.0000|0.6847|0.0000|

|Model / format / ranks|Phase|Full sequence|Payload sequence|
|---|---|---|---|
|1b / dcp / 1|cold-load|identical|identical|
|1b / dcp / 1|save|identical|identical|
|1b / dcp / 1|warm-load|identical|identical|
|1b / safetensors / 1|cold-load|identical|identical|
|1b / safetensors / 1|save|identical|identical|
|1b / safetensors / 1|warm-load|identical|identical|

Meta+sim minus GPU (paired repeats, per rank):

|Model / format / rank|Phase|Median seconds|Median percent|
|---|---|---:|---:|
|1b / dcp / 0|cold-load|-0.0109|-0.09%|
|1b / dcp / 0|save|-0.6182|-10.05%|
|1b / dcp / 0|warm-load|-0.1097|-2.26%|
|1b / safetensors / 0|cold-load|+0.0397|+0.34%|
|1b / safetensors / 0|save|-0.1547|-3.29%|
|1b / safetensors / 0|warm-load|-0.1132|-2.61%|
