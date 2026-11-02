# Serial vs concurrent save

Each round uses five paired Meta/CPU samples. Cross-round speedup is a descriptive ratio of medians: the rounds were measured separately and are not temporally paired. API interval overlaps do not measure device concurrency.

|Cell|Config|Wall serial→concurrent s|Speedup|Compressed span serial→concurrent s|Speedup|pwrite max ranks serial→concurrent|fsync max ranks serial→concurrent|pwrite∩fsync union serial→concurrent s|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|1b-4r-dcp|meta|14.374→13.137|1.09×|6.355→2.324|2.73×|1→4|1→4|0.000→0.082|
|1b-4r-dcp|cpu|14.366→13.100|1.10×|6.430→2.308|2.79×|1→4|1→4|0.000→0.046|
|1b-4r-safetensors|meta|11.848→12.422|0.95×|4.005→1.501|2.67×|1→4|1→4|0.000→0.069|
|1b-4r-safetensors|cpu|11.786→12.500|0.94×|4.017→1.495|2.69×|1→4|1→4|0.000→0.072|
|1b-8r-dcp|meta|14.047→13.292|1.06×|10.200→2.568|3.97×|1→8|1→8|0.000→0.364|
|1b-8r-dcp|cpu|14.109→13.294|1.06×|10.212→2.513|4.06×|1→8|1→8|0.000→0.578|
|1b-8r-safetensors|meta|12.003→12.642|0.95×|6.503→1.821|3.57×|1→8|1→8|0.000→0.236|
|1b-8r-safetensors|cpu|11.974→12.482|0.96×|6.425→1.760|3.65×|1→8|1→8|0.000→0.258|
|3b-4r-dcp|meta|51.630→40.147|1.29×|19.025→18.396|1.03×|1→4|1→4|0.000→2.085|
|3b-4r-dcp|cpu|51.793→40.225|1.29×|18.896→18.691|1.01×|1→4|1→4|0.000→1.926|
|3b-4r-safetensors|meta|46.834→38.649|1.21×|13.938→17.015|0.82×|1→4|1→4|0.000→2.350|
|3b-4r-safetensors|cpu|47.289→38.677|1.22×|14.050→16.872|0.83×|1→4|1→4|0.000→2.490|
|3b-8r-dcp|meta|56.321→42.380|1.33×|27.703→12.476|2.22×|1→8|1→8|0.000→12.363|
|3b-8r-dcp|cpu|56.599→42.389|1.34×|27.872→11.454|2.43×|1→8|1→8|0.000→13.752|
|3b-8r-safetensors|meta|46.917→40.082|1.17×|18.818→14.972|1.26×|1→8|1→8|0.000→7.853|
|3b-8r-safetensors|cpu|47.352→40.132|1.18×|19.036→13.222|1.44×|1→8|1→8|0.000→9.617|

Complete per-repeat and per-rank intervals/overlap, shape quantiles, paired alignment and percentage-point changes are retained in serial-comparison.json. Exact merged absolute intervals and verified raw-trace SHA256 are retained under overlap-intervals/.
