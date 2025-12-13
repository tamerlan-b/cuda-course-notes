### Профилирование с помощью ncu

Компиляция:
```bash
nvcc test.cu -o test
```

```bash
ncu ./test
```

Если команда выше не работает (в zsh и Linux), то можно более полную версию написать:
```bash
sudo /usr/local/cuda/bin/ncu ./test
```

Показать какую-то конкретную секцию:
```bash
ncu --section LaunchStats ./test
```

Полный список доступных секций можно посмотреть тут: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules


Получение списка всех возможных метрик:
```bash
ncu --query-metrics-mode all > all_metrics.txt
```

Вычисление определенных метрик при работе приложения:
```bash
ncu --metrics sm__inst_executed.avg,sm__inst_executed.min,sm__inst_executed.max,sm__inst_executed.sum ./test
```

Команда ниже эвивалентна команде выше:
```bash
ncu --metrics sm__inst_executed ./test
```

Сохранение метрик в csv:
```bash
ncu --metrics sm__inst_executed --csv ./test > metrics.csv
```

