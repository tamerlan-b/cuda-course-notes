### Сложение двух векторов размером 1024 на GPU с использованием CUDA

Но дополнительно c:
- замером времени работы кода на GPU
- использованием множества SM, что увеличивается быстродействие кода

Компиляция:
```bash
nvcc kernel.cu -o kernel
```

Замер времени работы кода на GPU:
```c++
#include "cuda_runtime.h"
#include <cuda.h>
...

// Создаем события для замера времени
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Замеряем начало выполнения кода
cudaEventRecord(start);

// Вызываем CUDA-код
// kernel<<<1024, 1024>>>();
// Ожидаем завершения выполнения кода на GPU
cudaDeviceSynchronize();

// Замеряем окончание выполнения кода
cudaEventRecord(stop);
cudaEventSynchronize(stop);
// Вычисяем затраченное время
float milliseconds = 0.F;
cudaEventElapsedTime(&milliseconds, start, stop);
...
```