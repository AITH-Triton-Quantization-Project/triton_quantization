# Замер INT4 на Llama-3.2-1B

Курс Efficient NN. Бриленков Евгений, Головкин Иван, Чуфистов Георгий.

Модель `unsloth/Llama-3.2-1B-Instruct`. Заменены 113 линейных слоёв: 16 блоков по 7 проекций и `lm_head`. `embed_tokens` оставлен как есть. Веса в `bfloat16`. Квантование симметричное INT4, группа 128 вдоль оси входа, коды от -8 до 7, два соседних кода в одном `int8`. Слой — `QuantizedLinear` с `backend="reference"`.

Прогон: Slurm job 19611, нода `aihub`, раздел `aichem`, 27.11.2026, 11:27–11:32. Сырые числа лежат в [a6000.json](a6000.json).

## Видеокарта

Снято `nvidia-smi` на `aihub` в день прогона. На ноде шесть одинаковых RTX A6000. Процесс видел все шесть и взял `cuda:0`: свободно 48268 МиБ из 49140.

| Параметр | Значение |
| --- | --- |
| Имя | NVIDIA RTX A6000 |
| Compute capability | 8.6 |
| Драйвер | 580.178.04 |
| CUDA в драйвере | 13.0 |
| VBIOS | 94.02.5C.00.02 |
| Память по `nvidia-smi` | 49140 МиБ |
| Макс. частота SM | 2100 МГц |
| Макс. частота памяти | 8001 МГц |
| Лимит мощности | 300 Вт |

Паспорт карты — [datasheet NVIDIA, май 2022](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a6000-datasheet.pdf). Пиковые TFLOPS там считаются на boost clock. Тензорная цифра дана со sparsity.

| Параметр | Значение |
| --- | --- |
| Архитектура | NVIDIA Ampere |
| CUDA-ядра | 10752 |
| Tensor-ядра | 336, 3-е поколение |
| RT-ядра | 84, 2-е поколение |
| Память | 48 ГБ GDDR6, ECC |
| Шина | 384 бит |
| Пропускная способность | 768 ГБ/с |
| FP32 | 38.7 TFLOPS |
| Tensor | 309.7 TFLOPS со sparsity |
| Интерфейс | PCIe 4.0 x16 |
| Потребление платы | 300 Вт |
| NVLink | две карты, 112.5 ГБ/с в обе стороны |

Capability 8.6, поэтому скрипт берёт `torch.bfloat16`.

## Окружение

| | |
| --- | --- |
| Нода | `aihub`, Ubuntu 22.04.5 LTS, ядро 5.15.0-198-generic |
| Slurm | 21.08.5 |
| Job | 19611, раздел `aichem` |
| Запрос | `gpu:a6000:1`, 4 CPU, 32 ГБ RAM, лимит 6 часов |
| Факт | 5 минут, устройство `cuda:0` |
| Python | `/usr/bin/python3.10`, CPython 3.10.12, собран GCC 11.4.0 |
| gcc на ноде | Ubuntu 11.4.0-1ubuntu1~22.04.3 |
| nvcc на ноде | 12.6.85 |
| CUDA колеса PyTorch | 12.4 (`torch==2.6.0+cu124`) |
| Каталог прогона | `/mnt/tank/scratch/igolovkin/triton_quantization` |
| `HF_HOME` | `/mnt/tank/scratch/igolovkin/triton_quantization/.cache/huggingface` |

`torch` и `triton` импортируются из `/mnt/tank/scratch/igolovkin/.venv`. Остальные пакеты стоят в `.deps` (`pip install --target`, без замены torch). `PYTHONPATH`: корень репозитория, `.deps`, `site-packages` этого venv.

```
torch==2.6.0+cu124
triton==3.2.0
transformers==4.52.4
datasets==3.6.0
accelerate==1.7.0
huggingface-hub==0.36.2
tokenizers==0.21.4
safetensors==0.8.0
numpy==2.2.6
```

## Протокол

Перплексия: WikiText-2 raw, сплит test, куски текста склеены через пустую строку. Окно 2048, шаг 512. На перекрытии предыдущего окна `labels = -100`. Итог — `exp(sum(loss * новые токены) / токены)`. В замере 289077 токенов и 562 окна. Токенизатор предупреждает, что вся строка длиннее `model_max_length` 131072; в модель каждый раз уходит окно не длиннее 2048.

Скорость: полный forward, batch 1, случайные `input_ids` длины 128, 512 и 2048. Два прогрева, медиана пяти прогонов, интервалы CUDA events. Токены в секунду — длина последовательности, делённая на медианное время.

Генерация: `What is the capital of France? Answer in one short sentence.`, `do_sample=False`.

## Память весов линейных слоёв

| | байты | МиБ |
| --- | ---: | ---: |
| bf16, 113 `nn.Linear` | 2471493632 | 2357 |
| INT4, только коды | 617873408 | 589.25 |
| INT4, коды + масштабы + нули | 656490496 | 626.08 |

Коды занимают ровно четверть bf16-весов. Масштабы и нули хранятся в bf16, форма `[K / 128, N]`, нули равны нулю. С ними отношение 2357 / 626.08 ≈ 3.76.

## Перплексия WikiText-2

| Модель | Перплексия | Токены | Окна |
| --- | ---: | ---: | ---: |
| bf16 | 11.497 | 289077 | 562 |
| INT4, эталон | 15.119 | 289077 | 562 |

Полные значения: 11.497151748072858 и 15.119093512029844.

Оба ответа на промпт: `The capital of France is Paris.`

## Скорость forward

Медиана пяти прогонов. Строка INT4 — эталон PyTorch: вес распаковывается, активации и вес приводятся к bf16, умножение выполняется в fp32.

| Длина | bf16, мс | bf16, ток/с | INT4, мс | INT4, ток/с | INT4 / bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 12.83 | 9974 | 162.34 | 788 | 12.65 |
| 512 | 18.13 | 28242 | 205.54 | 2491 | 11.34 |
| 2048 | 66.30 | 30889 | 378.95 | 5404 | 5.72 |

## Triton

Сверка GEMM 128×256×256 с эталоном до числа не дошла. `triton==3.2.0` собирал `cuda_utils` через `/usr/bin/gcc` и остановился на строке `#include <Python.h>`: `fatal error: Python.h: No such file or directory`, каталог поиска `-I/usr/include/python3.10`. На `aihub` файла `/usr/include/python3.10/Python.h` нет. На входной машине этот заголовок есть. В [a6000.json](a6000.json) у поля `triton` статус `error`.

## Файлы прогона

- [a6000.json](a6000.json) — запись job 19611
- [run_llama_eval.py](run_llama_eval.py) — скрипт замера
- [aichem_llama_eval.sbatch](aichem_llama_eval.sbatch) — задание Slurm

На scratch job вызывал `scripts/run_llama_eval.py`. Текст скрипта и sbatch тот же, что в этой папке.
