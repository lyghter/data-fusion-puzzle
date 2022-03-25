# Data Fusion 2022: быстрое нейросетевое решение задачи Puzzle

* Результаты на публичном лидерборде

| R1 | MRR@100 | Precision@100 |
| :---: | :---: | :---: |
| 0.0130320869 | 0.0069565311 | 0.1028890015 |

* Предобработка данных: **24 минуты**
* Обучение модели на 2/3 тренировочной выборки: **28 минут**
* Применение модели к тестовой выборке: **19 секунд** (34 секунды, если запускается не сразу после обучения)

* Количество параметров модели: **506 K**

* Размер файла, содержащего параметры модели и оптимизатора: **6 MB**

* Машина: 8 CPU, 30 GB RAM, 1 GPU (RTX5000, 16GB) 


## Воспроизведение

Необходимые пакеты можно установить через pip:
```
git clone https://github.com/lyghter/data-fusion-puzzle
cd data-fusion-puzzle
pip install -r env/requirements.txt
# eсли скорость загрузки пакетов мала
# pip install -r env/requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
либо использовать докер-образ `lyghter/gradient:data-fusion-puzzle`

Чтобы обучить модель, нужно запустить ноутбук [run.ipynb](ipynb/run.ipynb). При первом запуске будет создана папка `data`, куда с yandexcloud будут загружены файлы для дальнейшей предобработки. При повторном запуске загрузка и предобработка будет пропущена, будут использованы предобработанные данные. Веса модели сохранятся в папке `log`. После обучения следует изменить значение переменной `TASK` на "predict" и перезапустить ноутбук.

Ноутбук [plot.ipynb](ipynb/plot.ipynb) строит графики кривых обучения.

Примечание. После изменения параметров `splitter` и `splitter_pp` необходимо удалить из папки `data` файлы `XC.pt`,`YC.pt`,`XT.pt`,`YT.pt`, чтобы они были созданы новым сплиттером.

Если при запуске возникнут ошибки - сообщите автору, он попытается всё исправить :)

## Модель

Для получения векторных представлений последовательностей действий, совершаемых клиентами банка и провайдера цифровых услуг, данное решение использует нейросетевую модель BigBird, реализованную с помощью фреймворка Pytorch в библиотеке Transformers. BigBird имеет линейную зависимость объёма используемой памяти от длины обрабатываемой последовательности, в отличие от оригинальной версии Transformer, где зависимость квадратичная.

В данном решении модель обучается на данных только тех клиентов, для которых установлено соответствие между аккаунтами банка и провайдера.

Для построения эмбеддингов транзакции и кликстримы разделяются по календарным месяцам. На вход нейронной сети подаются последовательности целых чисел, кодирующих категориальные признаки: `cat_id` для кликстримов и `mcc_code` для транзакций (использование `currency_rk` и `transaction_amt` может улучшить точность модели). На выходе получаются векторные представления этих последовательностей. Таким образом, каждый месяц активности каждого клиента преобразуется в вектор. Поскольку одна нейронная сеть обрабатывает как транзакции, так и кликстримы, получаемые вектора принадлежат одному векторному пространству. 

Для сопоставления клиентов банка с клиентами провайдера каждому клиенту ставится в соответствие усреднённый вектор его активности, затем для каждого клиента банка рассчитывается евклидово расстояние от его вектора до векторов клиентов провайдера, из которых выбираются 100 ближайших.

В качестве функции потерь используется `MarginLoss` из библиотеки PyTorch Metric Learning. При её вычислении информация о происхождении вектора (из транзакций или кликстрима) не используется. 

Модель обучается с использованием фреймворка PyTorch Lightning.

## Результаты

Для оценки качества модели использовалась трёхфолдовая валидация, затем модель обучалась на полной обучающей выборке. Результаты представлены ниже. 

`[#]` - часть выборки, используемая для обучения модели

`[ ]` - часть выборки, используемая для проверки модели
		
Fold 1
| `[ ][#][#]` | local | public |
| :---: | :---: | :---: |
| R1 | 0.0133225331 | 0.0130320869 |
| MRR@100 | 0.0072424510 | 0.0069565311 |
| Precision@100 | 0.0830096095 | 0.1028890015 |

Fold 2
| `[#][ ][#]` | local | public |
| :---: | :---: | :---: |
| R1 | 0.0129775553 | 0.0110190587 |
| MRR@100 | 0.0070732435 | 0.0058535996 |
| Precision@100 | 0.0785276074 | 0.0937658388 |

Fold 3
| `[#][#][ ]` | local | public |
| :---: | :---: | :---: |
| R1 | 0.0129508933 | 0.0091964213 |
| MRR@100 | 0.0069980728 | 0.0048326666 |
| Precision@100 | 0.0867075665 | 0.0947795236 |
		
Full train
| `[#][#][#]`  | public |
| :---: | :---: |
| R1 | 0.0129382529 |
| MRR@100 | 0.006905455 |
| Precision@100 | 0.1023821591 |

Из таблицы видно, что точность модели, обученной на всей тренировочной выборке не превзошла точность модели, обученной на части тренировочной выборки. По-видимому, добавление данных сильно влияет на кривую обучения. Чтобы сократить bias, можно увеличить количество фолдов.

Количество эпох (10) было выбрано как оптимальное после построения многих кривых обучения. При дальнейшем обучении происходили колебания значения метрики на валидационной выборке, рост метрики на публичном лидерборде прекращался.









