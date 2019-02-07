# lingofunk-generate

## Вкратце

Модуль для генерации текстов с помощью [textgenrnn](https://github.com/minimaxir/textgenrnn).

Обращаясь к Flask-серверу, можно получать тексты в стилях:

    * `positive`
    * `negative`
    * `neutral`

Для каждого стиля обучается своя `textgenrnn` модель.
Обучается на .csv файле, где в табличке должны быть по крайней мере две колонки:

    * колонка с текстами
    * колонка со стилевыми метками

Предварительно указывается, какие метки к каким стилям относятся.

Репозиторий содержит

    * `lingofunk_generate`: модуль с .py файлами
    * `scripts`: скрипты, с помощью которых можно работать с модулем
    * `notebooks`: папка с ноутбуком, где происходило кое-какое знакомство с `textgenrnn`
    * `models`: папка, куда будут сохраняться модели
    * `data`: папка с данными для обучения

## lingofunk_generate

    * `constants.py`: некоторые константы, которые используются в других файлах модуля. В частности, там указываются стили (`text_style`), по которым будут обучаться модели
    * `utils.py`: пара функций "общего" назначения
    * `model_restore_utils.py`: функции, связанные с сохранением и восстановлением `textgenrnn` моделей (как предобученных, так и новых)
    * `fit.py`: обучение моделей.
    Файл можно запускать.
    Про параметры запуска можно узнать из `help` справки `python fi.py -h`.
    На всякий случай скопирую `help` строки и сюда:
        * `--debug`: Specify, if want to run fitting in debug mode. It means that if parameter nrows is also specified, only first nrows will be read from .csv file
        * `--nrows`: How many rows should be read in debug mode
        * `--data-path`: Path to data .csv file
        * `--text-col`: Text column name in data file
        * `--label-col`: Style label column name in data file
        * `--word-level`: Specify, if want to build word-level models (instead of default char-level models)
        * `--new-model`: Specify, if want to get new textgenrnn model, not pretrained one
        * `--train-size`: Train size (validation size = 1.0 - train size)
        * `--dropout Dropout (the proportion of tokens to be thrown away on each epoch)
        * `--num-epochs`: Number of epochs to train the model
        * `--gen-epochs`: Number of epochs, after each of which sample text generations ny the model will be displayed in console
        * `--max-length`: Maximum number of previous tokens (words or chars) to take into account while predicting the next one
        * `--max-gen-length`: Maximum number of tokens to generate as sample after gen_epochs
        * `--labels-<text_style>`: Texts of which labels should be treated as ones of style "<text_style>"
    * `server.py`:
        * `--port`: The port to listen on (default is 8001)
        * `--seed`: Random seed
        * `--temperature`: Low temperature (eg 0.2) makes the model more confident but also more conservative when generating response. High temperatures (eg 0.9, values bigger than 1.0 also possible) make responses diverse, but mistakes are also more likely to take place

## scripts

    * `link_data.sh`: делает ссылку в папке `data` на необходимый для обучения моделей .csv файл.
    При запуске скрипта необходимо указать аргументы
        1. Путь до папки с данными (где фактически лежит необходимый .csv файл)
        2. Путь до папки с проектом
        3. Имя файла, ссылку на который нужно создать в папке `data` проекта
    * `fit.sh`: Запуск `fit.py` с заданными внутри скрипта параметрами
    * `deploy.sh`: Запуск Flask-сервера, который будет обрабатывать запросы по генерации текста