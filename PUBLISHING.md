# Публикация на GitHub

Пошагово, что сделать после скачивания папки.

## 1. Заменить плейсхолдеры

В трёх файлах стоит `USERNAME` и в одном — имя автора:

```bash
# Подставьте свой логин GitHub
grep -rl USERNAME README.md README.ru.md | xargs sed -i 's/USERNAME/ваш-логин/g'

# И имя в лицензии
sed -i 's/<ВАШЕ ИМЯ>/Ваше Имя/' LICENSE
```

На macOS у `sed` другой синтаксис: `sed -i '' 's/.../.../'`.

## 2. Создать репозиторий

```bash
git init
git add .
git commit -m "Steppe fire spread from FIRMS hotspots and hourly wind"
git branch -M main
git remote add origin git@github.com:ваш-логин/steppe-fire-era5.git
git push -u origin main
```

## 3. Включить GitHub Pages

Settings → Pages → Source: **Deploy from a branch** → ветка `main`, папка `/docs`.

Через минуту демо будет доступно по адресу
`https://ваш-логин.github.io/steppe-fire-era5/` — это готовая ссылка для поста
в LinkedIn.

## 4. Заполнить About

В шапке репозитория, шестерёнка справа:

- **Description:** `Retrospective steppe wildfire analysis: FIRMS hotspots, ERA5 wind, validated against satellite overpasses`
- **Website:** ссылка на GitHub Pages
- **Topics:** `wildfire`, `remote-sensing`, `nasa-firms`, `era5`, `reanalysis`,
  `python`, `folium`, `kazakhstan`, `geospatial`, `dbscan`

Темы важны: по ним репозиторий находят через поиск GitHub.

## 5. Проверить, что CI зелёный

Вкладка Actions — сборка идёт на Python 3.10, 3.11 и 3.12, гоняет линтер,
78 тестов и собирает демо-карту офлайн. Зелёный бейдж в README появится сам.

## Что стоит добавить позже

- **Скриншот или гифка в начале README.** Сейчас там сразу текст. Запись
  экрана с проходом временной шкалы даст на порядок больше просмотров.
  Положите файл в `docs/preview.gif` и вставьте `![Demo](docs/preview.gif)`
  под бейджами.
- **Реальный прогон.** Демо идёт на синтетике; когда прогоните с ключом на
  настоящих данных, сохраните карту в `docs/` вторым файлом.
- **Раздел Citation.** Если проект попадёт в чью-то работу, пригодится
  `CITATION.cff`.

## 6. Связать с парным проектом

В README уже стоит ссылка на `fire-spread-sandbox`. Проверьте, что обратная
ссылка там ведёт сюда — так посетитель одного проекта находит второй.
