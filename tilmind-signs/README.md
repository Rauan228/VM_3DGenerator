# Til Mind — готовые 3D-анимации жестов РЖЯ

Готовые анимации жестов для сайта: два аватара Til Mind (парень и девушка) и клипы анимации к ним — **300 слов, которые
знает модель распознавания** (список — `model-words.json`, снимок `/v1/words`). Каждый клип двигает аватара в точности
как плеер 3DAnimationGPT, это проверяется при выгрузке.

## Откуда жесты

- **292 жеста — датасет Slovo** ([github.com/hukenovs/slovo](https://github.com/hukenovs/slovo)): на нём обучена наша модель
  распознавания. Для каждого слова из 20 записей Slovo выбрано самое типичное и лучше всего снятое исполнение; записи
  левой рукой и зеркальные (селфи) отражены, чтобы аватар везде показывал правой рукой.
- **8 жестов — signflow.ru** (белый, красный, синий, серый, коричневый, розовый, оранжевый, фиолетовый): эти слова есть и в
  Slovo, это тот же жест, оставлена студийная запись signflow.ru (видео используются по договорённости с владельцами сайта).

У каждого жеста в `signs.json` есть поле `source`. **Лицензия Slovo — вариант CC BY-SA 4.0:** анимации из Slovo можно
использовать на сайте, если указать источник (например, «Часть жестов — по датасету Slovo, github.com/hukenovs/slovo,
CC BY-SA 4.0») и распространять сами анимации под той же лицензией.

**Причёска парня** — модель Vincent Page из коллекции «FREE Male Fashion Hair collection 01 lowpoly»
([sketchfab.com/vincentpage](https://sketchfab.com/vincentpage)), **лицензия CC BY 4.0: на сайте, где показывается аватар,
нужно указать автора**, например: «Причёска аватара — Vincent Page, sketchfab.com/vincentpage, CC BY 4.0».

**Кисти и пальцы** во всех жестах сняты с видео трекером WiLoR. Его веса (CC BY-NC-ND) и модель кисти MANO разрешены
только для некоммерческого использования: для коммерческого сайта нужна лицензия их правообладателей или пересчёт
анимаций другим трекером.

## Что в папке

| Путь | Что это |
|---|---|
| `avatars/tilmind-male.glb`, `avatars/tilmind-female.glb` | Аватары: скелет, лицо с 52 формами мимики ARKit, глаза, рот с зубами и языком, одежда с надписью «Til Mind» |
| `clips/male/<slug>.json`, `clips/female/<slug>.json` | Клип жеста для своего аватара (формат `THREE.AnimationClip.toJSON`): повороты костей, мимика и артикуляция губами, 60 кадров в секунду |
| `signs.json` | Список жестов: `slug`, слово, длительность, пути к клипам, источник |
| `model-words.json` | Слова модели распознавания, по которым собран пакет |
| `react/TilMindSign.tsx` | Готовый компонент для React + three.js (`@react-three/fiber`, `@react-three/drei`) |

Клип сделан под своего аватара, потому что у парня и девушки разные пропорции. Клипы из `clips/male/` проигрывайте на
`tilmind-male.glb`, из `clips/female/` на `tilmind-female.glb`. Аватары и клипы обновляются только вместе: в клипах есть
дорожки зубов, языка и полости рта нового аватара.

## Как подключить к сайту (React)

1. Скопируйте папку в `public/` сайта, например в `public/tilmind-signs/`.
2. Скопируйте `react/TilMindSign.tsx` в `src/components/`.
3. Покажите жест:

```tsx
import { TilMindSignViewer } from "@/components/TilMindSign";

<TilMindSignViewer
  avatarUrl="/tilmind-signs/avatars/tilmind-male.glb"
  clipUrl="/tilmind-signs/clips/male/volk.json"
  isPlaying
  loopMode="repeat"      // или "once"
  playbackRate={1}       // 0.5 — вдвое медленнее
/>
```

Список слов для словаря берётся из `signs.json`:

```ts
const signs = await fetch("/tilmind-signs/signs.json").then((r) => r.json());
// [{ slug: "volk", word: "волк", duration: 1.767, clips: { male: "clips/male/volk.json", female: "clips/female/volk.json" },
//    source: "Slovo dataset (github.com/hukenovs/slovo), CC BY-SA 4.0" }, ...]
```

`TilMindSign` без обёртки ставится внутрь своего `<Canvas>`. Аватар стоит на `y = 0` в реальный рост (парень 1,76 м,
девушка 1,66 м). Так можно настроить свою камеру.

Без React в чистом three.js:

```js
const gltf = await new GLTFLoader().loadAsync("tilmind-signs/avatars/tilmind-male.glb");
const clip = THREE.AnimationClip.parse(await (await fetch("tilmind-signs/clips/male/volk.json")).json());
const mixer = new THREE.AnimationMixer(gltf.scene);
mixer.clipAction(clip).play();   // и mixer.update(dt) в каждом кадре
```

## Как пересобрать

В проекте 3DAnimationGPT нужно запустить бэкенд на `:8010` и Vite на `:5173` (`tools/viewer.sh servers`), потом выполнить
`<python с playwright> tools/export_clips.py --words data/model-vocabulary.json` (подробности в шапке скрипта). Без
`--words` выгружаются все жесты галереи.
