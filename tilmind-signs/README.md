# Til Mind — готовые 3D-анимации жестов РЖЯ

Готовые анимации жестов для сайта: два аватара Til Mind (парень и девушка) и клипы анимации к ним, 421 жест. Каждый
клип двигает аватара в точности как плеер 3DAnimationGPT, это проверяется при выгрузке.

## Откуда жесты

- **130 жестов — signflow.ru** (видео используются по договорённости с владельцами сайта).
- **291 жест — датасет Slovo** ([github.com/hukenovs/slovo](https://github.com/hukenovs/slovo)): 300 слов, на которых обучена
  наша модель распознавания жестов. Ещё 9 из этих 300 слов (белый, красный, синий, серый, розовый, оранжевый,
  коричневый, фиолетовый, нравиться) уже были в наборе signflow.ru, и это тот же жест — для них оставлена студийная
  запись signflow.ru. Для каждого слова из 20 записей Slovo выбрано самое типичное и лучше всего снятое исполнение;
  записи левой рукой и зеркальные (селфи) отражены, чтобы аватар везде показывал правой рукой.

У каждого жеста в `signs.json` есть поле `source`. **Лицензия Slovo — вариант CC BY-SA 4.0:** анимации из Slovo можно
использовать на сайте, если указать источник (например, «Часть жестов — по датасету Slovo, github.com/hukenovs/slovo,
CC BY-SA 4.0») и распространять сами анимации под той же лицензией.

## Что в папке

| Путь | Что это |
|---|---|
| `avatars/tilmind-male.glb`, `avatars/tilmind-female.glb` | Аватары: скелет, лицо с 52 формами мимики ARKit, одежда с надписью «Til Mind» |
| `clips/male/<slug>.json`, `clips/female/<slug>.json` | Клип жеста для своего аватара (формат `THREE.AnimationClip.toJSON`): повороты костей и мимика, 60 кадров в секунду |
| `signs.json` | Список жестов: `slug`, слово, длительность, пути к клипам, источник |
| `react/TilMindSign.tsx` | Готовый компонент для React + three.js (`@react-three/fiber`, `@react-three/drei`) |

Клип сделан под своего аватара, потому что у парня и девушки разные пропорции. Клипы из `clips/male/` проигрывайте на
`tilmind-male.glb`, из `clips/female/` на `tilmind-female.glb`.

## Как подключить к сайту (React, как на tilmind.vercel.app)

1. Скопируйте папку в `public/` сайта, например в `public/tilmind-signs/`.
2. Скопируйте `react/TilMindSign.tsx` в `src/components/`.
3. Покажите жест:

```tsx
import { TilMindSignViewer } from "@/components/TilMindSign";

<TilMindSignViewer
  avatarUrl="/tilmind-signs/avatars/tilmind-male.glb"
  clipUrl="/tilmind-signs/clips/male/privet.json"
  isPlaying
  loopMode="repeat"      // или "once"
  playbackRate={1}       // 0.5 — вдвое медленнее
/>
```

Список слов для словаря берётся из `signs.json`:

```ts
const signs = await fetch("/tilmind-signs/signs.json").then((r) => r.json());
// [{ slug: "privet", word: "Привет", duration: 1.533, clips: { male: "clips/male/privet.json", female: "clips/female/privet.json" },
//    source: "signflow.ru" }, ...]
```

`TilMindSign` без обёртки ставится внутрь своего `<Canvas>`. Аватар стоит на `y = 0` в реальный рост (парень 1,76 м,
девушка 1,66 м). Так можно настроить свою камеру, например пресеты из `GLBViewer`.

Без React в чистом three.js:

```js
const gltf = await new GLTFLoader().loadAsync("tilmind-signs/avatars/tilmind-male.glb");
const clip = THREE.AnimationClip.parse(await (await fetch("tilmind-signs/clips/male/privet.json")).json());
const mixer = new THREE.AnimationMixer(gltf.scene);
mixer.clipAction(clip).play();   // и mixer.update(dt) в каждом кадре
```

## Как пересобрать

В проекте 3DAnimationGPT нужно запустить бэкенд на `:8010` и Vite на `:5173`, потом выполнить
`<python с playwright> tools/export_clips.py` (подробности в шапке скрипта). Новые слова добавляются через
`tools/signflow_batch.py <slug> ...` (signflow.ru) или `tools/slovo/` + `tools/slovo_batch.py` (Slovo).
