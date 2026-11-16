# Avatars

**Current (Tripo):** the player's signers are generated with the Tripo API (v3) from the reference sheet:

1. `tools/tripo_avatar.py restyle` — the reference photo is redrawn by Tripo's image model into a stylized
   (Ready Player Me-like) character in A-pose with spread fingers (`models/tripo/<key>_prompt.txt`), ~10 credits;
2. `tools/tripo_avatar.py model` — image-to-model v3.1, detailed texture, ~40 credits;
3. `tools/tripo_avatar.py rig` — rig-check (free) + auto-rig v3.0, Mixamo skeleton with fingers, 25 credits;
4. `blender --background --python tools/build_tripo_avatars.py -- --out frontend/public/models` (no credits): the face
   and the chest print are re-projected from the reference image (registered to Tripo's own texture), the mirrored
   back print is painted out, the print's relief is flattened in the normal map, bones lose the `mixamorig:` prefix,
   `LeftEye`/`RightEye` bones are added from MediaPipe face landmarks on the reference
   (`models/tripo/ref_landmarks.json`), the 52 ARKit shapes are transferred from MakeHuman faceunits01, and the figure is
   stood up at real height facing the camera.

   The face is made animatable (`tools/tripo_face.py`): Tripo paints eyes and lips onto the skin, so the face is
   subdivided, the eye openings are cut out, and the lips are split along the line between them.

   The eyes are built as eyes (`tools/tripo_eyes.py`, 2026-11-12; before, the ball behind each opening carried the
   reference's painted eye projected from the front, lids, liner and lashes included, which turned with the gaze, and
   the user saw "just a picture, like a mask over the face"): a white, pinkish and darker toward the back, an iris of
   the reference's colour (sampled there) with radial fibres, a lighter ring round the pupil and a dark ring at its
   edge, at least 0.52 of the opening's height across (MediaPipe reads these cartoon irises at about half), a black
   pupil, under a clear-coated cornea that bulges a little in front, whose highlights make the eye look wet; skinned
   to `LeftEye`/`RightEye`. The texture is laid out from the pupil outward (to 100° round the ball): mapped round a
   pole at the pupil, it was squeezed to a point there, the GPU took a blur of iris and white from afar, and the
   irises looked grey and the eyes frightening. The opening's rim is spread evenly along smooth curves through
   MediaPipe's lid points (snapped vertex by vertex onto the straight lines between them, it was a polygon, its margin
   a frame with corners, and where the cut edge zigzagged the rim doubled back and the lid's inside stood out in pale
   flaps), and the skin round it is smoothed in depth (its creases caught the light as cracks under the eyes). The
   ball is tucked behind the skin wherever the skin covers it, facing any way forward and for small turns of the eye,
   the dents spread and smoothed over its hidden part, what shows through the opening at rest left round (tucked only
   at rest and only behind skin facing the front, the ball showed through by the outer corners, which face the side).
   The lids get their thickness, their edge turning in to a pink margin and down onto the ball (onto the ball as
   tucked)
   with a dark line where they meet it (kept on the ball through every lid shape), and single thin lashes stand on
   the upper lid (longer and flicking out at the outer corner for the girl, short for the boy; none on the lower lid,
   where they read as a row of teeth), closing with it: skinned like the lid they stand on (made without skin weights, they stayed where
   the head had been when it moved), and matt (glossy, their thin faces caught the light in white specks). The lids' inside and the lashes are painted into free squares of the body's texture,
   like the lips' inside. The shapes that matter most are built
   from that geometry instead of transferred: blink (upper lid over the eyeball), brows down / inner up / outer up.

   The mouth is built as a mouth (`tools/tripo_mouth.py`, 2026-11-11; before it the lips moved like a mask over the
   face: paint on a flat surface around a dark plug, the lower lip sliding over a chin that hardly moved): the lips get
   the volume of real lips from MediaPipe's outer lip contour (each lip's red stands out, the line between them lies
   behind, the fold under the lower lip deepens) and an inner side, three rings turned in from the opening's edge and
   coloured from the painted lip, so a parted mouth shows their thickness; their red is wet (low roughness in the ORM
   map). `jawOpen` turns the whole mandible about the jaw joints (at the height of the bottom of the nose, 0.3 of the
   way from the head bone to the lips) and slides it forward a little, as the condyles do, with the chin, the lower
   lip, the lower teeth and the tongue, the cheeks and the skin under the chin stretching between; at 1 the lower lip
   drops 0.58 of the mouth's width (23–25°). `mouthUpperUp`/`mouthLowerDown` lift the lips off the teeth (the
   transferred ones barely moved the painted lips). `mouthPucker` (у, ю) and `mouthFunnel` (о, ш) round the lips into a tube (2026-11-12, after "буква о страшно…
   будто просто сжали, а не губки трубочкой"): the mouth narrows evenly from corner to corner; in its middle the lips'
   edges part round a pointed-oval opening, beyond it they stay pressed together out to the corners; each lip's
   cross-section is laid on a tube along its edge (the edge on the tube's inner front, the red rolled over its front,
   less where the lip is thin by the corners, the inner side going on round behind), the upper lip arching over the
   opening. The opening's height is the jaw's: `mouthFunnel` is made for `jawOpen` 0.38 (beyond the opening it keeps
   the lips together against what that jaw does), `mouthPucker` for the jaw shut, a small hole the lips make
   themselves. The skin round the lips is carried with them as a membrane that also resists bending (conjugate
   gradients over the mesh's places, so the two sides of a texture seam move as one), bent with the jaw opened as the
   shape is made for, then smoothed a little where it is drawn in (the rest shape's bumps squeezed into ridges). The
   lips' outlines are refitted as smooth curves first (between MediaPipe's points they are straight, and the lips
   creased at every bend). Earlier tries: the lips' red ringed round the opening as thick as it is (the opening's sides
   were skin folded in: "squeezed"); laid round a whole ring (the corners were drawn into the opening's sides and the
   skin round each closed in a wedge). The mouth's cavity is kept behind the skin as these shapes move it (following
   the nearest skin, it was left behind by the corners and showed through the cheeks in black flaps). Teeth stand in arches of a person's proportions (incisors, canine,
   premolars and first molar a side, scaled to the mouth's width, the upper incisors over and before the lower) in
   scalloped gums (`TM_teeth_*`, `TM_gums_*`); the tongue (`TM_tongue`) lies in the lower arch with shapes of its own,
   `tongueTipUp` (tip on the ridge behind the upper teeth: т, д, н, л, р), `tongueBackUp` (к, г, х), `tongueFrontUp`
   (и, й) and `tongueOut`; the cavity (`TM_mouth_cavity`) is lined inside, its floor going with the jaw and its front
   with the skin. Everything inside is kept behind the skin in front of it and shaded darker deeper in through vertex
   colours (the player has no ambient occlusion). The shapes carry normal offsets, so the light follows the lips; the
   exporter writes those for every vertex, and the build rewrites them as sparse (46 MB → 11–12 MB a file).
   The player has no ambient occlusion, so the teeth, gums, tongue and cavity are drawn darker the smaller the mouth's
   opening (`face-expression.ts`, reckoned from the jaw and the lips' rounding and spread): behind the small hole of
   "у" the incisors shone white as in a wide-open "а".

   The boy's hair is cut anew (`tools/tripo_hair.py`, `STYLE = "short"`, 2026-11-17: "обычный короткий полубокс"; Tripo
   gave him a long cut's volume and a fringe that was a solid ledge over the forehead down to the brows, its surface
   crumpled under it, and every repair of that surface showed in blotches and creases). A skull is made under it: a
   sphere in a head's proportions fitted to Tripo's hair, brought in by the hair's thickness, takes the hair and the
   forehead above the brows (eased in from the face), its vertices spread over it so the fringe's folded layers no
   longer flicker. Over it lies the haircut (`TM_hair`), a shell on a fine icosphere round the head (each direction
   taken out to the hair, through the hood's collar behind but never through the face), a few millimetres thick on
   top, close-cropped at the sides and the back, where its texture shows the scalp through clipped hair, cut level at
   the nape. The shell goes on down over the forehead and the brows' top, lying flat on them, painted there as skin:
   the reference itself under the brows' top, its skin carried up above, darker toward the hairline (its shadow), so
   the crumpled surface under it never shows and the shell's edge hides in the brows. (Before: broad locks with a
   pointed fringe, "в целом плохая прическа"; a smooth shell, "a bowl cut"; about 750 lock tubes, "кучу … в голове";
   Tripo's cap repainted, a brain or a helmet; MakeHuman hair, rejected. The locks remain as `STYLE = "locks"`.)
   Where Tripo's hair meets the skin (both avatars) its surface steps in ragged spikes; the vertices a few rings either
   side of the hairline, above the brows, are smoothed (Taubin), and over the girl's forehead the step is flattened
   out. The face's texture above the part taken from the reference (both avatars) is repainted with the reference's skin
   just under that cut, carried up column by column (one flat colour with Tripo's shading kept read as a blotchy band),
   going over into the hair's own colour at the hair; on the girl the reference itself is then laid over the forehead
   and its hairline (a clean arc and the parting), the hair's upturned edge darkened (it caught the light from above in
   a light ring round the forehead); the light wisps Tripo painted into the girl's hair's edge are darkened to the hair; the
   texture's islands round the head are padded with their own edge colours (painted over, they no longer matched
   Tripo's padding, and the seams drew thin dark lines). Under the eyes the reference's painted white and lower lid,
   outside MediaPipe's contour, are filled in with skin before it is laid on, and round the eyes Tripo's own painted eye
   is not spared as "hair" (both showed as grey smudges under the 3D lids). The body's colour is exported at 2048 (the
   face was 200 texels across at 1024). The hood's shadow Tripo painted on the boy's neck is painted out, and the hood's
   inside round the neck (painted dark and spotted like hair) is the hoodie's colour in shadow.

   The chest print is a decal of its own (`TM_print`): a copy of the hoodie's surface under the letters, lifted 1.5 mm
   and skinned like it, textured with the team's wordmark "Til Mind" (`assets/logo/til-mind-text.png`, made by
   `tools/make_logo.py --out assets/logo/til-mind-text.png --align left`) in the avatar's ink colour (white on the
   purple hoodie, deep berry on the pink one), fitted into 85% of the box the reference's lettering took (toward the
   box's far end the chest turns to the sleeve seam and letters there are cut off). The print Tripo painted, its mirrored
   copy on the back, and their relief in the normal and occlusion maps are painted out, 4 texels past the triangles'
   edges so no letters are left in the UV seams. (Until 2026-11-03 the print read "Visual Mind" and the files were
   `visualmind-*.glb`; the team is called Til Mind.)

   The player's files are `frontend/public/models/tilmind-male.glb` and `tilmind-female.glb`; meshes, materials and
   images in them are named `TM_*`.

The API key lives in `~/.config/visualmind/tripo.env` (`TRIPO_API_KEY`), never in the repo. Tripo outputs and task
JSONs are kept in `models/tripo/`. Earlier avatars: `frontend/public/models/legacy-rpm/`, `previous-stylized/`.
Candidates from the web are compared at `/candidates.html` (files in `frontend/public/candidates/`).

## Earlier approaches (kept for reference)

**Clothes and hair** come from MakeHuman community packs and are carried onto the Quaternius body by
`tools/mpfb_outfit.py` (worn by a MakeHuman stand-in posed like the target, warped joint-to-joint / head-to-head,
pushed outside the skin, re-skinned with the target's weights; sleeves cut at the wrist). They are CC-BY, so the
site must credit them:

- Hooded sweat jacket `elvs_hooded_sweat_jacket1`, jeans `elvs_jeans_straight_leg`, hair `elvs_daisy_hair` and
  `elvs_grump_hair` — Elvaerwyn (MakeHuman community, CC-BY 4.0)
- Jeans `punkduck_male_classic_jeans` — punkduck (MakeHuman community, CC-BY 4.0)
- Sneakers `culturalibre_sneakers` — culturalibre (MakeHuman community, CC-BY 4.0)
- Base characters, face — Quaternius, *Universal Base Characters* (CC0)

Packs: shirts02, pants02, shoes02, hair02 from static.makehumancommunity.org/assets/assetpacks.html, unzipped into the
MPFB user data folder.

**Current (stylized):** `tools/build_stylized_avatars.py` builds the player's avatars from Quaternius' *Universal Base
Characters* (CC0, free "Standard" tier: Superhero proportions only; the Regular/Teen bodies are in the $19.99 Source
tier). The pack has no clothes and no facial blendshapes, so the script adds both: clothes are separate meshes cut from
the body's surface and reshaped (hoodie hanging loose from the chest with ribbed cuffs/waistband, hood and drawstrings,
straight jeans, sneakers with soles; purple/pink with the "Visual Mind" print after the reference renders in
`assets/avatars/`), skinned with the body's weights, with the skin under them removed. Hands are scaled to natural size
(the superhero base's are ~10% large). And the 52 ARKit blendshapes are transferred from MakeHuman's faceunits01
through a landmark warp (eyes, mouth corners and line, nose, chin; lips matched upper/lower, lids upper). Pack in
`~/.local/share/visualmind/quaternius/ubc`.

    blender --background --python tools/build_stylized_avatars.py -- --out frontend/public/models [--only male]

The MakeHuman build below is kept as an alternative (the user found its look unappealing).

## MakeHuman build (alternative)

The player's two signers (`frontend/public/models/visualmind-male.glb`, `visualmind-female.glb`) are generated by
`tools/build_mpfb_avatars.py` with MPFB 2 (the MakeHuman add-on for Blender). Every asset used is CC0.

## What the GLBs contain

- Mixamo-named skeleton the solvers expect (`Hips` … `RightHandPinky4`), plus tip bones after each last phalanx and
  after the head, and `LeftEye`/`RightEye` bones the eyeballs are skinned to (for gaze).
- 52 ARKit blendshapes (`jawOpen`, `browInnerUp`, `eyeBlinkLeft`, …) on the body, carried over to brows, lashes,
  eyes, teeth, tongue and hair. These are the names MediaPipe's face landmarker outputs, so expression data maps 1:1.
- Body skin hidden under the clothes is removed, so it cannot poke through when the arms move.
- The Visual Mind emblem (`assets/logo/visual-mind-emblem.png`, from the site's mark) as its own small mesh on the chest.
- Textures as WebP; about 7–8 MB per avatar.

The solvers measure each avatar when it loads (`frontend/src/body-shape.ts`: torso from the trunk's skin, chin from
the head's skin), so a new or changed avatar needs no hand-tuned constants.

## Rebuilding

One-time setup: install the MPFB extension into Blender (`blender --command extension install-file -r user_default
--enable add-on-mpfb-v2.0.18.zip`) and unzip the asset packs into
`~/.config/blender/5.2/extensions/.user/user_default/mpfb/data`: `makehuman_system_assets_cc0`, `faceunits01`,
`skins01_cc0`, `skins02_cc0`, `hair01_cc0`, `eyebrows01_cc0`, `eyelashes01_cc0`, `shirts01_cc0`, `pants01_cc0`,
`shoes01_cc0`, `system_hair_materials01_cc0` (from static.makehumancommunity.org). Downloads are kept in
`~/.local/share/visualmind/mpfb/downloads`.

    blender --background --python tools/build_mpfb_avatars.py -- --out frontend/public/models [--only male] [--blend]

Look (body shape, skin, hair, brows, clothes, top colour) is set in `CHARACTERS` and `TOP_COLOR` at the top of the
script. The previous Ready Player Me avatars are kept in `frontend/public/models/legacy-rpm/`.
