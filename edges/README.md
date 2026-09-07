# edges/ — the sample edge library

Primary reader: an agent about to apply or author a proportion edge. This folder is a **sample library and the test-fixture corpus** for `apply_proportion_edge`, not the source of a real apply: a real edge co-locates with the avatar it targets, in the Unity venue's `Blender/Avatars/<Family>/` bucket (`docs/LAYOUT.md`), and is copied there from here when one of these is the route in. The edge grammar is `avatarprep/core/proportions.py`'s `load_edge`; `docs/blender.md` §Edge filing owns naming and stamping.

**An edge with no `object`, `scales` or `shapekeys` block is an identity edge — a pure base relabel.** `load_edge` defaults every op block to empty, `--whatif` reports every stage at zero delta, and the apply writes only the stamps: `avatarprep_base` moves to `target_base`, state to `target`. It is the right shape when two vendor bases are the same mesh at the same scale and only the lineage name differs; it is not a stub awaiting ops.

| edge | kind | what it records |
|---|---|---|
| `plum-to-chiffon.json` / `chiffon-to-plum.json` | equivalency, uniform scale 0.9512 / 1.0513036 about the world origin | あまとうさぎ (Amatoususagi) ships Plum and Chiffon as one mesh, Plum uniformly larger |
| `chocolat-to-chiffon.json` / `chiffon-to-chocolat.json` | identity | the same vendor's Chocolat is the Chiffon body under another name |
| `eku-to-milfy.json` | identity | a second identity sample from another vendor family |
| `custom_chiffon.json` | same-base reshape, `unproportioned → custom` | the legacy "Custom Plum" second stage: 1.115 global, +0.018 lift, seven per-bone normal-space scales, `Breasts_small 0.6` |
| `custom_shinano.json` | same-base reshape, `unproportioned → custom` | the legacy Shinano reshape on the vendor Shinano rig |

Amatoususagi's four bases — Chocolat, Chiffon, Plum, Lime — share one 素体, so the two equivalency pairs above chain: a Chocolat import reaches Plum through `chocolat-to-chiffon` then `chiffon-to-plum`, and a same-base reshape targeting Plum applies after. Lime has no edge here yet.

**`custom` is a state name shared by two different bodies, and that is not an equivalence.** `custom_chiffon` and `custom_shinano` both land on state `custom` from different `source_base` values; the state string is base-neutral by design (`docs/blender.md` §State stamps), so equal state strings on two bases say nothing about shared geometry. The stamp gates compare `base` and `state` together for exactly this reason.
