# edges/ — the sample edge library

This folder is a **sample library and the test-fixture corpus** for `apply_proportion_edge`, not the source of a real apply, and it is not packaged into the extension build: a real edge co-locates with the avatar it targets, in the Unity venue's `Blender/Avatars/<Family>/` bucket (`docs/LAYOUT.md`), and is copied there from here when one of these is the route in. The edge grammar is `avatarprep/core/proportions.py`'s `load_edge`; `docs/blender.md` §Edge filing owns naming and stamping.

**An edge with no `object`, `scales`, `moves` or `shapekeys` block is an identity edge — a pure base relabel.** It is the right shape when two vendor bases are the same mesh at the same scale and only the lineage name differs; it is not a stub awaiting ops.

| edge | kind | what it records |
|---|---|---|
| `plum-to-chiffon.json` / `chiffon-to-plum.json` | equivalency, uniform scale 0.9512 / 1.0513036 about the world origin | あまとうさぎ (Amatoususagi) ships Plum and Chiffon as one mesh, Plum uniformly larger |
| `chocolat-to-chiffon.json` / `chiffon-to-chocolat.json` | identity | the same vendor's Chocolat is the Chiffon body under another name |
| `eku-to-milfy.json` | identity | a second identity sample from another vendor family |
| `custom_chiffon.json` | same-base reshape, `unproportioned → custom` | a global scale and lift, seven per-bone local-space scales and a shape-key value |
| `custom_shinano.json` | same-base reshape, `unproportioned → custom` | a reshape on the vendor Shinano rig |

Amatoususagi's four bases — Chocolat, Chiffon, Plum, Lime — share one 素体, so the two equivalency pairs above chain: a Chocolat import reaches Plum through `chocolat-to-chiffon` then `chiffon-to-plum`, and a same-base reshape targeting Plum applies after.

**`custom` is a state name shared by two different bodies, and that is not an equivalence.** `custom_chiffon` and `custom_shinano` both land on state `custom` from different `source_base` values; the state string is base-neutral (`docs/blender.md` §State stamps), so equal state strings on two bases say nothing about shared geometry.
