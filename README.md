[![Views](https://hits.sh/github.com/NightHawkHSI/UniView-Plugins.svg?label=views&color=4c1)](https://hits.sh/github.com/NightHawkHSI/UniView-Plugins/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/NightHawkHSI/UniView#run-from-source)
[![Stars](https://img.shields.io/github/stars/NightHawkHSI/UniView-Plugins?style=flat&color=yellow)](https://github.com/NightHawkHSI/UniView-Plugins/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/NightHawkHSI/UniView-Plugins)](https://github.com/NightHawkHSI/UniView-Plugins/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/NightHawkHSI/UniView-Plugins)](https://github.com/NightHawkHSI/UniView-Plugins)

# UniView Plugins

Engine and game plugins for [UniView](https://github.com/NightHawkHSI/UniView), the game asset viewer.

UniView itself reads Unity games and loose files. Everything else - Source, Source 2, Unreal, Terraria and the
other games below - comes from a plugin in this repository. Plugins live here, in their own repository, and are
**not affiliated with any game developer or publisher** (see [DISCLAIMER.md](DISCLAIMER.md)).

## Installing a plugin

**From UniView (easiest):** open the **Plugins** tab, pick a plugin and click **Install**. UniView downloads it from
this repository (the list comes from [`plugins.json`](plugins.json)), checks each file's SHA-256, and asks you to
restart. The same tab shows which version you have and offers updates.

**By hand:** download the plugin's `.py` file (and the helper folder it needs, see the table) and put it in the
`plugins` folder next to `UniView.exe` (or next to `unity_viewer.py` when running from source). Restart UniView.
**Options -> Plugin manager** shows what loaded. You can also drop a `.py` or `.zip` on UniView's window.

When you add a game UniView has no plugin for, but one exists here, UniView tells you which plugin to install.

## Plugins

| Plugin | File | Games | Needs |
|---|---|---|---|
| Source | `source.py` | Team Fortress 2, Half-Life 2, Portal, Counter-Strike: Source, Left 4 Dead 2, Garry's Mod | `valve_formats/` |
| Source 2 | `source2.py` | Counter-Strike 2, Dota 2, Deadlock, Half-Life: Alyx | `valve_formats/` |
| Unreal Engine 4/5 | `unreal.py` | Satisfactory, Headliners, Dead as Disco and other UE4/5 games | `unreal_formats/` |
| XNA / FNA | `terraria.py` | Terraria (also Stardew Valley, Celeste textures/sounds) | |
| Clyde (Spiral Knights) | `spiralknights.py` | Spiral Knights | |
| Wakfu | `wakfu.py` | Wakfu | |
| Spore | `spore.py` | Spore | |
| Fallout 1/2 | `fallout.py` | Fallout, Fallout 2 | |
| Fallout Tactics | `FalloutTac.py` | Fallout Tactics: Brotherhood of Steel | |
| Risk of Rain | `Ror.py` | Risk of Rain (2013) | |
| Serious Sam 2 | `SeriousSam2.py` | Serious Sam 2 | |

Versions, descriptions and file checksums are in [`plugins.json`](plugins.json).

## Writing a plugin

Copy [`_template.py`](_template.py) - a working "loose files" plugin - and read UniView's
[PLUGINS.md](https://github.com/NightHawkHSI/UniView/blob/main/PLUGINS.md) for the API. Rules for plugins in this
repository:

- Only read game files. Write nothing except caches in `engines.sdk.cache_dir()`. No network access, no
  `subprocess`, no `eval`/`exec`, no downloading or running other code.
- No game assets, sample files, screenshots, code copied from a game, or encryption keys. Tests build their own
  tiny fake files (see `tests/`).
- Don't strip copy protection or de-obfuscate data a game deliberately hides.
- After changing a plugin, bump its `version` and run `py _make_index.py` to update `plugins.json`.

Run the tests against a UniView checkout (this folder inside it, or `UNIVIEW_ROOT` pointing at one):

    py -m pytest tests -q

## License

GNU GPL v3 ([LICENSE](LICENSE)), like UniView. Code ported from other projects keeps its notices
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
