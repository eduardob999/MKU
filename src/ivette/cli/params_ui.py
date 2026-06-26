"""Reusable "advanced options" editor for any parameter stage.

One interactive component, driven by :mod:`ivette.core.params` (the field list +
help) and :mod:`ivette.util.presets` (named, saved configurations). Every
submenu calls :func:`configure_stage` to let the user review/edit parameters and
load/save presets, instead of hand-writing prompts. Keeps the data layer
UI-free; this module is the only place the parameter UI lives.
"""

from __future__ import annotations

from dataclasses import replace

from ivette.cli import ui
from ivette.core import params as P
from ivette.util import presets


def _fmt(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _show_table(title, infos, preset_name):
    ui.table(
        [("Parameter", "accent"), ("Value", "white"), ("Description", "muted")],
        [(fi.name, _fmt(fi.value), fi.help) for fi in infos],
        title=f"{title}   ·   preset: {preset_name or 'defaults / custom'}",
    )


def _edit_field(params, fi):
    """Prompt for one field by kind; return a new params instance (or unchanged)."""
    label = f"{fi.name} — {fi.help}"
    if fi.kind == "bool":
        newval = ui.confirm(label, default=bool(fi.value))
    elif fi.kind == "int":
        newval = ui.ask_int(label, int(fi.value))
    elif fi.kind == "float":
        newval = ui.ask_float(label, float(fi.value))
    elif fi.kind in ("ints", "strs"):
        raw = ui.ask_text(f"{label} (space-separated)", _fmt(fi.value))
        toks = raw.split()
        if not toks:
            newval = list(fi.value)
        elif fi.kind == "ints":
            try:
                newval = [int(t) for t in toks]
            except ValueError:
                ui.warn("Expected whole numbers — left unchanged.")
                return params
        else:
            newval = toks
    elif fi.choices:
        choice = ui.select(label, [(c, c) for c in fi.choices])
        newval = fi.value if choice is ui.CANCEL else choice
    else:
        newval = ui.ask_text(label, str(fi.value))
    return replace(params, **{fi.name: newval})


def _pick_field(infos):
    choice = ui.select(
        "Edit which parameter?",
        [(f"{fi.name}  =  {_fmt(fi.value)}", fi) for fi in infos] + [("← Back", None)],
    )
    return None if choice is ui.CANCEL else choice


def _pick_preset(stage, prompt):
    names = presets.list_presets(stage)
    if not names:
        ui.note("No saved presets for this stage yet.")
        ui.pause()
        return None
    choice = ui.select(prompt, [(n, n) for n in names] + [("← Back", None)])
    return None if choice is ui.CANCEL else choice


def configure_stage(stage, base=None, *, title=None):
    """Interactively review/edit the parameters for ``stage`` and return them.

    ``stage`` is a key of :data:`ivette.core.params.STAGES`. Starts from ``base``
    (or the documented defaults), lets the user edit fields and load/save named
    presets, and returns the chosen dataclass instance. Cancelling keeps the
    current values, so a caller can always proceed.
    """
    stage_title, cls = P.STAGES[stage]
    title = title or stage_title
    params = base if base is not None else cls()
    preset_name = None

    while True:
        infos = P.describe(params)
        _show_table(title, infos, preset_name)

        has_presets = bool(presets.list_presets(stage))
        choices = [
            ui.section("Parameters"),
            ("✓ Use these values", "done"),
            ("⚙ Edit a parameter", "edit"),
            ui.section("Presets"),
        ]
        if has_presets:
            choices.append(("📂 Load preset", "load"))
        choices.append(("💾 Save current as preset", "save"))
        if has_presets:
            choices.append(("🗑 Delete a preset", "delete"))
        choices += [
            ui.section(""),
            ("↺ Reset to defaults", "reset"),
            ("✕ Cancel (use current values)", "done"),
        ]

        action = ui.select(f"Configure — {title}", choices)
        action = "done" if action is ui.CANCEL else action

        if action == "done":
            return params
        if action == "edit":
            fi = _pick_field(infos)
            if fi is not None:
                params = _edit_field(params, fi)
        elif action == "load":
            name = _pick_preset(stage, "Load which preset?")
            if name:
                params = P.from_dict(cls, presets.load_preset(stage, name))
                preset_name = name
        elif action == "save":
            name = ui.ask_text("Preset name (e.g. fast, accurate, production)")
            if name:
                presets.save_preset(stage, name, P.to_dict(params))
                preset_name = name
                ui.success(f"Saved preset '{name}'.")
                ui.pause()
        elif action == "delete":
            name = _pick_preset(stage, "Delete which preset?")
            if name and presets.delete_preset(stage, name):
                if preset_name == name:
                    preset_name = None
                ui.success(f"Deleted preset '{name}'.")
                ui.pause()
        elif action == "reset":
            params = cls()
            preset_name = None
