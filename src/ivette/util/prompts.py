"""Interactive console prompt helpers (shared by the CLI and downloaders)."""


def ask(prompt, default, cast=str):
    """Prompt until a value that casts cleanly, returning ``default`` on blank."""
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return cast(default)
        try:
            return cast(raw)
        except ValueError:
            print(f"    ! Expected {cast.__name__}, got: {raw!r}")


def ask_yn(prompt, default=True):
    """Prompt for a yes/no answer, returning ``default`` on blank input."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")
