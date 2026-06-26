"""Headless service layer — the reusable "brains" of Ivette.

Modules here orchestrate real work (running calculations, etc.) **without any
user-interface dependency**: no ``rich``, no ``questionary``, no ``ivette.cli``.
Interaction happens through plain callbacks passed in by the caller, so the same
service can be driven by the terminal menu today and by a web API / job server
later. Keeping this rule (no UI imports in ``ivette.services``) is what makes the
modular website-and-servers direction possible.
"""
