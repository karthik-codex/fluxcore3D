_ALL_FACES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

def resolve_face_roles(cfg) -> dict:
    flow_dir   = cfg.flow_dir.upper().strip()
    sign       = +1 if flow_dir[0] == "+" else -1
    axis_char  = flow_dir[1]

    inlet_face  = ("-" if sign == +1 else "+") + axis_char
    outlet_face = ("+" if sign == +1 else "-") + axis_char

    user_walls   = {s.upper().strip() for s in getattr(cfg, 'domain_walls',   [])}
    user_outlets = {s.upper().strip() for s in getattr(cfg, 'domain_outlets', [])}
    legacy_walls = bool(getattr(cfg, 'transverse_walls', False))

    roles = {}
    for face in _ALL_FACES:
        if face == inlet_face:
            roles[face] = 'inlet'
        elif face in user_walls:      # ← user wall override comes BEFORE outlet
            roles[face] = 'wall'
        elif face == outlet_face:
            roles[face] = 'outlet'
        elif face in user_outlets:
            roles[face] = 'outlet'
        elif legacy_walls:
            roles[face] = 'wall'
        else:
            roles[face] = 'periodic'

    for face in user_walls:
        if face == inlet_face:
            import warnings
            warnings.warn(
                f"resolve_face_roles: '{face}' is the inlet face — "
                "cannot be walled. Override ignored.")

    for face in user_outlets:
        if face == inlet_face:
            import warnings
            warnings.warn(
                f"resolve_face_roles: '{face}' is the inlet face — "
                "cannot be set as outlet. Override ignored.")

    return roles


def _face_ax_idx_fsign(cfg_or_sim, face: str):
    ax = {"X": 0, "Y": 1, "Z": 2}[face[1]]
    n  = (cfg_or_sim.nx, cfg_or_sim.ny, cfg_or_sim.nz)[ax]
    if face[0] == "+":
        return ax, n - 1, +1
    else:
        return ax, 0, -1