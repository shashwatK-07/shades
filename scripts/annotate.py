"""Click reference landmarks and trace a shadow edge on a photo.

    python scripts/annotate.py --image ~/photos/0001.jpg --id 0001

Two phases, driven from the keyboard:

  phase 1  click each named landmark as the title prompts for it
  phase 2  press 'n', then click along the observed shadow edge
           'u' undo last click   'w' write JSON   'q' quit without writing

Writes validation/annotations/<id>.json. The image itself stays out of git --
only your clicks and the manifest URL are versioned.

IMPORTANT: the landmark world coordinates must be in the SAME frame as the
stadium YAML (x=east, y=north, z=up, metres). The built-in diamond below is
written with home plate at the origin and +y pointing at second base, which
is NOT automatically the frame your stadium file uses. Pass --landmarks with
your own file, or rotate/translate these, if the frames differ. Getting this
wrong silently rotates every shadow point you annotate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Standard MLB diamond, metres, home plate at origin, +y toward second base.
# Base paths are 90 ft (27.432 m); the rubber is 60 ft 6 in (18.44 m) from
# the plate. Foul poles are deliberately absent -- outfield distances are
# park-specific, so put those in your own --landmarks file.
DEFAULT_LANDMARKS = {
    "home_plate": [0.0, 0.0, 0.0],
    "first_base": [19.398, 19.398, 0.0],
    "second_base": [0.0, 38.795, 0.0],
    "third_base": [-19.398, 19.398, 0.0],
    "pitchers_rubber": [0.0, 18.44, 0.0],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--id", required=True, help="annotation id, e.g. 0001")
    parser.add_argument("--landmarks", help="JSON {name: [x, y, z]}; defaults to an MLB diamond")
    parser.add_argument("--out", default="validation/annotations")
    args = parser.parse_args()

    landmarks = (
        json.loads(Path(args.landmarks).read_text())
        if args.landmarks
        else DEFAULT_LANDMARKS
    )
    names = list(landmarks)

    state = {"phase": "refs", "refs": [], "shadow": []}

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(plt.imread(args.image))
    ax.set_axis_off()

    def title() -> None:
        if state["phase"] == "refs":
            i = len(state["refs"])
            if i < len(names):
                msg = f"click landmark: {names[i]}  ({i + 1}/{len(names)})"
            else:
                msg = "all landmarks clicked -- press 'n' to trace the shadow"
        else:
            msg = f"tracing shadow edge: {len(state['shadow'])} points"
        ax.set_title(f"{msg}\n[u]ndo  [n]ext phase  [w]rite  [q]uit")
        fig.canvas.draw_idle()

    def redraw() -> None:
        for artist in list(ax.lines) + list(ax.collections):
            artist.remove()
        if state["refs"]:
            xs = [r["px"] for r in state["refs"]]
            ys = [r["py"] for r in state["refs"]]
            ax.scatter(xs, ys, c="red", s=60, marker="x", zorder=5)
        if state["shadow"]:
            xs = [p[0] for p in state["shadow"]]
            ys = [p[1] for p in state["shadow"]]
            ax.plot(xs, ys, "-o", c="yellow", ms=4, lw=1.5, zorder=5)
        title()

    def on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            return
        if state["phase"] == "refs":
            i = len(state["refs"])
            if i >= len(names):
                return
            state["refs"].append(
                {"px": float(event.xdata), "py": float(event.ydata),
                 "name": names[i], "world": landmarks[names[i]]}
            )
        else:
            state["shadow"].append([float(event.xdata), float(event.ydata)])
        redraw()

    def on_key(event) -> None:
        if event.key == "n":
            state["phase"] = "shadow"
        elif event.key == "u":
            bucket = "refs" if state["phase"] == "refs" else "shadow"
            if state[bucket]:
                state[bucket].pop()
        elif event.key == "w":
            write()
        elif event.key == "q":
            plt.close(fig)
            return
        redraw()

    def write() -> None:
        if len(state["refs"]) < 4:
            print(f"need at least 4 landmarks for a homography, have {len(state['refs'])}")
            return
        if len(state["shadow"]) < 2:
            print("trace at least 2 shadow points first")
            return
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.id}.json"
        path.write_text(
            json.dumps(
                {"id": args.id, "refs": state["refs"], "shadow_line_px": state["shadow"]},
                indent=2,
            )
        )
        print(f"wrote {path} ({len(state['refs'])} refs, {len(state['shadow'])} shadow points)")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
