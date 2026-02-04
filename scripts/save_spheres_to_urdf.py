import xmltodict
from pathlib import Path
import json
from typing import Any
import tyro

def _urdf_clean_filename(filename: str) -> str:
    if filename.startswith('package://'):
        filename = filename[len('package://'):]

    return filename

def load_urdf(urdf_path: Path):
    with open(urdf_path, 'r') as f:
        xml = xmltodict.parse(f.read())
        xml['robot']['@path'] = str(urdf_path)
        return xml

def save_urdf(urdf: dict[str, Any], filename: Path):
    with open(filename, 'w') as f:
        f.write(xmltodict.unparse(urdf, pretty = True))


def load_spheres(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

def set_urdf_spheres(urdf, spheres):
    total_spheres = 0
    for link in urdf['robot']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        collisions = link['collision']
        print(collisions)

        if not isinstance(collisions, list):
            collisions = [collisions]

        spherizations = []
        for i, collision in enumerate(collisions):

            geometry = collision['geometry']

            if 'box' in geometry or 'cylinder' in geometry or 'sphere' in geometry:
                key = f"{name}::primitive{i}"
                if key in spheres:
                    spherizations.append(spheres[key])

            elif 'mesh' in geometry:
                mesh = geometry['mesh']
                filename = _urdf_clean_filename(mesh['@filename'])
                key = f"{name}::{filename}"
                print(key)

                if name in spheres:
                    spherizations.append(spheres[name])

        collision = []
        for spherization in spherizations:
            print(spherization)
            for sphere_center, sphere_radius in zip(spherization["centers"], spherization["radii"]):
                total_spheres += 1
                collision.append(
                    {
                        'geometry': {
                            'sphere': {
                                '@radius': sphere_radius
                                }
                            },
                        'origin': {
                            '@xyz': ' '.join(map(str, sphere_center)), '@rpy': '0 0 0'
                            }
                        }
                    )

        link['collision'] = collision
    print(f"spheres: {total_spheres}")


if __name__ == "__main__":
    def main(input_urdf: Path, spheres_path: Path, output_urdf: Path) -> None:
        """Load `input_urdf`, inject spheres from `spheres_path`, and write `output_urdf`."""

        # Validate inputs
        if not input_urdf.exists() or not input_urdf.is_file():
            raise SystemExit(f"Input URDF not found: {input_urdf}")
        if not spheres_path.exists() or not spheres_path.is_file():
            raise SystemExit(f"Spheres JSON not found: {spheres_path}")

        urdf = load_urdf(input_urdf)
        spheres = load_spheres(spheres_path)
        set_urdf_spheres(urdf, spheres)

        # Ensure output directory exists
        output_urdf.parent.mkdir(parents=True, exist_ok=True)
        save_urdf(urdf, output_urdf)

    tyro.cli(main)
