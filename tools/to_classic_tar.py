import argparse
import gzip
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def convert(src, dst, tag):
    tmp = Path(tempfile.mkdtemp(prefix="oci2docker-"))
    with tarfile.open(src) as archive:
        archive.extractall(tmp, filter="data")
    index = json.loads((tmp / "index.json").read_text(encoding="utf-8"))
    digest = index["manifests"][0]["digest"].split(":", 1)[1]
    manifest = json.loads((tmp / "blobs" / "sha256" / digest).read_text(encoding="utf-8"))
    config = manifest["config"]["digest"].split(":", 1)[1]
    layers = [(item["digest"].split(":", 1)[1], item["mediaType"]) for item in manifest["layers"]]
    out = tmp / "out"
    out.mkdir()
    shutil.copyfile(tmp / "blobs" / "sha256" / config, out / (config + ".json"))
    parent = None
    for layer, media_type in layers:
        target = out / layer
        target.mkdir()
        blob = tmp / "blobs" / "sha256" / layer
        opener = gzip.open if media_type.endswith("+gzip") else open
        with opener(blob, "rb") as source, open(target / "layer.tar", "wb") as plain:
            shutil.copyfileobj(source, plain)
        (target / "VERSION").write_text("1.0", encoding="utf-8")
        meta = {"id": layer}
        if parent:
            meta["parent"] = parent
        (target / "json").write_text(json.dumps(meta), encoding="utf-8")
        parent = layer
    name, version = tag.rsplit(":", 1)
    (out / "manifest.json").write_text(json.dumps([{
        "Config": config + ".json", "RepoTags": [tag],
        "Layers": [layer + "/layer.tar" for layer, _ in layers],
    }]), encoding="utf-8")
    (out / "repositories").write_text(json.dumps({name: {version: layers[-1][0]}}), encoding="utf-8")
    with tarfile.open(dst, "w") as archive:
        for path in sorted(out.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(out)))
    print("saved", dst, "layers", len(layers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--tag", default="aiplus-harness:0.2")
    args = parser.parse_args()
    convert(args.src, args.dst, args.tag)


if __name__ == "__main__":
    main()
