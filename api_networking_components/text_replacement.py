"""Replace supplied text boxes and keep a verified ZIP of the rendered images."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid
import zipfile

from PIL import Image, ImageDraw, ImageFont


def _wrap(text, draw, font, width):
    lines=[]
    for paragraph in text.split("\n"):
        line=""
        for token in re.findall(r"\S+|[^\S\n]+", paragraph):
            candidate=line+token
            if draw.textlength(candidate, font=font)<=width:
                line=candidate
                continue
            if line.strip():
                lines.append(line.rstrip())
                line=""
            token=token.lstrip()
            if font.size>8 and re.search(r"[A-Za-z]", token) and draw.textlength(token, font=font)>width:
                # Prefer a smaller font over arbitrary breaks inside English words.
                line=token
                continue
            for character in token:
                if line and draw.textlength(line+character, font=font)>width:
                    lines.append(line)
                    line=""
                line+=character
        lines.append(line.rstrip())
    return "\n".join(lines)


def _render_region(image, region, font_path):
    if not isinstance(region, dict):
        raise ValueError("Each region must be an object.")
    box=region.get("box")
    if not isinstance(box, (list, tuple)) or len(box)!=4 or any(type(n) is not int for n in box):
        raise ValueError("box must contain four integer pixel coordinates.")
    left, top, right, bottom=box
    if not (0<=left<right<=image.width and 0<=top<bottom<=image.height):
        raise ValueError("Text box is outside the image or has no area.")
    text=region.get("text")
    if not isinstance(text, str):
        raise ValueError("Each region needs translated text as a string.")
    layout=region.get("layout_box", box)
    if (not isinstance(layout, (list, tuple)) or len(layout)!=4
            or any(type(n) is not int for n in layout)
            or not (0<=layout[0]<=left<right<=layout[2]<=image.width
                    and 0<=layout[1]<=top<bottom<=layout[3]<=image.height)):
        raise ValueError("layout_box must be in the image and contain the wipe box.")
    width, height=right-left, bottom-top
    background=region.get("background", "white")
    if isinstance(background, list):
        background=tuple(background)
    color=region.get("color", "black")
    if isinstance(color, list):
        color=tuple(color)
    wipe_rects=region.get("wipe_rects")
    if wipe_rects is not None:
        if not isinstance(wipe_rects, list) or not wipe_rects:
            raise ValueError("wipe_rects must be a nonempty list.")
        for rect in wipe_rects:
            if (not isinstance(rect, (list, tuple)) or len(rect)!=4
                    or any(type(n) is not int for n in rect)
                    or not (left<=rect[0]<rect[2]<=right and top<=rect[1]<rect[3]<=bottom)):
                raise ValueError("Each wipe rectangle must be inside the wipe box.")
        for x1, y1, x2, y2 in wipe_rects:
            ImageDraw.Draw(image).rectangle((x1, y1, x2-1, y2-1), fill=background)
    elif layout!=box:
        # Clear only the identified source ink; retain surrounding bubble pixels.
        ImageDraw.Draw(image).rectangle((left, top, right-1, bottom-1), fill=background)
    left, top, right, bottom=layout
    width, height=right-left, bottom-top
    canvas=image.crop(layout) if wipe_rects is not None or layout!=box else Image.new("RGBA", (width, height), background)
    with canvas as patch:
        draw=ImageDraw.Draw(patch)
        if text.strip():
            padding=2
            maximum=region.get("max_font_size", 64)
            if type(maximum) is not int or not 8<=maximum<=64:
                raise ValueError("max_font_size must be an integer from 8 to 64.")
            for size in range(min(maximum, height), 7, -1):
                font=ImageFont.truetype(str(font_path), size)
                wrapped=_wrap(text, draw, font, width-2*padding)
                bounds=draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=2, align="center")
                text_width, text_height=bounds[2]-bounds[0], bounds[3]-bounds[1]
                if text_width<=width-2*padding and text_height<=height-2*padding:
                    position=((width-text_width)/2-bounds[0], (height-text_height)/2-bounds[1])
                    draw.multiline_text(position, wrapped, font=font,
                                        fill=color, spacing=2, align="center")
                    break
            else:
                raise ValueError(f"Translated text does not fit at 8 pixels: region {region.get('id', 'manual')}, layout {width}x{height}, {len(text)} characters.")
        image.paste(patch, (left, top))


def replace_text(image_path, regions, output_path, *, font_path):
    """Write a PNG copy with replaced boxes. Coordinates use stored image pixels.

    Boxes use [left, top, right, bottom], with exclusive right and bottom edges.
    Regions are applied in order. The caller must supply a font for the language.
    """
    if not isinstance(regions, list):
        raise ValueError("regions must be a list; use [] for an unchanged image.")
    if Path(image_path).resolve()==Path(output_path).resolve():
        raise ValueError("Output must not replace a source image.")
    with Image.open(image_path) as source:
        if getattr(source, "n_frames", 1)!=1:
            raise ValueError("Only single-frame images are supported.")
        with source.convert("RGBA") as rendered:
            for region in regions:
                _render_region(rendered, region, font_path)
            with Path(output_path).open("xb") as destination:
                rendered.save(destination, format="PNG")


def _digest(stream):
    digest=hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024*1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _publish_archive(source, output):
    """Copy into the output directory to inherit its ACL before exclusive publication."""
    publication=output.parent/f".publish-{uuid.uuid4().hex}.zip"
    try:
        with source.open("rb") as incoming, publication.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        with source.open("rb") as original, publication.open("rb") as copied:
            if _digest(original)!=_digest(copied):
                raise RuntimeError("Publication copy failed hash verification.")
        os.link(publication, output)
    finally:
        if publication.exists():
            publication.unlink()


def replace_and_zip(images, output_zip, *, font_path, batch_size=50):
    """Render batches into one temporary folder, verify a ZIP, then delete the folder.

    Each image object needs path and regions; optional labels are kept in the manifest.
    Existing outputs and input images are never overwritten. On failure, staging
    files are kept and their path is included in the error for recovery.
    """
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a nonempty list.")
    if type(batch_size) is not int or not 1<=batch_size<=50:
        raise ValueError("batch_size must be an integer between 1 and 50.")
    font_path=Path(font_path).resolve(strict=True)
    ImageFont.truetype(str(font_path), 8)
    output=Path(output_zip).absolute()
    if output.suffix.lower()!=".zip":
        raise ValueError("Output must have a .zip extension.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output=output.parent.resolve()/output.name
    staging=Path(tempfile.mkdtemp(prefix=".text-replacement-", dir=output.parent)).resolve()
    try:
        records=[]
        for start in range(0, len(images), batch_size):
            for index, entry in enumerate(images[start:start+batch_size], start=start+1):
                if not isinstance(entry, dict) or "path" not in entry or "regions" not in entry:
                    raise ValueError("Each image needs path and regions fields.")
                source=Path(entry["path"])
                safe_stem=re.sub(r"[^\w.-]", "_", source.stem)[:80] or "image"
                name=f"{index:06d}_{safe_stem}.png"
                replace_text(source, entry["regions"], staging/name, font_path=font_path)
                records.append({**entry, "path": str(source), "output": name})
        (staging/"manifest.json").write_text(
            json.dumps({"images": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        files=sorted(staging.iterdir())
        completed_zip=staging/"result.zip"
        with zipfile.ZipFile(completed_zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=path.name)
        with zipfile.ZipFile(completed_zip) as archive:
            if sorted(archive.namelist())!=[path.name for path in files] or archive.testzip() is not None:
                raise RuntimeError("ZIP content verification failed.")
            for path in files:
                with archive.open(path.name) as stored, path.open("rb") as original:
                    if _digest(stored)!=_digest(original):
                        raise RuntimeError(f"ZIP hash verification failed: {path.name}")
        # A same-filesystem hard link publishes only a complete archive and refuses collisions.
        _publish_archive(completed_zip, output)
    except Exception as error:
        raise RuntimeError(f"Replacement failed; recovery folder: {staging}. {error}") from error
    # Remove only the unique folder created by this call, after checking its location.
    if staging.is_symlink() or staging.resolve().parent!=output.parent or not staging.name.startswith(".text-replacement-"):
        raise RuntimeError(f"ZIP saved; unexpected cleanup path retained: {staging}")
    try:
        shutil.rmtree(staging)
    except OSError as error:
        raise RuntimeError(f"ZIP saved at {output}; could not remove recovery folder {staging}.") from error
    return output


def main():
    parser=argparse.ArgumentParser(description="Replace translated text boxes and save images in one ZIP.")
    parser.add_argument("manifest", type=Path, help="JSON with an images list containing paths and translated regions.")
    parser.add_argument("output", type=Path, help="New ZIP path; existing files are never overwritten.")
    parser.add_argument("--font", required=True, type=Path, help="TTF/OTF font with glyphs for the target language.")
    parser.add_argument("--batch-size", type=int, default=50)
    args=parser.parse_args()
    try:
        data=json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("images"), list):
            raise ValueError("Manifest must contain an images list.")
        entries=data["images"]
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("Each image needs a string path.")
            entry["path"]=str(args.manifest.resolve().parent/entry["path"])
        output=replace_and_zip(entries, args.output, font_path=args.font, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
