#!/usr/bin/env python3
"""
Vision model discovery for the species app.

The vision model was hardcoded; this makes it selectable. Two rules learned the
hard way:

  * Classify by GGUF `general.architecture`, NOT by filename or folder. The file
    doing the work today sits in `qwen_image/text_encoders/` because it was
    downloaded as a diffusion text encoder, and another file named
    "Qwen-Image-2.0-...-Uncensored" is in fact a qwen3vl vision model.
  * A GGUF diffusion/vision model is useless for image work without its matching
    projector (`mmproj`), so discover models and projectors separately and pair
    them.

The GGUF reader must SKIP array payloads correctly: reading only the first few
elements of a tokenizer array desynchronises the stream, after which the next
"length" is garbage and you get a MemoryError trying to allocate it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# GGUF value type ids
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_FMT = {_U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
        _F32: "<f", _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d"}
_SIZE = {_U8: 1, _I8: 1, _U16: 2, _I16: 2, _U32: 4, _I32: 4, _F32: 4,
         _BOOL: 1, _U64: 8, _I64: 8, _F64: 8}

# keys that tell us what a file actually is
WANTED = ("general.architecture", "general.name", "general.size_label",
          "general.file_type", "clip.has_vision_encoder", "clip.has_audio_encoder",
          "clip.projector_type", "clip.vision.image_size", "clip.vision.embedding_length",
          "block_count")

VISION_ARCHES = ("qwen3vl", "qwen2vl", "qwen2.5vl", "llava", "gemma3", "pixtral",
                 "internvl", "minicpmv", "mllama", "clip")


def gguf_meta(path: str | Path, max_kv: int = 4000) -> dict:
    """Read selected GGUF metadata, skipping array payloads safely."""
    out: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {"_error": "not a GGUF file"}
            struct.unpack("<I", f.read(4))          # version
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            out["_n_tensors"] = n_tensors

            def rstr() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                if n > 1_000_000:                    # desync guard
                    raise ValueError("implausible string length")
                return f.read(n).decode("utf-8", "replace")

            def skip_value(t: int) -> object:
                """Read a value, or consume it without keeping it."""
                if t in _FMT:
                    return struct.unpack(_FMT[t], f.read(_SIZE[t]))[0]
                if t == _STR:
                    return rstr()
                if t == _ARR:
                    et = struct.unpack("<I", f.read(4))[0]
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > 10_000_000:
                        raise ValueError("implausible array length")
                    if et in _SIZE:
                        f.seek(_SIZE[et] * n, 1)     # skip wholesale, stay aligned
                        return f"<{n} values>"
                    if et == _STR:
                        first = rstr() if n else ""
                        for _ in range(max(0, n - 1)):   # consume the rest
                            rstr()
                        return f"<{n} strings, first={first[:24]!r}>"
                    for _ in range(n):               # nested / unknown: recurse
                        skip_value(et)
                    return f"<{n} nested>"
                raise ValueError(f"unknown gguf type {t}")

            for _ in range(min(n_kv, max_kv)):
                key = rstr()
                vtype = struct.unpack("<I", f.read(4))[0]
                val = skip_value(vtype)
                if key in WANTED or key.startswith("general."):
                    out[key] = val
    except Exception as exc:
        out.setdefault("_error", f"{type(exc).__name__}: {exc}")
    return out


@dataclass
class VisionOption:
    model: Path
    projector: Path | None
    arch: str = ""
    name: str = ""
    label: str = ""
    vision_ok: bool = False
    size_gb: float = 0.0
    note: str = ""
    extra: dict = field(default_factory=dict)

    def title(self) -> str:
        return f"{self.name or self.model.stem}  ({self.size_gb:.1f} GB)"


def _is_projector(p: Path) -> bool:
    return "mmproj" in p.name.lower()


def _vision_capable(p: Path, meta: dict) -> tuple[bool, str]:
    """Decide whether a GGUF can see images."""
    arch = str(meta.get("general.architecture", "")).lower()
    if meta.get("clip.has_vision_encoder") is True:
        return True, "projector advertises a vision encoder"
    if any(a in arch for a in VISION_ARCHES):
        return True, f"architecture '{arch}' is multimodal"
    return False, f"architecture '{arch or '?'}' has no vision encoder"


def discover(model_dirs: list[Path], verify: bool = True) -> list[VisionOption]:
    """Find vision models and their projectors under the given directories."""
    models: list[Path] = []
    projectors: list[Path] = []
    for d in model_dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*.gguf"):
            (projectors if _is_projector(p) else models).append(p)

    # only read headers for files that could plausibly be vision models or
    # projectors; skip the multi-part big text LLMs to keep this fast
    opts: list[VisionOption] = []
    for m in sorted(models):
        try:
            gb = m.stat().st_size / 1e9
        except OSError:
            continue
        if gb > 12:                      # 120B-class text models are not vision
            continue
        meta = gguf_meta(m) if verify else {}
        ok, why = _vision_capable(m, meta)
        if not ok and verify:
            # keep it out of the list but remember why, in case the user asks
            continue
        opts.append(VisionOption(model=m, projector=None,
                                 arch=str(meta.get("general.architecture", "")),
                                 name=str(meta.get("general.name", "")),
                                 label=why, vision_ok=ok, size_gb=gb, extra=meta))

    projs: list[tuple[Path, dict, float]] = []
    for p in sorted(projectors):
        meta = gguf_meta(p) if verify else {}
        projs.append((p, meta, p.stat().st_size / 1e9))

    # pair each model with the best-matching projector (name similarity)
    for o in opts:
        stem = o.model.stem.lower()
        best, best_score = None, -1.0
        for p, pmeta, pgb in projs:
            pname = p.name.lower().replace("mmproj", "").replace(".gguf", "")
            score = 0.0
            if o.arch and o.arch.lower().replace("_", "") in pname.replace("-", "").replace("_", ""):
                score += 3
            for tok in stem.replace("-", " ").replace("_", " ").split():
                if len(tok) > 3 and tok in pname:
                    score += 1
            if score > best_score:
                best, best_score = (p, pmeta, pgb), score
        if best and best_score > 0:
            p, pmeta, pgb = best
            o.projector = p
            o.note = f"projector {pgb:.2f} GB" + (
                " · vision encoder confirmed" if pmeta.get("clip.has_vision_encoder") else "")
            o.vision_ok = True
    return opts


if __name__ == "__main__":
    import sys
    dirs = [Path.home() / "Models"]
    if len(sys.argv) > 1:
        dirs = [Path(a) for a in sys.argv[1:]]
    found = discover(dirs)
    print(f"vision-capable models found: {len(found)}")
    for o in found:
        print(f"\n  {o.title()}")
        print(f"     model     : {o.model}")
        print(f"     arch      : {o.arch}   name={o.name}")
        print(f"     capability: {o.label}")
        print(f"     projector : {o.projector}")
        if o.note:
            print(f"     note      : {o.note}")
