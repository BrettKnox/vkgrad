"""GLSL compute source -> SPIR-V, with an on-disk cache.

ponytail: glslc already is a shader compiler. We generate text and shell out.
The cache key is (source, flags, glslc version), so a toolchain bump
invalidates automatically.
"""

import glob
import hashlib
import os
import subprocess
import tempfile

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spv_cache")


def find_glslc():
    env = os.environ.get("GLSLC")
    if env and os.path.exists(env):
        return env
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        p = os.path.join(sdk, "Bin", "glslc.exe")
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(r"C:\VulkanSDK\*\Bin\glslc.exe"))
    if hits:
        return hits[-1]
    from shutil import which
    p = which("glslc")
    if p:
        return p
    raise RuntimeError(
        "glslc not found. Install the Vulkan SDK (winget install KhronosGroup.VulkanSDK) "
        "or set the GLSLC environment variable.")


GLSLC = None
_VERSION = None


def _version():
    global GLSLC, _VERSION
    if _VERSION is None:
        GLSLC = find_glslc()
        _VERSION = subprocess.run([GLSLC, "--version"], capture_output=True,
                                  text=True).stdout.strip()
    return _VERSION


class ShaderError(RuntimeError):
    pass


def compile_glsl(src, target_env="vulkan1.3", optimize=True, name="shader"):
    """Compile a GLSL compute shader to SPIR-V bytes. Cached on disk."""
    ver = _version()
    flags = [f"--target-env={target_env}", "-fshader-stage=compute"]
    if optimize:
        flags.append("-O")
    key = hashlib.sha256(("\0".join([src, ver, *flags])).encode()).hexdigest()[:32]
    os.makedirs(_CACHE, exist_ok=True)
    cached = os.path.join(_CACHE, key + ".spv")
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            return f.read()

    with tempfile.TemporaryDirectory() as td:
        srcp = os.path.join(td, name + ".comp")
        outp = os.path.join(td, name + ".spv")
        with open(srcp, "w", encoding="utf-8") as f:
            f.write(src)
        r = subprocess.run([GLSLC, *flags, srcp, "-o", outp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            numbered = "\n".join(f"{i + 1:4d} | {l}" for i, l in enumerate(src.splitlines()))
            raise ShaderError(f"glslc failed for {name}:\n{r.stderr}\n--- source ---\n{numbered}")
        with open(outp, "rb") as f:
            spv = f.read()

    tmp = cached + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(spv)
    os.replace(tmp, cached)  # atomic: concurrent builds can't read a partial file
    return spv


if __name__ == "__main__":
    print("glslc:", find_glslc())
    print(_version())
    spv = compile_glsl("""#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer X { float x[]; };
void main() { x[gl_GlobalInvocationID.x] *= 2.0; }
""", name="selftest")
    assert spv[:4] == b"\x03\x02\x23\x07", "not a SPIR-V module"
    print(f"OK: {len(spv)} bytes, cache at {_CACHE}")
