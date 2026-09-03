"""Will vkgrad train on this GPU, and what will it get?

Run this first on any machine. It reports what the device supports, what that
enables, and what is merely nice to have, so the answer is never "it crashed
and I do not know why".

  python check_device.py
"""

import sys

from vk import Device, VkError

# (label, key, what it gates, is it required)
CHECKS = [
    ("16-bit storage", "storageBuffer16BitAccess",
     "f16 activations and weights: halves all operand traffic", True),
    ("f16 arithmetic", "shaderFloat16",
     "f16 math in shaders", True),
    ("cooperative matrix", "cooperativeMatrix",
     "matrix units. Worth ~6% here; the scalar path runs without it", False),
    ("subgroup size control", "subgroupSizeControl",
     "wave32 for cooperative matrix; unused on the scalar path", False),
    ("float atomics", "shaderBufferFloat32AtomicAdd",
     "transformer embeddings and fast bias reductions; the MLP path works without", False),
]


def main():
    try:
        dev = Device()
    except VkError as e:
        print(f"No usable Vulkan device: {e}")
        print("vkgrad needs a Vulkan 1.1 driver with a compute queue.")
        return 1

    api = f"{dev.api_version >> 22}.{(dev.api_version >> 12) & 0x3ff}"
    kind = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}
    print(f"device : {dev.name}")
    print(f"vulkan : {api}   type: {kind.get(dev.device_type)}   "
          f"vendor: 0x{dev.vendor_id:04x}")
    core = dev.shader_core_props()
    if core:
        print(f"cores  : {core['cus']} CUs, wave{core['wavefront_size']}, "
              f"{core['vgprs_per_simd']} VGPRs/SIMD")
    heaps = dev.heap_report()
    print("memory : " + ", ".join(f"{h['gib']:.1f} GiB"
                                  f"{' device-local' if h['device_local'] else ''}"
                                  for h in heaps))
    print()

    required_ok = True
    print(f"{'capability':<24}{'':<5}{'enables'}")
    print("-" * 78)
    for label, key, why, required in CHECKS:
        ok = dev.features.get(key, False)
        mark = "OK " if ok else ("MISSING" if required else "absent")
        if required and not ok:
            required_ok = False
        print(f"{label:<24}{mark:<9}{why}")

    print()
    extra = [("multi-device training", "VK_EXT_external_memory_host",
              "gradient exchange with other GPUs through shared host memory"),
             ("pipeline statistics", "VK_KHR_pipeline_executable_properties",
              "occupancy-aware autotuning")]
    for label, ext, why in extra:
        ok = ext in dev.extensions
        print(f"{label:<24}{'OK ' if ok else 'absent':<9}{why}")

    print()
    if not required_ok:
        print("VERDICT: this device cannot run vkgrad as written.")
        print("It needs 16-bit storage and f16 arithmetic, which essentially")
        print("every GPU since about 2016 has. Older parts would need an")
        print("fp32-only pipeline, which is not implemented.")
        dev.destroy()
        return 1

    if dev.features.get("cooperativeMatrix"):
        speed = "matrix units present: the cooperative-matrix path will be used"
    else:
        speed = ("no matrix units: the scalar fallback will be used, which costs "
                 "about 6%\n         on a real training step, not the 2-3x the peak "
                 "numbers imply")
    print(f"VERDICT: vkgrad will train on this device.\n         {speed}")
    if not dev.features.get("shaderBufferFloat32AtomicAdd"):
        print("         Without float atomics the MLP path works; the transformer")
        print("         needs them for embedding gradients.")
    dev.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
