"""Minimal Vulkan compute runtime via ctypes. No SDK bindings, no C compiler.

Only the entrypoints a training stack needs: device, unified memory, storage
buffers, compute pipelines, dispatch, timing.

ponytail: no graphics, no swapchain, no images. Compute-only by construction.
"""

import ctypes as C
import os
import time

_lib = C.CDLL("vulkan-1.dll")

# ---------------------------------------------------------------- constants

# VkStructureType
ST_APP_INFO = 0
ST_INSTANCE_CI = 1
ST_DEVICE_QUEUE_CI = 2
ST_DEVICE_CI = 3
ST_SUBMIT_INFO = 4
ST_MEMORY_ALLOCATE_INFO = 5
ST_MAPPED_MEMORY_RANGE = 6
ST_FENCE_CI = 8
ST_BUFFER_CI = 12
ST_SHADER_MODULE_CI = 16
ST_PIPELINE_SHADER_STAGE_CI = 18
ST_COMPUTE_PIPELINE_CI = 29
ST_PIPELINE_LAYOUT_CI = 30
ST_DESCRIPTOR_SET_LAYOUT_CI = 32
ST_DESCRIPTOR_POOL_CI = 33
ST_DESCRIPTOR_SET_ALLOCATE_INFO = 34
ST_WRITE_DESCRIPTOR_SET = 35
ST_COMMAND_POOL_CI = 39
ST_COMMAND_BUFFER_ALLOCATE_INFO = 40
ST_COMMAND_BUFFER_BEGIN_INFO = 42
ST_MEMORY_BARRIER = 46
ST_PHYS_VULKAN_11_FEATURES = 49
ST_PHYS_VULKAN_12_FEATURES = 51
ST_PHYS_VULKAN_13_FEATURES = 53
ST_PHYS_FEATURES_2 = 1000059000
ST_PHYS_PROPERTIES_2 = 1000059001
ST_PHYS_SUBGROUP_PROPERTIES = 1000094000
ST_PHYS_SUBGROUP_SIZE_CONTROL_PROPS = 1000225000
ST_REQUIRED_SUBGROUP_SIZE_CI = 1000225001
ST_PHYS_COOP_MATRIX_FEATURES_KHR = 1000506000
ST_COOP_MATRIX_PROPS_KHR = 1000506001
ST_PHYS_PIPELINE_EXEC_PROPS_FEATURES_KHR = 1000269000
ST_PIPELINE_INFO_KHR = 1000269001
ST_PIPELINE_EXECUTABLE_PROPERTIES_KHR = 1000269002
ST_PIPELINE_EXECUTABLE_INFO_KHR = 1000269003
ST_PIPELINE_EXECUTABLE_STATISTIC_KHR = 1000269004

ST_EXTERNAL_MEMORY_BUFFER_CI = 1000072001
ST_IMPORT_MEMORY_HOST_POINTER_INFO_EXT = 1000178000
ST_MEMORY_HOST_POINTER_PROPERTIES_EXT = 1000178001
ST_PHYS_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT = 1000178002
EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT = 0x80

PIPELINE_CREATE_CAPTURE_STATISTICS = 0x40
MAX_SETS_PER_KERNEL = 64

# VkMemoryPropertyFlagBits
MEM_DEVICE_LOCAL = 0x1
MEM_HOST_VISIBLE = 0x2
MEM_HOST_COHERENT = 0x4
MEM_HOST_CACHED = 0x8

QUEUE_GRAPHICS = 0x1
QUEUE_COMPUTE = 0x2

BUF_TRANSFER_SRC = 0x1
BUF_TRANSFER_DST = 0x2
BUF_STORAGE = 0x20

DESC_STORAGE_BUFFER = 7
STAGE_COMPUTE = 0x20
PIPELINE_STAGE_COMPUTE = 0x800
ACCESS_SHADER_READ = 0x20
ACCESS_SHADER_WRITE = 0x40
BIND_POINT_COMPUTE = 1
CMD_POOL_RESET_BIT = 0x2
CMD_ONE_TIME_SUBMIT = 0x1
REQUIRE_FULL_SUBGROUPS = 0x2
WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF

# VkComponentTypeKHR -> short name
COMPONENT_TYPE = {
    0: "f16", 1: "f32", 2: "f64", 3: "i8", 4: "i16", 5: "i32", 6: "i64",
    7: "u8", 8: "u16", 9: "u32", 10: "u64", 1000141000: "bf16",
}
# VkScopeKHR
SCOPE = {1: "device", 2: "workgroup", 3: "subgroup", 5: "queue_family"}


class VkError(RuntimeError):
    pass


def _check(r, what):
    if r != 0:
        raise VkError(f"{what} failed: VkResult {r}")


# ------------------------------------------------------------------ structs
# ctypes inserts alignment padding automatically; do not add manual pads.

def _bools(names):
    return [(n, C.c_uint32) for n in names]


class AppInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("pApplicationName", C.c_char_p), ("applicationVersion", C.c_uint32),
                ("pEngineName", C.c_char_p), ("engineVersion", C.c_uint32),
                ("apiVersion", C.c_uint32)]


class InstanceCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("pApplicationInfo", C.POINTER(AppInfo)),
                ("enabledLayerCount", C.c_uint32), ("ppEnabledLayerNames", C.c_void_p),
                ("enabledExtensionCount", C.c_uint32), ("ppEnabledExtensionNames", C.c_void_p)]


class QueueCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("queueFamilyIndex", C.c_uint32), ("queueCount", C.c_uint32),
                ("pQueuePriorities", C.POINTER(C.c_float))]


class DeviceCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("queueCreateInfoCount", C.c_uint32), ("pQueueCreateInfos", C.POINTER(QueueCI)),
                ("enabledLayerCount", C.c_uint32), ("ppEnabledLayerNames", C.c_void_p),
                ("enabledExtensionCount", C.c_uint32), ("ppEnabledExtensionNames", C.c_void_p),
                ("pEnabledFeatures", C.c_void_p)]


class Features2(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("features", C.c_uint32 * 55)]


V11 = ["storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess",
       "storagePushConstant16", "storageInputOutput16", "multiview",
       "multiviewGeometryShader", "multiviewTessellationShader",
       "variablePointersStorageBuffer", "variablePointers", "protectedMemory",
       "samplerYcbcrConversion", "shaderDrawParameters"]

V12 = ["samplerMirrorClampToEdge", "drawIndirectCount", "storageBuffer8BitAccess",
       "uniformAndStorageBuffer8BitAccess", "storagePushConstant8",
       "shaderBufferInt64Atomics", "shaderSharedInt64Atomics", "shaderFloat16",
       "shaderInt8", "descriptorIndexing",
       "shaderInputAttachmentArrayDynamicIndexing",
       "shaderUniformTexelBufferArrayDynamicIndexing",
       "shaderStorageTexelBufferArrayDynamicIndexing",
       "shaderUniformBufferArrayNonUniformIndexing",
       "shaderSampledImageArrayNonUniformIndexing",
       "shaderStorageBufferArrayNonUniformIndexing",
       "shaderStorageImageArrayNonUniformIndexing",
       "shaderInputAttachmentArrayNonUniformIndexing",
       "shaderUniformTexelBufferArrayNonUniformIndexing",
       "shaderStorageTexelBufferArrayNonUniformIndexing",
       "descriptorBindingUniformBufferUpdateAfterBind",
       "descriptorBindingSampledImageUpdateAfterBind",
       "descriptorBindingStorageImageUpdateAfterBind",
       "descriptorBindingStorageBufferUpdateAfterBind",
       "descriptorBindingUniformTexelBufferUpdateAfterBind",
       "descriptorBindingStorageTexelBufferUpdateAfterBind",
       "descriptorBindingUpdateUnusedWhilePending", "descriptorBindingPartiallyBound",
       "descriptorBindingVariableDescriptorCount", "runtimeDescriptorArray",
       "samplerFilterMinmax", "scalarBlockLayout", "imagelessFramebuffer",
       "uniformBufferStandardLayout", "shaderSubgroupExtendedTypes",
       "separateDepthStencilLayouts", "hostQueryReset", "timelineSemaphore",
       "bufferDeviceAddress", "bufferDeviceAddressCaptureReplay",
       "bufferDeviceAddressMultiDevice", "vulkanMemoryModel",
       "vulkanMemoryModelDeviceScope", "vulkanMemoryModelAvailabilityVisibilityChains",
       "shaderOutputViewportIndex", "shaderOutputLayer", "subgroupBroadcastDynamicId"]

V13 = ["robustImageAccess", "inlineUniformBlock",
       "descriptorBindingInlineUniformBlockUpdateAfterBind",
       "pipelineCreationCacheControl", "privateData",
       "shaderDemoteToHelperInvocation", "shaderTerminateInvocation",
       "subgroupSizeControl", "computeFullSubgroups", "synchronization2",
       "textureCompressionASTC_HDR", "shaderZeroInitializeWorkgroupMemory",
       "dynamicRendering", "shaderIntegerDotProduct", "maintenance4"]


class Vulkan11Features(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p)] + _bools(V11)


class Vulkan12Features(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p)] + _bools(V12)


class Vulkan13Features(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p)] + _bools(V13)


class CoopMatrixFeatures(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("cooperativeMatrix", C.c_uint32),
                ("cooperativeMatrixRobustBufferAccess", C.c_uint32)]


class CoopMatrixProps(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("MSize", C.c_uint32), ("NSize", C.c_uint32), ("KSize", C.c_uint32),
                ("AType", C.c_uint32), ("BType", C.c_uint32), ("CType", C.c_uint32),
                ("ResultType", C.c_uint32), ("saturatingAccumulation", C.c_uint32),
                ("scope", C.c_uint32)]


ATOMIC_FLOAT_FIELDS = [
    "shaderBufferFloat32Atomics", "shaderBufferFloat32AtomicAdd",
    "shaderBufferFloat64Atomics", "shaderBufferFloat64AtomicAdd",
    "shaderSharedFloat32Atomics", "shaderSharedFloat32AtomicAdd",
    "shaderSharedFloat64Atomics", "shaderSharedFloat64AtomicAdd",
    "shaderImageFloat32Atomics", "shaderImageFloat32AtomicAdd",
    "sparseImageFloat32Atomics", "sparseImageFloat32AtomicAdd"]


class AtomicFloatFeatures(C.Structure):
    """VK_EXT_shader_atomic_float. Needed for the embedding gradient, which is
    a scatter-add: many tokens in a batch hit the same vocabulary row."""
    _fields_ = ([("sType", C.c_uint32), ("pNext", C.c_void_p)]
                + [(n, C.c_uint32) for n in ATOMIC_FLOAT_FIELDS])


class ShaderCoreProps(C.Structure):
    """VK_AMD_shader_core_properties. Gives the real register file size, which
    turns a VGPR count into an occupancy number instead of a guess."""
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("shaderEngineCount", C.c_uint32), ("shaderArraysPerEngineCount", C.c_uint32),
                ("computeUnitsPerShaderArray", C.c_uint32), ("simdPerComputeUnit", C.c_uint32),
                ("wavefrontsPerSimd", C.c_uint32), ("wavefrontSize", C.c_uint32),
                ("sgprsPerSimd", C.c_uint32), ("minSgprAllocation", C.c_uint32),
                ("maxSgprAllocation", C.c_uint32), ("sgprAllocationGranularity", C.c_uint32),
                ("vgprsPerSimd", C.c_uint32), ("minVgprAllocation", C.c_uint32),
                ("maxVgprAllocation", C.c_uint32), ("vgprAllocationGranularity", C.c_uint32)]


class PipelineExecPropsFeatures(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("pipelineExecutableInfo", C.c_uint32)]


class PipelineInfoKHR(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("pipeline", C.c_void_p)]


class PipelineExecutableProps(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("stages", C.c_uint32),
                ("name", C.c_char * 256), ("description", C.c_char * 256),
                ("subgroupSize", C.c_uint32)]


class PipelineExecutableInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("pipeline", C.c_void_p),
                ("executableIndex", C.c_uint32)]


class _StatValue(C.Union):
    _fields_ = [("b32", C.c_uint32), ("i64", C.c_int64), ("u64", C.c_uint64),
                ("f64", C.c_double)]


class PipelineExecutableStatistic(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("name", C.c_char * 256), ("description", C.c_char * 256),
                ("format", C.c_uint32), ("value", _StatValue)]


class SubgroupProps(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("subgroupSize", C.c_uint32), ("supportedStages", C.c_uint32),
                ("supportedOperations", C.c_uint32),
                ("quadOperationsInAllStages", C.c_uint32)]


class SubgroupSizeControlProps(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("minSubgroupSize", C.c_uint32), ("maxSubgroupSize", C.c_uint32),
                ("maxComputeWorkgroupSubgroups", C.c_uint32),
                ("requiredSubgroupSizeStages", C.c_uint32)]


class Properties2Head(C.Structure):
    """Only the header; the payload lives in pNext structs we care about."""
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("properties", C.c_ubyte * 1024)]


class MemType(C.Structure):
    _fields_ = [("propertyFlags", C.c_uint32), ("heapIndex", C.c_uint32)]


class MemHeap(C.Structure):
    _fields_ = [("size", C.c_uint64), ("flags", C.c_uint32)]


class MemProps(C.Structure):
    _fields_ = [("memoryTypeCount", C.c_uint32), ("memoryTypes", MemType * 32),
                ("memoryHeapCount", C.c_uint32), ("memoryHeaps", MemHeap * 16)]


class QueueFamilyProps(C.Structure):
    _fields_ = [("queueFlags", C.c_uint32), ("queueCount", C.c_uint32),
                ("timestampValidBits", C.c_uint32),
                ("minImageTransferGranularity", C.c_uint32 * 3)]


class ExtProps(C.Structure):
    _fields_ = [("extensionName", C.c_char * 256), ("specVersion", C.c_uint32)]


class BufferCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("size", C.c_uint64), ("usage", C.c_uint32), ("sharingMode", C.c_uint32),
                ("queueFamilyIndexCount", C.c_uint32), ("pQueueFamilyIndices", C.c_void_p)]


class MemReq(C.Structure):
    _fields_ = [("size", C.c_uint64), ("alignment", C.c_uint64),
                ("memoryTypeBits", C.c_uint32)]


class MemAllocInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("allocationSize", C.c_uint64), ("memoryTypeIndex", C.c_uint32)]


class MappedRange(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("memory", C.c_void_p),
                ("offset", C.c_uint64), ("size", C.c_uint64)]


class DSLBinding(C.Structure):
    _fields_ = [("binding", C.c_uint32), ("descriptorType", C.c_uint32),
                ("descriptorCount", C.c_uint32), ("stageFlags", C.c_uint32),
                ("pImmutableSamplers", C.c_void_p)]


class DSLCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("bindingCount", C.c_uint32), ("pBindings", C.POINTER(DSLBinding))]


class DescPoolSize(C.Structure):
    _fields_ = [("type", C.c_uint32), ("descriptorCount", C.c_uint32)]


class DescPoolCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("maxSets", C.c_uint32), ("poolSizeCount", C.c_uint32),
                ("pPoolSizes", C.POINTER(DescPoolSize))]


class DescSetAllocInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("descriptorPool", C.c_void_p), ("descriptorSetCount", C.c_uint32),
                ("pSetLayouts", C.POINTER(C.c_void_p))]


class DescBufferInfo(C.Structure):
    _fields_ = [("buffer", C.c_void_p), ("offset", C.c_uint64), ("range", C.c_uint64)]


class WriteDescSet(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("dstSet", C.c_void_p),
                ("dstBinding", C.c_uint32), ("dstArrayElement", C.c_uint32),
                ("descriptorCount", C.c_uint32), ("descriptorType", C.c_uint32),
                ("pImageInfo", C.c_void_p), ("pBufferInfo", C.POINTER(DescBufferInfo)),
                ("pTexelBufferView", C.c_void_p)]


class PushConstantRange(C.Structure):
    _fields_ = [("stageFlags", C.c_uint32), ("offset", C.c_uint32), ("size", C.c_uint32)]


class PipelineLayoutCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("setLayoutCount", C.c_uint32), ("pSetLayouts", C.POINTER(C.c_void_p)),
                ("pushConstantRangeCount", C.c_uint32),
                ("pPushConstantRanges", C.POINTER(PushConstantRange))]


class ShaderModuleCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("codeSize", C.c_size_t), ("pCode", C.c_void_p)]


class RequiredSubgroupSizeCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("requiredSubgroupSize", C.c_uint32)]


class ShaderStageCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("stage", C.c_uint32), ("module", C.c_void_p), ("pName", C.c_char_p),
                ("pSpecializationInfo", C.c_void_p)]


class ComputePipelineCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("stage", ShaderStageCI), ("layout", C.c_void_p),
                ("basePipelineHandle", C.c_void_p), ("basePipelineIndex", C.c_int32)]


class CommandPoolCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("queueFamilyIndex", C.c_uint32)]


class CommandBufferAllocInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("commandPool", C.c_void_p),
                ("level", C.c_uint32), ("commandBufferCount", C.c_uint32)]


class CommandBufferBeginInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("pInheritanceInfo", C.c_void_p)]


class SubmitInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("waitSemaphoreCount", C.c_uint32), ("pWaitSemaphores", C.c_void_p),
                ("pWaitDstStageMask", C.c_void_p), ("commandBufferCount", C.c_uint32),
                ("pCommandBuffers", C.POINTER(C.c_void_p)),
                ("signalSemaphoreCount", C.c_uint32), ("pSignalSemaphores", C.c_void_p)]


class FenceCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32)]


class ExternalMemoryBufferCI(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("handleTypes", C.c_uint32)]


class ImportMemoryHostPointerInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("handleType", C.c_uint32), ("pHostPointer", C.c_void_p)]


class MemoryHostPointerProperties(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("memoryTypeBits", C.c_uint32)]


class ExternalMemoryHostProps(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("minImportedHostPointerAlignment", C.c_uint64)]


class BufferCopy(C.Structure):
    _fields_ = [("srcOffset", C.c_uint64), ("dstOffset", C.c_uint64), ("size", C.c_uint64)]


class MemoryBarrier(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("srcAccessMask", C.c_uint32), ("dstAccessMask", C.c_uint32)]


# ------------------------------------------------------------------ helpers

def _strings(items):
    """Keep the array AND the c_char_p buffers alive; return (ptr, count, keep)."""
    encoded = [s.encode() if isinstance(s, str) else s for s in items]
    arr = (C.c_char_p * len(encoded))(*encoded)
    return C.cast(arr, C.c_void_p), len(encoded), (arr, encoded)


class Buffer:
    """A GPU storage buffer. If host-visible, .array() is a live numpy view of
    the same bytes the GPU reads: no copies, ever."""

    def __init__(self, dev, size, kind="shared", usage=None, host_ptr=None):
        self.dev = dev
        self.size = size
        self.kind = kind
        self.host_ptr = host_ptr
        usage = usage if usage is not None else (BUF_STORAGE | BUF_TRANSFER_SRC | BUF_TRANSFER_DST)
        self._keep = []

        ext = None
        pnext = None
        if host_ptr is not None:
            ext = ExternalMemoryBufferCI(ST_EXTERNAL_MEMORY_BUFFER_CI, None,
                                         EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT)
            pnext = C.cast(C.pointer(ext), C.c_void_p)
            self._keep.append(ext)
        ci = BufferCI(ST_BUFFER_CI, pnext, 0, size, usage, 0, 0, None)
        self.buf = C.c_void_p()
        _check(_lib.vkCreateBuffer(dev.dev, C.byref(ci), None, C.byref(self.buf)),
               "vkCreateBuffer")

        req = MemReq()
        _lib.vkGetBufferMemoryRequirements(dev.dev, self.buf, C.byref(req))

        if host_ptr is not None:
            fn = dev._proc("vkGetMemoryHostPointerPropertiesEXT")
            if not fn:
                raise VkError("vkGetMemoryHostPointerPropertiesEXT unavailable")
            F = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p)(fn)
            hp = MemoryHostPointerProperties(ST_MEMORY_HOST_POINTER_PROPERTIES_EXT, None, 0)
            _check(F(dev.dev, EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                     C.c_void_p(host_ptr), C.byref(hp)),
                   "vkGetMemoryHostPointerPropertiesEXT")
            bits = req.memoryTypeBits & hp.memoryTypeBits
            if not bits:
                raise VkError("no memory type can import this host pointer")
            # Prefer a host-coherent type so both sides see writes without
            # explicit flushes.
            self.mem_type = None
            for i in range(dev.mem_props.memoryTypeCount):
                if not (bits & (1 << i)):
                    continue
                f = dev.mem_props.memoryTypes[i].propertyFlags
                if f & MEM_HOST_VISIBLE and f & MEM_HOST_COHERENT:
                    self.mem_type = i
                    break
            if self.mem_type is None:
                self.mem_type = (bits & -bits).bit_length() - 1
            imp = ImportMemoryHostPointerInfo(
                ST_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, None,
                EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, C.c_void_p(host_ptr))
            self._keep.append(imp)
            ai = MemAllocInfo(ST_MEMORY_ALLOCATE_INFO,
                              C.cast(C.pointer(imp), C.c_void_p), req.size, self.mem_type)
        else:
            self.mem_type = dev.find_memory_type(req.memoryTypeBits, kind)
            ai = MemAllocInfo(ST_MEMORY_ALLOCATE_INFO, None, req.size, self.mem_type)
        self.mem = C.c_void_p()
        _check(_lib.vkAllocateMemory(dev.dev, C.byref(ai), None, C.byref(self.mem)),
               "vkAllocateMemory")
        _check(_lib.vkBindBufferMemory(dev.dev, self.buf, self.mem, C.c_uint64(0)),
               "vkBindBufferMemory")

        self.flags = dev.mem_props.memoryTypes[self.mem_type].propertyFlags
        self.host_visible = bool(self.flags & MEM_HOST_VISIBLE)
        self.coherent = bool(self.flags & MEM_HOST_COHERENT)
        self.ptr = None
        if self.host_visible:
            p = C.c_void_p()
            _check(_lib.vkMapMemory(dev.dev, self.mem, C.c_uint64(0), C.c_uint64(size),
                                    0, C.byref(p)), "vkMapMemory")
            self.ptr = p

    def array(self, dtype, shape=None):
        """Live numpy view. Writes land in memory the GPU is already reading."""
        import numpy as np
        if self.ptr is None:
            raise VkError(f"buffer kind={self.kind!r} is not host-visible")
        raw = (C.c_ubyte * self.size).from_address(self.ptr.value)
        a = np.frombuffer(raw, dtype=dtype)
        return a if shape is None else a.reshape(shape)

    def flush(self):
        """CPU writes -> visible to GPU. No-op on coherent memory."""
        if self.coherent or self.ptr is None:
            return
        r = MappedRange(ST_MAPPED_MEMORY_RANGE, None, self.mem, 0, WHOLE_SIZE)
        _check(_lib.vkFlushMappedMemoryRanges(self.dev.dev, 1, C.byref(r)), "vkFlush")

    def invalidate(self):
        """GPU writes -> visible to CPU. No-op on coherent memory."""
        if self.coherent or self.ptr is None:
            return
        r = MappedRange(ST_MAPPED_MEMORY_RANGE, None, self.mem, 0, WHOLE_SIZE)
        _check(_lib.vkInvalidateMappedMemoryRanges(self.dev.dev, 1, C.byref(r)),
               "vkInvalidate")

    def destroy(self):
        if self.ptr is not None:
            _lib.vkUnmapMemory(self.dev.dev, self.mem)
            self.ptr = None
        if self.buf:
            _lib.vkDestroyBuffer(self.dev.dev, self.buf, None)
            self.buf = None
        if self.mem:
            _lib.vkFreeMemory(self.dev.dev, self.mem, None)
            self.mem = None


class Kernel:
    """A compiled compute pipeline plus its descriptor plumbing."""

    def __init__(self, dev, spirv, n_buffers, push_size=0, subgroup_size=None,
                 entry=b"main", name="kernel", capture_stats=False):
        self.dev = dev
        self.name = name
        self.capture_stats = capture_stats and dev.has_pipeline_stats
        self.n_buffers = n_buffers
        self.push_size = push_size
        self._keep = []

        code = (C.c_ubyte * len(spirv)).from_buffer_copy(spirv)
        smci = ShaderModuleCI(ST_SHADER_MODULE_CI, None, 0, len(spirv), C.cast(code, C.c_void_p))
        self.module = C.c_void_p()
        _check(_lib.vkCreateShaderModule(dev.dev, C.byref(smci), None, C.byref(self.module)),
               "vkCreateShaderModule")
        self._keep.append(code)

        binds = (DSLBinding * n_buffers)()
        for i in range(n_buffers):
            binds[i] = DSLBinding(i, DESC_STORAGE_BUFFER, 1, STAGE_COMPUTE, None)
        dslci = DSLCI(ST_DESCRIPTOR_SET_LAYOUT_CI, None, 0, n_buffers, binds)
        self.dsl = C.c_void_p()
        _check(_lib.vkCreateDescriptorSetLayout(dev.dev, C.byref(dslci), None,
                                                C.byref(self.dsl)), "vkCreateDescriptorSetLayout")
        self._keep.append(binds)

        # One descriptor set per distinct buffer tuple, not one per kernel: a
        # recorded command buffer holds its bindings, so two dispatches of the
        # same kernel on different buffers need different sets.
        #
        # Pools are chained rather than fixed. A deeper model binds the same
        # kernel to more tensors (an 8-layer transformer needs well over 64 sets
        # for the optimiser alone), and a fixed pool made model depth fail at
        # construction time for no hardware reason.
        self.pool_size = MAX_SETS_PER_KERNEL
        self.pools = []
        self._free_in_pool = 0
        self._sets = {}
        layouts = (C.c_void_p * 1)(self.dsl)
        self.dset = self._alloc_set()

        pcr = PushConstantRange(STAGE_COMPUTE, 0, push_size)
        plci = PipelineLayoutCI(ST_PIPELINE_LAYOUT_CI, None, 0, 1, layouts,
                                1 if push_size else 0,
                                C.pointer(pcr) if push_size else None)
        self.layout = C.c_void_p()
        _check(_lib.vkCreatePipelineLayout(dev.dev, C.byref(plci), None, C.byref(self.layout)),
               "vkCreatePipelineLayout")
        self._keep += [layouts, pcr]

        stage_flags = 0
        pnext = None
        if subgroup_size is not None and not dev.can_set_subgroup_size:
            subgroup_size = None   # device cannot honour it; kernels adapt instead
        if subgroup_size is not None:
            rss = RequiredSubgroupSizeCI(ST_REQUIRED_SUBGROUP_SIZE_CI, None, subgroup_size)
            pnext = C.cast(C.pointer(rss), C.c_void_p)
            stage_flags = REQUIRE_FULL_SUBGROUPS
            self._keep.append(rss)

        stage = ShaderStageCI(ST_PIPELINE_SHADER_STAGE_CI, pnext, stage_flags,
                              STAGE_COMPUTE, self.module, entry, None)
        pflags = PIPELINE_CREATE_CAPTURE_STATISTICS if self.capture_stats else 0
        cpci = ComputePipelineCI(ST_COMPUTE_PIPELINE_CI, None, pflags, stage,
                                 self.layout, None, 0)
        self.pipeline = C.c_void_p()
        _check(_lib.vkCreateComputePipelines(dev.dev, None, 1, C.byref(cpci), None,
                                             C.byref(self.pipeline)), "vkCreateComputePipelines")

    def statistics(self):
        """Driver-reported resource usage for this pipeline: VGPRs, SGPRs, LDS,
        occupancy. Lets the autotuner reject a tile configuration on register
        pressure without ever running it.

        Requires capture_stats=True at construction. Returns {} otherwise.
        """
        if not self.capture_stats:
            return {}
        dev = self.dev
        get_props = dev._proc("vkGetPipelineExecutablePropertiesKHR")
        get_stats = dev._proc("vkGetPipelineExecutableStatisticsKHR")
        if not (get_props and get_stats):
            return {}
        fn_props = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p,
                               C.POINTER(C.c_uint32), C.c_void_p)(get_props)
        fn_stats = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p,
                               C.POINTER(C.c_uint32), C.c_void_p)(get_stats)

        pinfo = PipelineInfoKHR(ST_PIPELINE_INFO_KHR, None, self.pipeline)
        n = C.c_uint32(0)
        fn_props(dev.dev, C.byref(pinfo), C.byref(n), None)
        if n.value == 0:
            return {}
        props = (PipelineExecutableProps * n.value)()
        for p in props:
            p.sType = ST_PIPELINE_EXECUTABLE_PROPERTIES_KHR
        fn_props(dev.dev, C.byref(pinfo), C.byref(n), C.byref(props))

        out = {}
        for i in range(n.value):
            einfo = PipelineExecutableInfo(ST_PIPELINE_EXECUTABLE_INFO_KHR, None,
                                           self.pipeline, i)
            m = C.c_uint32(0)
            fn_stats(dev.dev, C.byref(einfo), C.byref(m), None)
            stats = (PipelineExecutableStatistic * m.value)()
            for s in stats:
                s.sType = ST_PIPELINE_EXECUTABLE_STATISTIC_KHR
            fn_stats(dev.dev, C.byref(einfo), C.byref(m), C.byref(stats))
            for s in stats:
                key = s.name.decode()
                fmt = s.format
                val = (bool(s.value.b32) if fmt == 0 else s.value.i64 if fmt == 1
                       else s.value.u64 if fmt == 2 else s.value.f64)
                out[key] = val
            out["subgroupSize"] = props[i].subgroupSize
        return out

    def _new_pool(self):
        ps = DescPoolSize(DESC_STORAGE_BUFFER, self.n_buffers * self.pool_size)
        pci = DescPoolCI(ST_DESCRIPTOR_POOL_CI, None, 0, self.pool_size, 1,
                         C.pointer(ps))
        pool = C.c_void_p()
        _check(_lib.vkCreateDescriptorPool(self.dev.dev, C.byref(pci), None,
                                           C.byref(pool)), "vkCreateDescriptorPool")
        self.pools.append(pool)
        self._keep.append(ps)
        self._free_in_pool = self.pool_size
        return pool

    def _alloc_set(self):
        if self._free_in_pool == 0:
            self._new_pool()
        layouts = (C.c_void_p * 1)(self.dsl)
        dsai = DescSetAllocInfo(ST_DESCRIPTOR_SET_ALLOCATE_INFO, None,
                                self.pools[-1], 1, layouts)
        s = C.c_void_p()
        _check(_lib.vkAllocateDescriptorSets(self.dev.dev, C.byref(dsai), C.byref(s)),
               "vkAllocateDescriptorSets")
        self._free_in_pool -= 1
        self._keep.append(layouts)
        return s

    def _write_set(self, dset, buffers):
        infos = (DescBufferInfo * len(buffers))()
        writes = (WriteDescSet * len(buffers))()
        for i, b in enumerate(buffers):
            infos[i] = DescBufferInfo(b.buf, 0, WHOLE_SIZE)
            writes[i] = WriteDescSet(ST_WRITE_DESCRIPTOR_SET, None, dset, i, 0, 1,
                                     DESC_STORAGE_BUFFER, None,
                                     C.cast(C.byref(infos, C.sizeof(DescBufferInfo) * i),
                                            C.POINTER(DescBufferInfo)), None)
        _lib.vkUpdateDescriptorSets(self.dev.dev, len(buffers), writes, 0, None)

    def set_for(self, buffers):
        """A descriptor set dedicated to this buffer tuple, allocated once and
        cached. Required for recorded command buffers, where bindings are baked
        in at record time."""
        if len(buffers) != self.n_buffers:
            raise VkError(f"{self.name}: expected {self.n_buffers} buffers, got {len(buffers)}")
        key = tuple(b.buf.value for b in buffers)
        s = self._sets.get(key)
        if s is None:
            s = self._alloc_set()
            self._write_set(s, buffers)
            self._sets[key] = s
        return s

    def bind(self, buffers):
        if len(buffers) != self.n_buffers:
            raise VkError(f"{self.name}: expected {self.n_buffers} buffers, got {len(buffers)}")
        self._write_set(self.dset, buffers)

    def destroy(self):
        d = self.dev.dev
        for pool in self.pools:
            _lib.vkDestroyDescriptorPool(d, pool, None)
        self.pools = []
        for h, fn in ((self.pipeline, _lib.vkDestroyPipeline),
                      (self.layout, _lib.vkDestroyPipelineLayout),
                      (self.dsl, _lib.vkDestroyDescriptorSetLayout),
                      (self.module, _lib.vkDestroyShaderModule)):
            if h:
                fn(d, h, None)
        self.pipeline = self.layout = self.dsl = self.module = None


class Graph:
    """A pre-recorded sequence of dispatches, replayed with one submit.

    Measured on this hardware: a batched dispatch costs 1.7 us, a submit with a
    fence costs 127 us. A training step is ~15 dispatches, so submitting each
    one separately spends ~1.9 ms of pure overhead before any work happens.
    Recording the whole step once and replaying it removes essentially all of
    that.

    Push constants are baked in at record time, so anything that changes per
    step (learning rate, Adam bias correction) must live in a buffer the CPU
    writes instead. On unified memory that write is free.
    """

    def __init__(self, dev, name="graph"):
        self.dev = dev
        self.name = name
        self.n_dispatch = 0
        self._finished = False
        ai = CommandBufferAllocInfo(ST_COMMAND_BUFFER_ALLOCATE_INFO, None,
                                    dev.cmd_pool, 0, 1)
        self.cmd = C.c_void_p()
        _check(_lib.vkAllocateCommandBuffers(dev.dev, C.byref(ai), C.byref(self.cmd)),
               "vkAllocateCommandBuffers")
        # Not ONE_TIME_SUBMIT: this buffer is replayed every step.
        bi = CommandBufferBeginInfo(ST_COMMAND_BUFFER_BEGIN_INFO, None, 0, None)
        _check(_lib.vkBeginCommandBuffer(self.cmd, C.byref(bi)), "vkBeginCommandBuffer")
        self._barrier = MemoryBarrier(ST_MEMORY_BARRIER, None, ACCESS_SHADER_WRITE,
                                      ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
        self._keep = []
        # (name, compulsory bytes, bytes including tile re-reads). Lets a step
        # be compared against measured DRAM bandwidth instead of guessed at.
        self.traffic = []
        self.mm_split = []  # (operand read bytes, output write bytes) per matmul

    def record(self, kernel, buffers, groups, push=b"", bytes_hint=None):
        if self._finished:
            raise VkError(f"{self.name}: already finished")
        gx, gy, gz = (tuple(groups) + (1, 1))[:3] if isinstance(groups, (tuple, list)) \
            else (groups, 1, 1)
        dset = kernel.set_for(buffers)
        if self.n_dispatch:
            _lib.vkCmdPipelineBarrier(self.cmd, PIPELINE_STAGE_COMPUTE,
                                      PIPELINE_STAGE_COMPUTE, 0,
                                      1, C.byref(self._barrier), 0, None, 0, None)
        _lib.vkCmdBindPipeline(self.cmd, BIND_POINT_COMPUTE, kernel.pipeline)
        sets = (C.c_void_p * 1)(dset)
        _lib.vkCmdBindDescriptorSets(self.cmd, BIND_POINT_COMPUTE, kernel.layout,
                                     0, 1, sets, 0, None)
        self._keep.append(sets)
        if push:
            if len(push) != kernel.push_size:
                raise VkError(f"{kernel.name}: push size {len(push)} != {kernel.push_size}")
            pbuf = (C.c_ubyte * len(push)).from_buffer_copy(push)
            _lib.vkCmdPushConstants(self.cmd, kernel.layout, STAGE_COMPUTE, 0,
                                    len(push), pbuf)
            self._keep.append(pbuf)
        _lib.vkCmdDispatch(self.cmd, gx, gy, gz)
        compulsory = sum(b.size for b in buffers)
        self.traffic.append((kernel.name, compulsory,
                             compulsory if bytes_hint is None else bytes_hint))
        self.n_dispatch += 1

    def finish(self):
        _check(_lib.vkEndCommandBuffer(self.cmd), "vkEndCommandBuffer")
        self._finished = True
        self._cmds = (C.c_void_p * 1)(self.cmd)
        self._si = SubmitInfo(ST_SUBMIT_INFO, None, 0, None, None, 1, self._cmds, 0, None)
        return self

    def submit(self):
        if not self._finished:
            raise VkError(f"{self.name}: call finish() before submit()")
        d = self.dev
        _check(_lib.vkResetFences(d.dev, 1, C.byref(d.fence)), "vkResetFences")
        t0 = time.perf_counter()
        _check(_lib.vkQueueSubmit(d.queue, 1, C.byref(self._si), d.fence), "vkQueueSubmit")
        r = _lib.vkWaitForFences(d.dev, 1, C.byref(d.fence), 1, C.c_uint64(10_000_000_000))
        dt = time.perf_counter() - t0
        if r == 2:
            raise VkError(f"{self.name}: timed out after 10s")
        _check(r, "vkWaitForFences")
        return dt


class Device:
    def __init__(self, validate=None, prefer_async_compute=True):
        if validate is None:
            validate = os.environ.get("VKGRAD_VALIDATE", "0") == "1"
        self.validate = validate
        self._create_instance(validate)
        self._pick_physical()
        self._create_device()
        self.memory_tier = {}
        self._init_subgroup_plan()
        self._create_pools()

    # -- setup ------------------------------------------------------------

    def _create_instance(self, validate):
        ai = AppInfo(ST_APP_INFO, None, b"vkgrad", 1, b"vkgrad", 1, (1 << 22) | (3 << 12))
        layers, nlayers, keep = ([], 0, None)
        if validate:
            layers, nlayers, keep = _strings(["VK_LAYER_KHRONOS_validation"])
        self._keep_layers = keep
        ci = InstanceCI(ST_INSTANCE_CI, None, 0, C.pointer(ai), nlayers,
                        layers if validate else None, 0, None)
        self.instance = C.c_void_p()
        r = _lib.vkCreateInstance(C.byref(ci), None, C.byref(self.instance))
        if r != 0 and validate:
            # Validation layer missing -> retry without it rather than dying.
            self.validate = False
            ci = InstanceCI(ST_INSTANCE_CI, None, 0, C.pointer(ai), 0, None, 0, None)
            r = _lib.vkCreateInstance(C.byref(ci), None, C.byref(self.instance))
        _check(r, "vkCreateInstance")
        self._app_info = ai

    def _proc(self, name):
        _lib.vkGetInstanceProcAddr.restype = C.c_void_p
        _lib.vkGetInstanceProcAddr.argtypes = [C.c_void_p, C.c_char_p]
        return _lib.vkGetInstanceProcAddr(self.instance, name.encode())

    def _pick_physical(self):
        n = C.c_uint32(0)
        _lib.vkEnumeratePhysicalDevices(self.instance, C.byref(n), None)
        if n.value == 0:
            raise VkError("no Vulkan physical devices")
        devs = (C.c_void_p * n.value)()
        _lib.vkEnumeratePhysicalDevices(self.instance, C.byref(n), devs)

        # Prefer discrete, then integrated; skip CPU/software.
        best, best_score = None, -1
        for i in range(n.value):
            d = C.c_void_p(devs[i])
            buf = (C.c_ubyte * 2048)()
            _lib.vkGetPhysicalDeviceProperties(d, buf)
            dtype = C.cast(buf, C.POINTER(C.c_uint32))[4]
            score = {2: 3, 1: 2, 0: 1, 3: 1, 4: 0}.get(dtype, 0)
            if score > best_score:
                best, best_score, self._props_buf = d, score, buf
        self.phys = best

        p = C.cast(self._props_buf, C.POINTER(C.c_uint32))
        self.api_version = p[0]
        self.driver_version = p[1]
        self.vendor_id = p[2]
        self.device_id = p[3]
        self.device_type = p[4]
        self.name = bytes(self._props_buf[20:276]).split(b"\0")[0].decode()

        self.mem_props = MemProps()
        _lib.vkGetPhysicalDeviceMemoryProperties(self.phys, C.byref(self.mem_props))

        qn = C.c_uint32(0)
        _lib.vkGetPhysicalDeviceQueueFamilyProperties(self.phys, C.byref(qn), None)
        qs = (QueueFamilyProps * qn.value)()
        _lib.vkGetPhysicalDeviceQueueFamilyProperties(self.phys, C.byref(qn), qs)
        self.queue_families = [(i, qs[i].queueFlags, qs[i].queueCount) for i in range(qn.value)]
        # Async-compute family (compute without graphics) avoids fighting the
        # desktop compositor on an iGPU.
        async_only = [i for i, f, _ in self.queue_families
                      if (f & QUEUE_COMPUTE) and not (f & QUEUE_GRAPHICS)]
        any_compute = [i for i, f, _ in self.queue_families if f & QUEUE_COMPUTE]
        if not any_compute:
            raise VkError("no compute queue family")
        self.qfam = async_only[0] if async_only else any_compute[0]

        en = C.c_uint32(0)
        _lib.vkEnumerateDeviceExtensionProperties(self.phys, None, C.byref(en), None)
        exts = (ExtProps * en.value)()
        _lib.vkEnumerateDeviceExtensionProperties(self.phys, None, C.byref(en), exts)
        self.extensions = {e.extensionName.decode() for e in exts}

    def _create_device(self):
        want = ["VK_KHR_cooperative_matrix", "VK_EXT_subgroup_size_control",
                "VK_KHR_shader_float16_int8", "VK_KHR_16bit_storage",
                "VK_KHR_8bit_storage", "VK_KHR_pipeline_executable_properties",
                "VK_EXT_memory_budget", "VK_EXT_memory_priority",
                "VK_EXT_shader_atomic_float", "VK_EXT_external_memory_host",
                "VK_KHR_external_memory"]
        self.enabled_extensions = [e for e in want if e in self.extensions]
        ext_ptr, ext_n, keep_ext = _strings(self.enabled_extensions)

        self.has_coop_matrix = "VK_KHR_cooperative_matrix" in self.extensions

        # Ask the device what it supports BEFORE requesting anything. Enabling
        # an unsupported feature makes vkCreateDevice fail outright, so a
        # speculative request is the difference between "degrades gracefully"
        # and "does not run at all" on older hardware.
        s13 = Vulkan13Features(ST_PHYS_VULKAN_13_FEATURES, None)
        s12 = Vulkan12Features(ST_PHYS_VULKAN_12_FEATURES,
                               C.cast(C.pointer(s13), C.c_void_p))
        s11 = Vulkan11Features(ST_PHYS_VULKAN_11_FEATURES,
                               C.cast(C.pointer(s12), C.c_void_p))
        shead = C.cast(C.pointer(s11), C.c_void_p)
        scm = None
        if self.has_coop_matrix:
            scm = CoopMatrixFeatures(ST_PHYS_COOP_MATRIX_FEATURES_KHR, shead, 0, 0)
            shead = C.cast(C.pointer(scm), C.c_void_p)
        saf = None
        if "VK_EXT_shader_atomic_float" in self.extensions:
            saf = AtomicFloatFeatures(1000260000, shead)
            shead = C.cast(C.pointer(saf), C.c_void_p)
        sf2 = Features2(ST_PHYS_FEATURES_2, shead)
        _lib.vkGetPhysicalDeviceFeatures2(self.phys, C.byref(sf2))

        # Only these are actually used by a kernel. Anything else was
        # speculative and is not requested.
        want11 = ["storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess"]
        want12 = ["shaderFloat16"]
        want13 = ["subgroupSizeControl", "computeFullSubgroups"]

        f13 = Vulkan13Features(ST_PHYS_VULKAN_13_FEATURES, None)
        f12 = Vulkan12Features(ST_PHYS_VULKAN_12_FEATURES,
                               C.cast(C.pointer(f13), C.c_void_p))
        f11 = Vulkan11Features(ST_PHYS_VULKAN_11_FEATURES,
                               C.cast(C.pointer(f12), C.c_void_p))

        self.features = {}
        for src_, dst_, names in ((s11, f11, want11), (s12, f12, want12),
                                  (s13, f13, want13)):
            for nm in names:
                ok = bool(getattr(src_, nm))
                self.features[nm] = ok
                if ok:
                    setattr(dst_, nm, 1)

        head = C.cast(C.pointer(f11), C.c_void_p)
        cm = None
        if self.has_coop_matrix and scm and scm.cooperativeMatrix:
            cm = CoopMatrixFeatures(ST_PHYS_COOP_MATRIX_FEATURES_KHR, head, 1, 0)
            head = C.cast(C.pointer(cm), C.c_void_p)
        else:
            self.has_coop_matrix = False
        self.features["cooperativeMatrix"] = self.has_coop_matrix

        af = None
        self.has_atomic_float = bool(saf and saf.shaderBufferFloat32AtomicAdd)
        if self.has_atomic_float:
            af = AtomicFloatFeatures(1000260000, head)
            af.shaderBufferFloat32Atomics = 1
            af.shaderBufferFloat32AtomicAdd = 1
            head = C.cast(C.pointer(af), C.c_void_p)
        self.features["shaderBufferFloat32AtomicAdd"] = self.has_atomic_float

        pep = None
        self.has_pipeline_stats = "VK_KHR_pipeline_executable_properties" in self.extensions
        if self.has_pipeline_stats:
            pep = PipelineExecPropsFeatures(ST_PHYS_PIPELINE_EXEC_PROPS_FEATURES_KHR,
                                            head, 1)
            head = C.cast(C.pointer(pep), C.c_void_p)

        f2 = Features2(ST_PHYS_FEATURES_2, head)
        prio = (C.c_float * 1)(1.0)
        qci = QueueCI(ST_DEVICE_QUEUE_CI, None, 0, self.qfam, 1, prio)
        dci = DeviceCI(ST_DEVICE_CI, C.cast(C.pointer(f2), C.c_void_p), 0,
                       1, C.pointer(qci), 0, None, ext_n, ext_ptr, None)

        self.dev = C.c_void_p()
        _check(_lib.vkCreateDevice(self.phys, C.byref(dci), None, C.byref(self.dev)),
               "vkCreateDevice")
        self._keep_device = (f2, f11, f12, f13, cm, pep, af, prio, qci,
                             keep_ext, ext_ptr, s11, s12, s13, scm, saf, sf2)

        self.queue = C.c_void_p()
        _lib.vkGetDeviceQueue(self.dev, self.qfam, 0, C.byref(self.queue))

    def _create_pools(self):
        ci = CommandPoolCI(ST_COMMAND_POOL_CI, None, CMD_POOL_RESET_BIT, self.qfam)
        self.cmd_pool = C.c_void_p()
        _check(_lib.vkCreateCommandPool(self.dev, C.byref(ci), None, C.byref(self.cmd_pool)),
               "vkCreateCommandPool")
        ai = CommandBufferAllocInfo(ST_COMMAND_BUFFER_ALLOCATE_INFO, None, self.cmd_pool, 0, 1)
        self.cmd = C.c_void_p()
        _check(_lib.vkAllocateCommandBuffers(self.dev, C.byref(ai), C.byref(self.cmd)),
               "vkAllocateCommandBuffers")
        fci = FenceCI(ST_FENCE_CI, None, 0)
        self.fence = C.c_void_p()
        _check(_lib.vkCreateFence(self.dev, C.byref(fci), None, C.byref(self.fence)),
               "vkCreateFence")

    # -- queries ----------------------------------------------------------

    def _init_subgroup_plan(self):
        """Decide the width row-reduction kernels will be generated for.

        Kernels that stride a row across the lanes of one subgroup must use a
        stride equal to the actual subgroup size. A mismatch does not crash, it
        silently returns wrong sums (measured: 84% error), so the width has to
        be negotiated rather than assumed. Prefer 32 where the device will grant
        it, otherwise generate for whatever the device natively uses.
        """
        props = SubgroupProps(ST_PHYS_SUBGROUP_PROPERTIES, None)
        head = Properties2Head(ST_PHYS_PROPERTIES_2, C.cast(C.pointer(props), C.c_void_p))
        _lib.vkGetPhysicalDeviceProperties2(self.phys, C.byref(head))
        self.subgroup_size_native = int(props.subgroupSize) or 32
        lo, hi, stages = self.subgroup_size_range()
        can_set = (self.features.get("subgroupSizeControl", False)
                   and bool(stages & STAGE_COMPUTE) and lo <= 32 <= hi)
        # Simulate a device without subgroup size control (older AMD at wave64,
        # Intel at 8/16, anything pre-Vulkan-1.3) to exercise the adaptive path.
        if os.environ.get("VKGRAD_NATIVE_SUBGROUP") == "1":
            can_set = False
        self.can_set_subgroup_size = can_set
        self.row_subgroup_size = 32 if can_set else self.subgroup_size_native
        # The cooperative-matrix kernels are generated for 32-wide subgroups
        # (local_size = sg*32, tiled by gl_SubgroupID). That is only valid if
        # the device is natively 32 wide or will grant 32 on request; otherwise
        # the subgroup tiling would silently mis-index and the scalar path is
        # used instead.
        self.coopmat_wave32_ok = can_set or self.subgroup_size_native == 32
        if self.has_coop_matrix and not self.coopmat_wave32_ok:
            self.has_coop_matrix = False
            self.features["cooperativeMatrix"] = False

    def subgroup_size_range(self):
        props = SubgroupSizeControlProps(ST_PHYS_SUBGROUP_SIZE_CONTROL_PROPS, None)
        head = Properties2Head(ST_PHYS_PROPERTIES_2, C.cast(C.pointer(props), C.c_void_p))
        _lib.vkGetPhysicalDeviceProperties2(self.phys, C.byref(head))
        return props.minSubgroupSize, props.maxSubgroupSize, props.requiredSubgroupSizeStages

    def coop_matrix_configs(self):
        if not self.has_coop_matrix:
            return []
        addr = self._proc("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
        if not addr:
            return []
        fn = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_uint32), C.c_void_p)(addr)
        cnt = C.c_uint32(0)
        fn(self.phys, C.byref(cnt), None)
        arr = (CoopMatrixProps * cnt.value)()
        for x in arr:
            x.sType = ST_COOP_MATRIX_PROPS_KHR
        fn(self.phys, C.byref(cnt), C.byref(arr))
        out = []
        for x in arr:
            out.append(dict(M=x.MSize, N=x.NSize, K=x.KSize,
                            A=COMPONENT_TYPE.get(x.AType, x.AType),
                            B=COMPONENT_TYPE.get(x.BType, x.BType),
                            C=COMPONENT_TYPE.get(x.CType, x.CType),
                            R=COMPONENT_TYPE.get(x.ResultType, x.ResultType),
                            sat=bool(x.saturatingAccumulation),
                            scope=SCOPE.get(x.scope, x.scope)))
        return out

    # Ordered preference per role, most specific first, each entry
    # (required flags, forbidden flags). The last entry of every list is
    # deliberately permissive.
    #
    # The strict forms describe a discrete GPU with separate VRAM. A fully
    # unified device (Intel integrated, Apple via MoltenVK, Mali, Adreno, some
    # AMD APU configurations) exposes NO memory that is device-local and not
    # host-visible, so demanding that outright makes every allocation fail and
    # the framework dead on arrival there.
    KINDS = {
        "device": [(MEM_DEVICE_LOCAL, MEM_HOST_VISIBLE),   # discrete VRAM
                   (MEM_DEVICE_LOCAL, 0),                  # unified: still local
                   (0, 0)],                                # anything at all
        "shared": [(MEM_HOST_VISIBLE | MEM_HOST_COHERENT, MEM_HOST_CACHED),
                   (MEM_HOST_VISIBLE | MEM_HOST_COHERENT, 0),
                   (MEM_HOST_VISIBLE, 0)],
        "cached": [(MEM_HOST_VISIBLE | MEM_HOST_CACHED, 0),
                   (MEM_HOST_VISIBLE | MEM_HOST_COHERENT, 0),
                   (MEM_HOST_VISIBLE, 0)],
        "bar":    [(MEM_DEVICE_LOCAL | MEM_HOST_VISIBLE, 0),
                   (MEM_HOST_VISIBLE, 0)],
        "imported": [(MEM_HOST_VISIBLE | MEM_HOST_COHERENT, 0),
                     (MEM_HOST_VISIBLE, 0)],
    }

    def find_memory_type(self, type_bits, kind):
        """Best available memory type for a role, degrading rather than failing.

        Returns the type index; `self.memory_tier[kind]` records which
        preference level was actually satisfied, so a device that only offers a
        weaker form is visible rather than silent.
        """
        if kind not in self.KINDS:
            raise VkError(f"unknown memory kind {kind!r}; use {list(self.KINDS)}")
        prefs = self.KINDS[kind]
        if os.environ.get("VKGRAD_UMA") == "1" and kind == "device":
            # Simulate a fully unified device, where every memory type is
            # host-visible and none is private VRAM.
            prefs = [(MEM_DEVICE_LOCAL | MEM_HOST_VISIBLE, 0),
                     (MEM_HOST_VISIBLE, 0)]
        for tier, (required, forbidden) in enumerate(prefs):
            for i in range(self.mem_props.memoryTypeCount):
                if not (type_bits & (1 << i)):
                    continue
                f = self.mem_props.memoryTypes[i].propertyFlags
                if (f & required) == required and not (f & forbidden):
                    self.memory_tier[kind] = tier
                    return i
        raise VkError(f"no memory type for kind={kind!r} (type_bits=0x{type_bits:x})")

    def host_pointer_alignment(self):
        """Alignment an imported host allocation must satisfy, or None if this
        device cannot import host memory."""
        if "VK_EXT_external_memory_host" not in self.extensions:
            return None
        props = ExternalMemoryHostProps(ST_PHYS_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT, None)
        head = Properties2Head(ST_PHYS_PROPERTIES_2, C.cast(C.pointer(props), C.c_void_p))
        _lib.vkGetPhysicalDeviceProperties2(self.phys, C.byref(head))
        return int(props.minImportedHostPointerAlignment)

    def import_host_buffer(self, ptr, size, usage=None):
        """Wrap an existing host allocation as a GPU buffer, without copying.

        The same host pointer can be imported by several independent VkDevices,
        including devices from different vendors. That makes plain system memory
        a shared arena between GPUs: on unified memory it is literally the same
        DRAM, and elsewhere it is the host side of each device's PCIe path.

        This is the primitive a cross-vendor gradient exchange needs, and it
        requires no vendor interconnect, no NCCL, and no common driver stack.
        """
        return Buffer(self, size, kind="imported", usage=usage, host_ptr=ptr)

    def shader_core_props(self):
        """CU counts and register file sizes, or None on non-AMD drivers."""
        if "VK_AMD_shader_core_properties" not in self.extensions:
            return None
        sc = ShaderCoreProps(1000185000, None)
        head = Properties2Head(ST_PHYS_PROPERTIES_2, C.cast(C.pointer(sc), C.c_void_p))
        _lib.vkGetPhysicalDeviceProperties2(self.phys, C.byref(head))
        return dict(cus=sc.shaderEngineCount * sc.shaderArraysPerEngineCount
                    * sc.computeUnitsPerShaderArray,
                    simd_per_cu=sc.simdPerComputeUnit,
                    waves_per_simd=sc.wavefrontsPerSimd,
                    wavefront_size=sc.wavefrontSize,
                    vgprs_per_simd=sc.vgprsPerSimd,
                    vgpr_granularity=sc.vgprAllocationGranularity,
                    max_vgpr_alloc=sc.maxVgprAllocation)

    def kind_heap_bytes(self, kind):
        """Size of the heap a given memory kind allocates from. Lets callers
        size allocations to fit (the BAR heap is only 256 MiB here)."""
        idx = self.find_memory_type(0xFFFFFFFF, kind)
        return self.mem_props.memoryHeaps[self.mem_props.memoryTypes[idx].heapIndex].size

    def heap_report(self):
        out = []
        for h in range(self.mem_props.memoryHeapCount):
            out.append(dict(heap=h, gib=self.mem_props.memoryHeaps[h].size / 2**30,
                            device_local=bool(self.mem_props.memoryHeaps[h].flags & 1)))
        return out

    # -- execution --------------------------------------------------------

    def buffer(self, size, kind="shared", usage=None):
        return Buffer(self, size, kind, usage)

    def graph(self, name="graph"):
        return Graph(self, name)

    def kernel(self, spirv, n_buffers, push_size=0, subgroup_size=None, name="kernel",
               capture_stats=False):
        return Kernel(self, spirv, n_buffers, push_size, subgroup_size, name=name,
                      capture_stats=capture_stats)

    def copy(self, src, dst, size=None, repeat=1):
        """Staging copy src->dst on the GPU. Returns wall seconds.

        This is the tax a discrete GPU pays on every tensor. On unified memory
        it should be avoidable entirely; bench/roofline.py measures whether
        avoiding it actually wins.
        """
        size = size if size is not None else min(src.size, dst.size)
        region = BufferCopy(0, 0, size)
        _check(_lib.vkResetCommandBuffer(self.cmd, 0), "vkResetCommandBuffer")
        bi = CommandBufferBeginInfo(ST_COMMAND_BUFFER_BEGIN_INFO, None, CMD_ONE_TIME_SUBMIT, None)
        _check(_lib.vkBeginCommandBuffer(self.cmd, C.byref(bi)), "vkBeginCommandBuffer")
        for _ in range(repeat):
            _lib.vkCmdCopyBuffer(self.cmd, src.buf, dst.buf, 1, C.byref(region))
        _check(_lib.vkEndCommandBuffer(self.cmd), "vkEndCommandBuffer")

        cmds = (C.c_void_p * 1)(self.cmd)
        si = SubmitInfo(ST_SUBMIT_INFO, None, 0, None, None, 1, cmds, 0, None)
        _check(_lib.vkResetFences(self.dev, 1, C.byref(self.fence)), "vkResetFences")
        t0 = time.perf_counter()
        _check(_lib.vkQueueSubmit(self.queue, 1, C.byref(si), self.fence), "vkQueueSubmit")
        r = _lib.vkWaitForFences(self.dev, 1, C.byref(self.fence), 1, C.c_uint64(10_000_000_000))
        dt = time.perf_counter() - t0
        if r == 2:
            raise VkError("copy timed out after 10s")
        _check(r, "vkWaitForFences")
        return dt

    def run(self, kernel, buffers, groups, push=b"", repeat=1, wait=True):
        """Dispatch `kernel` `repeat` times back to back. Returns wall seconds.

        Barriers separate repeats so each run observes the previous one; the
        barrier cost is negligible next to any dispatch worth timing.
        """
        gx, gy, gz = (groups + (1, 1))[:3] if isinstance(groups, tuple) else (groups, 1, 1)
        kernel.bind(buffers)

        _check(_lib.vkResetCommandBuffer(self.cmd, 0), "vkResetCommandBuffer")
        bi = CommandBufferBeginInfo(ST_COMMAND_BUFFER_BEGIN_INFO, None, CMD_ONE_TIME_SUBMIT, None)
        _check(_lib.vkBeginCommandBuffer(self.cmd, C.byref(bi)), "vkBeginCommandBuffer")
        _lib.vkCmdBindPipeline(self.cmd, BIND_POINT_COMPUTE, kernel.pipeline)
        sets = (C.c_void_p * 1)(kernel.dset)
        _lib.vkCmdBindDescriptorSets(self.cmd, BIND_POINT_COMPUTE, kernel.layout,
                                     0, 1, sets, 0, None)
        if push:
            if len(push) != kernel.push_size:
                raise VkError(f"push constant size {len(push)} != {kernel.push_size}")
            pbuf = (C.c_ubyte * len(push)).from_buffer_copy(push)
            _lib.vkCmdPushConstants(self.cmd, kernel.layout, STAGE_COMPUTE, 0,
                                    len(push), pbuf)
        barrier = MemoryBarrier(ST_MEMORY_BARRIER, None,
                                ACCESS_SHADER_WRITE, ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
        for i in range(repeat):
            if i:
                _lib.vkCmdPipelineBarrier(self.cmd, PIPELINE_STAGE_COMPUTE,
                                          PIPELINE_STAGE_COMPUTE, 0,
                                          1, C.byref(barrier), 0, None, 0, None)
            _lib.vkCmdDispatch(self.cmd, gx, gy, gz)
        _check(_lib.vkEndCommandBuffer(self.cmd), "vkEndCommandBuffer")

        cmds = (C.c_void_p * 1)(self.cmd)
        si = SubmitInfo(ST_SUBMIT_INFO, None, 0, None, None, 1, cmds, 0, None)
        _check(_lib.vkResetFences(self.dev, 1, C.byref(self.fence)), "vkResetFences")
        t0 = time.perf_counter()
        _check(_lib.vkQueueSubmit(self.queue, 1, C.byref(si), self.fence), "vkQueueSubmit")
        if not wait:
            return 0.0
        # 10s timeout: longer than any sane kernel, shorter than a wedged session.
        r = _lib.vkWaitForFences(self.dev, 1, C.byref(self.fence), 1, C.c_uint64(10_000_000_000))
        dt = time.perf_counter() - t0
        if r == 2:
            raise VkError("kernel timed out after 10s (VK_TIMEOUT) - likely a hung shader")
        _check(r, "vkWaitForFences")
        return dt

    def destroy(self):
        if getattr(self, "dev", None):
            _lib.vkDeviceWaitIdle(self.dev)
            _lib.vkDestroyFence(self.dev, self.fence, None)
            _lib.vkDestroyCommandPool(self.dev, self.cmd_pool, None)
            _lib.vkDestroyDevice(self.dev, None)
            self.dev = None
        if getattr(self, "instance", None):
            _lib.vkDestroyInstance(self.instance, None)
            self.instance = None

    def __repr__(self):
        t = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}
        return (f"<Device {self.name!r} {t.get(self.device_type)} "
                f"api={self.api_version >> 22}.{(self.api_version >> 12) & 0x3ff} "
                f"qfam={self.qfam} coop_matrix={self.has_coop_matrix}>")


if __name__ == "__main__":
    d = Device()
    print(d)
    print("queue families:", d.queue_families)
    print("heaps:", d.heap_report())
    print("subgroup size range (min,max,stages):", d.subgroup_size_range())
    print("enabled extensions:", d.enabled_extensions)
    for cfg in d.coop_matrix_configs():
        print("  coopmat", cfg)
    for kind in ("device", "shared", "cached", "bar"):
        try:
            b = d.buffer(1 << 20, kind)
            print(f"  {kind:7s} -> memtype {b.mem_type} flags 0x{b.flags:x} "
                  f"host_visible={b.host_visible}")
            b.destroy()
        except VkError as e:
            print(f"  {kind:7s} -> {e}")
    d.destroy()
