#include <metal_stdlib>
using namespace metal;

// BT.601 full-range YUV→RGB conversion
inline float4 yuv_to_rgb(float y, float cb, float cr) {
    float r = y + 1.402 * (cr - 0.5);
    float g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5);
    float b = y + 1.772 * (cb - 0.5);
    return float4(saturate(r), saturate(g), saturate(b), 1.0);
}

// ─── BGRA input kernels ────────────────────────────────────────────

kernel void downsample_bilinear(
    texture2d<float, access::sample> inTexture  [[texture(0)]],
    texture2d<float, access::write>  outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    constexpr sampler s(filter::linear, address::clamp_to_edge);

    float2 outSize = float2(outTexture.get_width(), outTexture.get_height());
    float2 uv = (float2(gid) + 0.5) / outSize;

    float4 color = inTexture.sample(s, uv);
    outTexture.write(color, gid);
}

struct LetterboxParams {
    float2 offset;
    float2 innerSize;
};

kernel void downsample_letterbox(
    texture2d<float, access::sample> inTexture  [[texture(0)]],
    texture2d<float, access::write>  outTexture [[texture(1)]],
    constant LetterboxParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    float2 pos = float2(gid) - params.offset;

    if (pos.x < 0.0 || pos.y < 0.0 || pos.x >= params.innerSize.x || pos.y >= params.innerSize.y) {
        outTexture.write(float4(0.5, 0.5, 0.5, 1.0), gid);
        return;
    }

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (pos + 0.5) / params.innerSize;
    outTexture.write(inTexture.sample(s, uv), gid);
}

// ─── YUV (NV12) input kernels ──────────────────────────────────────
// Y plane: r8Unorm (full width × height)
// UV plane: rg8Unorm (half width × half height, r=Cb, g=Cr)

kernel void downsample_yuv_bilinear(
    texture2d<float, access::sample> yTexture  [[texture(0)]],
    texture2d<float, access::sample> uvTexture [[texture(1)]],
    texture2d<float, access::write>  outTexture [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(outTexture.get_width(), outTexture.get_height());

    float y  = yTexture.sample(s, uv).r;
    float cb = uvTexture.sample(s, uv).r;
    float cr = uvTexture.sample(s, uv).g;

    outTexture.write(yuv_to_rgb(y, cb, cr), gid);
}

kernel void downsample_yuv_letterbox(
    texture2d<float, access::sample> yTexture  [[texture(0)]],
    texture2d<float, access::sample> uvTexture [[texture(1)]],
    texture2d<float, access::write>  outTexture [[texture(2)]],
    constant LetterboxParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float2 pos = float2(gid) - params.offset;
    if (pos.x < 0.0 || pos.y < 0.0 || pos.x >= params.innerSize.x || pos.y >= params.innerSize.y) {
        outTexture.write(float4(0.5, 0.5, 0.5, 1.0), gid);
        return;
    }

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (pos + 0.5) / params.innerSize;

    float y  = yTexture.sample(s, uv).r;
    float cb = uvTexture.sample(s, uv).r;
    float cr = uvTexture.sample(s, uv).g;

    outTexture.write(yuv_to_rgb(y, cb, cr), gid);
}

// Full-resolution YUV→BGRA conversion (used by MPS as intermediate step)
kernel void yuv_to_bgra(
    texture2d<float, access::sample> yTexture  [[texture(0)]],
    texture2d<float, access::sample> uvTexture [[texture(1)]],
    texture2d<float, access::write>  outTexture [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(outTexture.get_width(), outTexture.get_height());

    float y  = yTexture.sample(s, uv).r;
    float cb = uvTexture.sample(s, uv).r;
    float cr = uvTexture.sample(s, uv).g;

    outTexture.write(yuv_to_rgb(y, cb, cr), gid);
}

// ─── NCHW Direct Output kernels (zero-copy AI inference path) ─────
// Output: device float* buffer in NCHW layout [R_plane, G_plane, B_plane]
// Skips texture→CGImage→tensor conversion entirely

struct NCHWParams {
    float offsetX;
    float offsetY;
    float innerSizeX;
    float innerSizeY;
    int   width;
    int   height;
    float scaleX;
    float scaleY;
    float scaleZ;
    float biasX;
    float biasY;
    float biasZ;
};

kernel void downsample_yuv_bilinear_nchw(
    texture2d<float, access::sample> yTexture  [[texture(0)]],
    texture2d<float, access::sample> uvTexture [[texture(1)]],
    device float *outputBuffer                 [[buffer(0)]],
    constant NCHWParams &params                [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(params.width, params.height);

    float y  = yTexture.sample(s, uv).r;
    float cb = uvTexture.sample(s, uv).r;
    float cr = uvTexture.sample(s, uv).g;

    float4 rgb = yuv_to_rgb(y, cb, cr);

    int idx = gid.y * params.width + gid.x;
    int planeSize = params.width * params.height;
    outputBuffer[0 * planeSize + idx] = rgb.r * params.scaleX + params.biasX;
    outputBuffer[1 * planeSize + idx] = rgb.g * params.scaleY + params.biasY;
    outputBuffer[2 * planeSize + idx] = rgb.b * params.scaleZ + params.biasZ;
}

kernel void downsample_yuv_letterbox_nchw(
    texture2d<float, access::sample> yTexture  [[texture(0)]],
    texture2d<float, access::sample> uvTexture [[texture(1)]],
    device float *outputBuffer                 [[buffer(0)]],
    constant NCHWParams &params                [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    int idx = gid.y * params.width + gid.x;
    int planeSize = params.width * params.height;

    float2 pos = float2(gid) - float2(params.offsetX, params.offsetY);

    if (pos.x < 0.0 || pos.y < 0.0 || pos.x >= params.innerSizeX || pos.y >= params.innerSizeY) {
        float grayR = 0.5 * params.scaleX + params.biasX;
        float grayG = 0.5 * params.scaleY + params.biasY;
        float grayB = 0.5 * params.scaleZ + params.biasZ;
        outputBuffer[0 * planeSize + idx] = grayR;
        outputBuffer[1 * planeSize + idx] = grayG;
        outputBuffer[2 * planeSize + idx] = grayB;
        return;
    }

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (pos + 0.5) / float2(params.innerSizeX, params.innerSizeY);

    float y  = yTexture.sample(s, uv).r;
    float cb = uvTexture.sample(s, uv).r;
    float cr = uvTexture.sample(s, uv).g;

    float4 rgb = yuv_to_rgb(y, cb, cr);

    outputBuffer[0 * planeSize + idx] = rgb.r * params.scaleX + params.biasX;
    outputBuffer[1 * planeSize + idx] = rgb.g * params.scaleY + params.biasY;
    outputBuffer[2 * planeSize + idx] = rgb.b * params.scaleZ + params.biasZ;
}

kernel void downsample_bilinear_nchw(
    texture2d<float, access::sample> inTexture [[texture(0)]],
    device float *outputBuffer                 [[buffer(0)]],
    constant NCHWParams &params                [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(params.width, params.height);
    float4 color = inTexture.sample(s, uv);

    int idx = gid.y * params.width + gid.x;
    int planeSize = params.width * params.height;
    outputBuffer[0 * planeSize + idx] = color.r * params.scaleX + params.biasX;
    outputBuffer[1 * planeSize + idx] = color.g * params.scaleY + params.biasY;
    outputBuffer[2 * planeSize + idx] = color.b * params.scaleZ + params.biasZ;
}

kernel void downsample_letterbox_nchw(
    texture2d<float, access::sample> inTexture [[texture(0)]],
    device float *outputBuffer                 [[buffer(0)]],
    constant NCHWParams &params                [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    int idx = gid.y * params.width + gid.x;
    int planeSize = params.width * params.height;

    float2 pos = float2(gid) - float2(params.offsetX, params.offsetY);

    if (pos.x < 0.0 || pos.y < 0.0 || pos.x >= params.innerSizeX || pos.y >= params.innerSizeY) {
        float grayR = 0.5 * params.scaleX + params.biasX;
        float grayG = 0.5 * params.scaleY + params.biasY;
        float grayB = 0.5 * params.scaleZ + params.biasZ;
        outputBuffer[0 * planeSize + idx] = grayR;
        outputBuffer[1 * planeSize + idx] = grayG;
        outputBuffer[2 * planeSize + idx] = grayB;
        return;
    }

    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (pos + 0.5) / float2(params.innerSizeX, params.innerSizeY);
    float4 color = inTexture.sample(s, uv);

    outputBuffer[0 * planeSize + idx] = color.r * params.scaleX + params.biasX;
    outputBuffer[1 * planeSize + idx] = color.g * params.scaleY + params.biasY;
    outputBuffer[2 * planeSize + idx] = color.b * params.scaleZ + params.biasZ;
}

// ─── Utility kernels ───────────────────────────────────────────────

kernel void downsample_area_average(
    texture2d<float, access::read>  inTexture  [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    float scaleX = float(inTexture.get_width())  / float(outTexture.get_width());
    float scaleY = float(inTexture.get_height()) / float(outTexture.get_height());

    uint2 srcStart = uint2(float2(gid) * float2(scaleX, scaleY));
    uint2 srcEnd   = uint2(float2(gid + 1) * float2(scaleX, scaleY));
    srcEnd = min(srcEnd, uint2(inTexture.get_width(), inTexture.get_height()));

    float4 sum = float4(0.0);
    float count = 0.0;

    for (uint y = srcStart.y; y < srcEnd.y; y++) {
        for (uint x = srcStart.x; x < srcEnd.x; x++) {
            sum += inTexture.read(uint2(x, y));
            count += 1.0;
        }
    }

    outTexture.write(sum / max(count, 1.0), gid);
}
