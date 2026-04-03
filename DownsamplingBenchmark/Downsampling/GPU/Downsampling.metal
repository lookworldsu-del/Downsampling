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
