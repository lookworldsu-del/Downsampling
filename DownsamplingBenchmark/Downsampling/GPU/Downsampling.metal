#include <metal_stdlib>
using namespace metal;

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
