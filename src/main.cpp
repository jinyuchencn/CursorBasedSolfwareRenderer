#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;

    Vec4() = default;
    Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

static inline Vec2 operator+(const Vec2& a, const Vec2& b) { return Vec2(a.x + b.x, a.y + b.y); }
static inline Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2(a.x - b.x, a.y - b.y); }
static inline Vec2 operator*(const Vec2& a, float s) { return Vec2(a.x * s, a.y * s); }
static inline Vec2 operator/(const Vec2& a, float s) { return Vec2(a.x / s, a.y / s); }

static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 operator-(const Vec3& v) { return Vec3(-v.x, -v.y, -v.z); }
static inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
static inline Vec3 operator*(float s, const Vec3& a) { return Vec3(a.x * s, a.y * s, a.z * s); }
static inline Vec3 operator/(const Vec3& a, float s) { return Vec3(a.x / s, a.y / s, a.z / s); }
static inline Vec3 operator*(const Vec3& a, const Vec3& b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }

static inline Vec4 operator+(const Vec4& a, const Vec4& b) { return Vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static inline Vec4 operator-(const Vec4& a, const Vec4& b) { return Vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static inline Vec4 operator*(const Vec4& a, float s) { return Vec4(a.x * s, a.y * s, a.z * s, a.w * s); }

static inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
static inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
static inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    return (len > 0.0f) ? (v / len) : Vec3(0.0f, 0.0f, 0.0f);
}
static inline float saturate(float v) { return std::min(1.0f, std::max(0.0f, v)); }
static inline Vec3 clamp01(const Vec3& v) { return Vec3(saturate(v.x), saturate(v.y), saturate(v.z)); }

struct Mat4 {
    float m[4][4]{};

    static Mat4 identity() {
        Mat4 r{};
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
        return r;
    }

    static Mat4 translation(const Vec3& t) {
        Mat4 r = identity();
        r.m[0][3] = t.x;
        r.m[1][3] = t.y;
        r.m[2][3] = t.z;
        return r;
    }

    static Mat4 scale(const Vec3& s) {
        Mat4 r{};
        r.m[0][0] = s.x;
        r.m[1][1] = s.y;
        r.m[2][2] = s.z;
        r.m[3][3] = 1.0f;
        return r;
    }

    static Mat4 rotationY(float radians) {
        Mat4 r = identity();
        float c = std::cos(radians);
        float s = std::sin(radians);
        r.m[0][0] = c;
        r.m[0][2] = s;
        r.m[2][0] = -s;
        r.m[2][2] = c;
        return r;
    }

    static Mat4 rotationX(float radians) {
        Mat4 r = identity();
        float c = std::cos(radians);
        float s = std::sin(radians);
        r.m[1][1] = c;
        r.m[1][2] = -s;
        r.m[2][1] = s;
        r.m[2][2] = c;
        return r;
    }

    static Mat4 perspective(float fovY, float aspect, float zNear, float zFar) {
        Mat4 r{};
        float f = 1.0f / std::tan(fovY * 0.5f);
        r.m[0][0] = f / aspect;
        r.m[1][1] = f;
        r.m[2][2] = zFar / (zFar - zNear);
        r.m[2][3] = (-zFar * zNear) / (zFar - zNear);
        r.m[3][2] = 1.0f;
        return r;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        Vec3 f = normalize(target - eye);
        Vec3 s = normalize(cross(f, up));
        Vec3 u = cross(s, f);

        Mat4 r = identity();
        r.m[0][0] = s.x;
        r.m[0][1] = s.y;
        r.m[0][2] = s.z;
        r.m[1][0] = u.x;
        r.m[1][1] = u.y;
        r.m[1][2] = u.z;
        r.m[2][0] = f.x;
        r.m[2][1] = f.y;
        r.m[2][2] = f.z;
        r.m[0][3] = -dot(s, eye);
        r.m[1][3] = -dot(u, eye);
        r.m[2][3] = -dot(f, eye);
        return r;
    }
};

static inline Vec4 operator*(const Mat4& a, const Vec4& v) {
    Vec4 r{};
    r.x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w;
    r.y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w;
    r.z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w;
    r.w = a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w;
    return r;
}

static inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[i][k] * b.m[k][j];
            }
            r.m[i][j] = sum;
        }
    }
    return r;
}

static inline Vec3 transformDirection(const Mat4& m, const Vec3& v) {
    return Vec3(
        m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
        m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
        m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z
    );
}

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    Vec3 tangent;
};

struct VSOut {
    Vec4 clip;
    Vec3 worldPos;
    Vec3 normal;
    Vec2 uv;
    Vec3 tangent;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<int> indices;
};

struct Material {
    Vec3 baseColor;
    float metallic = 0.0f;
    float roughness = 0.5f;
    float normalScale = 1.0f;
};

struct Renderable {
    Mesh mesh;
    Mat4 model;
    Material material;
};

struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    float fovY = 60.0f * kPi / 180.0f;
    float zNear = 0.1f;
    float zFar = 100.0f;
};

struct DirectionalLight {
    Vec3 direction;
    Vec3 color;
    float intensity = 2.0f;
};

struct PointLight {
    Vec3 position;
    Vec3 color;
    float intensity = 40.0f;
};

struct RenderTarget {
    int width = 0;
    int height = 0;
    std::vector<Vec3> color;
    std::vector<float> depth;

    RenderTarget(int w, int h) : width(w), height(h), color(w * h), depth(w * h) {}

    void clear(const Vec3& c) {
        std::fill(color.begin(), color.end(), c);
        std::fill(depth.begin(), depth.end(), std::numeric_limits<float>::infinity());
    }
};

struct ShadowMap {
    int width = 0;
    int height = 0;
    std::vector<float> depth;

    ShadowMap(int w, int h) : width(w), height(h), depth(w * h) {}

    void clear() {
        std::fill(depth.begin(), depth.end(), std::numeric_limits<float>::infinity());
    }
};

static inline Vec3 toSRGB(const Vec3& linear) {
    auto convert = [](float c) {
        return std::pow(saturate(c), 1.0f / 2.2f);
    };
    return Vec3(convert(linear.x), convert(linear.y), convert(linear.z));
}

static inline Vec3 toneMapReinhard(const Vec3& c) {
    return Vec3(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
}

static inline float edgeFunction(const Vec2& a, const Vec2& b, const Vec2& c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

static inline Vec3 sampleChecker(const Vec2& uv) {
    int x = static_cast<int>(std::floor(uv.x * 8.0f));
    int y = static_cast<int>(std::floor(uv.y * 8.0f));
    int check = (x + y) & 1;
    Vec3 a(0.8f, 0.2f, 0.15f);
    Vec3 b(0.15f, 0.6f, 0.85f);
    return check ? a : b;
}

static inline Vec3 sampleNormalMap(const Vec2& uv) {
    float sx = std::sin(uv.x * kPi * 4.0f);
    float sy = std::sin(uv.y * kPi * 4.0f);
    Vec3 n(sx, sy, 1.0f);
    return normalize(n);
}

static inline float distributionGGX(const Vec3& n, const Vec3& h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = std::max(dot(n, h), 0.0f);
    float nDotH2 = nDotH * nDotH;
    float denom = (nDotH2 * (a2 - 1.0f) + 1.0f);
    return a2 / (kPi * denom * denom + 1e-6f);
}

static inline float geometrySchlickGGX(float nDotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    float denom = nDotV * (1.0f - k) + k;
    return nDotV / (denom + 1e-6f);
}

static inline float geometrySmith(const Vec3& n, const Vec3& v, const Vec3& l, float roughness) {
    float nDotV = std::max(dot(n, v), 0.0f);
    float nDotL = std::max(dot(n, l), 0.0f);
    float ggxV = geometrySchlickGGX(nDotV, roughness);
    float ggxL = geometrySchlickGGX(nDotL, roughness);
    return ggxV * ggxL;
}

static inline Vec3 fresnelSchlick(float cosTheta, const Vec3& f0) {
    float pow5 = std::pow(1.0f - cosTheta, 5.0f);
    return f0 + (Vec3(1.0f, 1.0f, 1.0f) - f0) * pow5;
}

static Mesh makeSphere(int slices, int stacks, float radius) {
    Mesh mesh;
    for (int y = 0; y <= stacks; ++y) {
        float v = static_cast<float>(y) / static_cast<float>(stacks);
        float phi = v * kPi;
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        for (int x = 0; x <= slices; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(slices);
            float theta = u * kPi * 2.0f;
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            Vec3 pos(radius * sinPhi * cosTheta, radius * cosPhi, radius * sinPhi * sinTheta);
            Vec3 normal = normalize(pos);
            Vec3 tangent(-sinTheta, 0.0f, cosTheta);
            mesh.vertices.push_back(Vertex{pos, normal, Vec2(u, v), tangent});
        }
    }

    int stride = slices + 1;
    for (int y = 0; y < stacks; ++y) {
        for (int x = 0; x < slices; ++x) {
            int i0 = y * stride + x;
            int i1 = i0 + 1;
            int i2 = i0 + stride;
            int i3 = i2 + 1;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }
    return mesh;
}

static Mesh makePlane(float size) {
    Mesh mesh;
    float h = size * 0.5f;
    mesh.vertices = {
        {{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{ h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{-h, 0.0f,  h}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
        {{ h, 0.0f,  h}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    };
    mesh.indices = {0, 2, 1, 1, 2, 3};
    return mesh;
}

static Vec3 sampleShadow(const ShadowMap& shadowMap, const Vec4& lightClip, float bias) {
    if (lightClip.w <= 0.0f) {
        return Vec3(1.0f, 1.0f, 1.0f);
    }
    Vec3 ndc(lightClip.x / lightClip.w, lightClip.y / lightClip.w, lightClip.z / lightClip.w);
    float u = ndc.x * 0.5f + 0.5f;
    float v = 1.0f - (ndc.y * 0.5f + 0.5f);
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return Vec3(1.0f, 1.0f, 1.0f);
    }
    float depth = ndc.z;
    int x = static_cast<int>(u * static_cast<float>(shadowMap.width - 1));
    int y = static_cast<int>(v * static_cast<float>(shadowMap.height - 1));
    float shadow = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = std::min(shadowMap.width - 1, std::max(0, x + dx));
            int sy = std::min(shadowMap.height - 1, std::max(0, y + dy));
            float mapDepth = shadowMap.depth[sy * shadowMap.width + sx];
            shadow += (depth - bias > mapDepth) ? 0.0f : 1.0f;
            ++count;
        }
    }
    float factor = shadow / static_cast<float>(count);
    return Vec3(factor, factor, factor);
}

static void rasterizeTriangle(
    const VSOut& v0,
    const VSOut& v1,
    const VSOut& v2,
    RenderTarget& target,
    const std::function<Vec3(const VSOut&, float)>& shadePixel)
{
    if (v0.clip.w <= 0.0f || v1.clip.w <= 0.0f || v2.clip.w <= 0.0f) {
        return;
    }

    Vec3 p0(v0.clip.x / v0.clip.w, v0.clip.y / v0.clip.w, v0.clip.z / v0.clip.w);
    Vec3 p1(v1.clip.x / v1.clip.w, v1.clip.y / v1.clip.w, v1.clip.z / v1.clip.w);
    Vec3 p2(v2.clip.x / v2.clip.w, v2.clip.y / v2.clip.w, v2.clip.z / v2.clip.w);

    Vec2 s0((p0.x * 0.5f + 0.5f) * (target.width - 1), (1.0f - (p0.y * 0.5f + 0.5f)) * (target.height - 1));
    Vec2 s1((p1.x * 0.5f + 0.5f) * (target.width - 1), (1.0f - (p1.y * 0.5f + 0.5f)) * (target.height - 1));
    Vec2 s2((p2.x * 0.5f + 0.5f) * (target.width - 1), (1.0f - (p2.y * 0.5f + 0.5f)) * (target.height - 1));

    float area = edgeFunction(s0, s1, s2);
    if (area >= 0.0f) {
        return;
    }

    float invW0 = 1.0f / v0.clip.w;
    float invW1 = 1.0f / v1.clip.w;
    float invW2 = 1.0f / v2.clip.w;

    int minX = static_cast<int>(std::floor(std::min({s0.x, s1.x, s2.x})));
    int maxX = static_cast<int>(std::ceil(std::max({s0.x, s1.x, s2.x})));
    int minY = static_cast<int>(std::floor(std::min({s0.y, s1.y, s2.y})));
    int maxY = static_cast<int>(std::ceil(std::max({s0.y, s1.y, s2.y})));

    minX = std::max(0, minX);
    minY = std::max(0, minY);
    maxX = std::min(target.width - 1, maxX);
    maxY = std::min(target.height - 1, maxY);

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Vec2 p(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
            float w0 = edgeFunction(s1, s2, p);
            float w1 = edgeFunction(s2, s0, p);
            float w2 = edgeFunction(s0, s1, p);
            if (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f) {
                w0 /= area;
                w1 /= area;
                w2 /= area;

                float invW = w0 * invW0 + w1 * invW1 + w2 * invW2;
                float z = (w0 * p0.z * invW0 + w1 * p1.z * invW1 + w2 * p2.z * invW2) / invW;

                int idx = y * target.width + x;
                if (z < target.depth[idx]) {
                    target.depth[idx] = z;

                    VSOut interp{};
                    interp.worldPos = (v0.worldPos * (w0 * invW0) + v1.worldPos * (w1 * invW1) + v2.worldPos * (w2 * invW2)) / invW;
                    interp.normal = normalize((v0.normal * (w0 * invW0) + v1.normal * (w1 * invW1) + v2.normal * (w2 * invW2)) / invW);
                    interp.uv = (v0.uv * (w0 * invW0) + v1.uv * (w1 * invW1) + v2.uv * (w2 * invW2)) / invW;
                    interp.tangent = normalize((v0.tangent * (w0 * invW0) + v1.tangent * (w1 * invW1) + v2.tangent * (w2 * invW2)) / invW);

                    target.color[idx] = shadePixel(interp, z);
                }
            }
        }
    }
}

static void rasterizeTriangleDepth(
    const VSOut& v0,
    const VSOut& v1,
    const VSOut& v2,
    ShadowMap& shadowMap)
{
    if (v0.clip.w <= 0.0f || v1.clip.w <= 0.0f || v2.clip.w <= 0.0f) {
        return;
    }

    Vec3 p0(v0.clip.x / v0.clip.w, v0.clip.y / v0.clip.w, v0.clip.z / v0.clip.w);
    Vec3 p1(v1.clip.x / v1.clip.w, v1.clip.y / v1.clip.w, v1.clip.z / v1.clip.w);
    Vec3 p2(v2.clip.x / v2.clip.w, v2.clip.y / v2.clip.w, v2.clip.z / v2.clip.w);

    Vec2 s0((p0.x * 0.5f + 0.5f) * (shadowMap.width - 1), (1.0f - (p0.y * 0.5f + 0.5f)) * (shadowMap.height - 1));
    Vec2 s1((p1.x * 0.5f + 0.5f) * (shadowMap.width - 1), (1.0f - (p1.y * 0.5f + 0.5f)) * (shadowMap.height - 1));
    Vec2 s2((p2.x * 0.5f + 0.5f) * (shadowMap.width - 1), (1.0f - (p2.y * 0.5f + 0.5f)) * (shadowMap.height - 1));

    float area = edgeFunction(s0, s1, s2);
    if (area >= 0.0f) {
        return;
    }

    float invW0 = 1.0f / v0.clip.w;
    float invW1 = 1.0f / v1.clip.w;
    float invW2 = 1.0f / v2.clip.w;

    int minX = static_cast<int>(std::floor(std::min({s0.x, s1.x, s2.x})));
    int maxX = static_cast<int>(std::ceil(std::max({s0.x, s1.x, s2.x})));
    int minY = static_cast<int>(std::floor(std::min({s0.y, s1.y, s2.y})));
    int maxY = static_cast<int>(std::ceil(std::max({s0.y, s1.y, s2.y})));

    minX = std::max(0, minX);
    minY = std::max(0, minY);
    maxX = std::min(shadowMap.width - 1, maxX);
    maxY = std::min(shadowMap.height - 1, maxY);

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Vec2 p(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
            float w0 = edgeFunction(s1, s2, p);
            float w1 = edgeFunction(s2, s0, p);
            float w2 = edgeFunction(s0, s1, p);
            if (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f) {
                w0 /= area;
                w1 /= area;
                w2 /= area;

                float invW = w0 * invW0 + w1 * invW1 + w2 * invW2;
                float z = (w0 * p0.z * invW0 + w1 * p1.z * invW1 + w2 * p2.z * invW2) / invW;

                int idx = y * shadowMap.width + x;
                if (z < shadowMap.depth[idx]) {
                    shadowMap.depth[idx] = z;
                }
            }
        }
    }
}

static void writePPM(const std::string& path, const RenderTarget& target) {
    std::ofstream ofs(path, std::ios::binary);
    ofs << "P6\n" << target.width << " " << target.height << "\n255\n";
    for (const Vec3& c : target.color) {
        Vec3 srgb = toSRGB(clamp01(c));
        uint8_t r = static_cast<uint8_t>(std::round(srgb.x * 255.0f));
        uint8_t g = static_cast<uint8_t>(std::round(srgb.y * 255.0f));
        uint8_t b = static_cast<uint8_t>(std::round(srgb.z * 255.0f));
        ofs.write(reinterpret_cast<const char*>(&r), 1);
        ofs.write(reinterpret_cast<const char*>(&g), 1);
        ofs.write(reinterpret_cast<const char*>(&b), 1);
    }
    ofs.close();
}

int main() {
    const int width = 800;
    const int height = 600;
    RenderTarget target(width, height);
    ShadowMap shadowMap(1024, 1024);

    Camera camera;
    camera.position = Vec3(0.0f, 2.5f, -6.0f);
    camera.target = Vec3(0.0f, 1.0f, 0.0f);
    camera.up = Vec3(0.0f, 1.0f, 0.0f);

    DirectionalLight dirLight;
    dirLight.direction = normalize(Vec3(-0.6f, -1.0f, -0.3f));
    dirLight.color = Vec3(1.0f, 0.98f, 0.9f);
    dirLight.intensity = 4.0f;

    PointLight pointA{Vec3(2.5f, 2.0f, -1.5f), Vec3(1.0f, 0.4f, 0.3f), 60.0f};
    PointLight pointB{Vec3(-2.0f, 3.0f, 2.0f), Vec3(0.2f, 0.6f, 1.0f), 50.0f};

    Renderable sphere;
    sphere.mesh = makeSphere(64, 32, 1.0f);
    sphere.model = Mat4::translation(Vec3(0.0f, 1.2f, 0.0f)) * Mat4::rotationY(0.6f);
    sphere.material.baseColor = Vec3(0.8f, 0.8f, 0.8f);
    sphere.material.metallic = 0.4f;
    sphere.material.roughness = 0.35f;
    sphere.material.normalScale = 1.0f;

    Renderable floor;
    floor.mesh = makePlane(8.0f);
    floor.model = Mat4::identity();
    floor.material.baseColor = Vec3(0.75f, 0.75f, 0.75f);
    floor.material.metallic = 0.0f;
    floor.material.roughness = 0.9f;
    floor.material.normalScale = 0.6f;

    std::vector<Renderable> renderables = {sphere, floor};

    Mat4 view = Mat4::lookAt(camera.position, camera.target, camera.up);
    Mat4 proj = Mat4::perspective(camera.fovY, static_cast<float>(width) / static_cast<float>(height), camera.zNear, camera.zFar);
    Mat4 viewProj = proj * view;

    Vec3 lightPos = camera.target - dirLight.direction * 10.0f;
    Mat4 lightView = Mat4::lookAt(lightPos, camera.target, Vec3(0.0f, 1.0f, 0.0f));
    Mat4 lightProj = Mat4::perspective(50.0f * kPi / 180.0f, 1.0f, 0.1f, 30.0f);
    Mat4 lightViewProj = lightProj * lightView;

    shadowMap.clear();
    for (const auto& renderable : renderables) {
        for (size_t i = 0; i < renderable.mesh.indices.size(); i += 3) {
            const Vertex& a = renderable.mesh.vertices[renderable.mesh.indices[i]];
            const Vertex& b = renderable.mesh.vertices[renderable.mesh.indices[i + 1]];
            const Vertex& c = renderable.mesh.vertices[renderable.mesh.indices[i + 2]];

            VSOut v0{};
            VSOut v1{};
            VSOut v2{};

            Vec4 wa(a.position.x, a.position.y, a.position.z, 1.0f);
            Vec4 wb(b.position.x, b.position.y, b.position.z, 1.0f);
            Vec4 wc(c.position.x, c.position.y, c.position.z, 1.0f);
            Vec4 w0 = renderable.model * wa;
            Vec4 w1 = renderable.model * wb;
            Vec4 w2 = renderable.model * wc;

            v0.clip = lightViewProj * w0;
            v1.clip = lightViewProj * w1;
            v2.clip = lightViewProj * w2;

            rasterizeTriangleDepth(v0, v1, v2, shadowMap);
        }
    }

    target.clear(Vec3(0.05f, 0.06f, 0.08f));

    for (const auto& renderable : renderables) {
        Mat4 model = renderable.model;
        Mat4 modelViewProj = viewProj * model;

        for (size_t i = 0; i < renderable.mesh.indices.size(); i += 3) {
            const Vertex& a = renderable.mesh.vertices[renderable.mesh.indices[i]];
            const Vertex& b = renderable.mesh.vertices[renderable.mesh.indices[i + 1]];
            const Vertex& c = renderable.mesh.vertices[renderable.mesh.indices[i + 2]];

            VSOut v0{};
            VSOut v1{};
            VSOut v2{};

            Vec4 wa(a.position.x, a.position.y, a.position.z, 1.0f);
            Vec4 wb(b.position.x, b.position.y, b.position.z, 1.0f);
            Vec4 wc(c.position.x, c.position.y, c.position.z, 1.0f);

            Vec4 worldA = model * wa;
            Vec4 worldB = model * wb;
            Vec4 worldC = model * wc;

            v0.clip = modelViewProj * wa;
            v1.clip = modelViewProj * wb;
            v2.clip = modelViewProj * wc;

            v0.worldPos = Vec3(worldA.x, worldA.y, worldA.z);
            v1.worldPos = Vec3(worldB.x, worldB.y, worldB.z);
            v2.worldPos = Vec3(worldC.x, worldC.y, worldC.z);

            v0.normal = normalize(transformDirection(model, a.normal));
            v1.normal = normalize(transformDirection(model, b.normal));
            v2.normal = normalize(transformDirection(model, c.normal));

            v0.tangent = normalize(transformDirection(model, a.tangent));
            v1.tangent = normalize(transformDirection(model, b.tangent));
            v2.tangent = normalize(transformDirection(model, c.tangent));

            v0.uv = a.uv;
            v1.uv = b.uv;
            v2.uv = c.uv;

            auto shade = [&](const VSOut& input, float depth) -> Vec3 {
                Vec3 N = normalize(input.normal);
                Vec3 T = normalize(input.tangent);
                Vec3 B = normalize(cross(N, T));

                Vec3 normalSample = sampleNormalMap(input.uv) * renderable.material.normalScale;
                Vec3 mapped = normalize(Vec3(
                    T.x * normalSample.x + B.x * normalSample.y + N.x * normalSample.z,
                    T.y * normalSample.x + B.y * normalSample.y + N.y * normalSample.z,
                    T.z * normalSample.x + B.z * normalSample.y + N.z * normalSample.z
                ));

                Vec3 albedo = sampleChecker(input.uv) * renderable.material.baseColor;
                float metallic = renderable.material.metallic;
                float roughness = std::max(0.05f, renderable.material.roughness);

                Vec3 V = normalize(camera.position - input.worldPos);
                Vec3 F0 = renderable.material.baseColor * metallic + Vec3(0.04f, 0.04f, 0.04f) * (1.0f - metallic);

                Vec3 Lo(0.0f, 0.0f, 0.0f);

                Vec3 Ld = normalize(-dirLight.direction);
                Vec3 H = normalize(V + Ld);
                float NDF = distributionGGX(mapped, H, roughness);
                float G = geometrySmith(mapped, V, Ld, roughness);
                Vec3 F = fresnelSchlick(std::max(dot(H, V), 0.0f), F0);

                Vec3 kS = F;
                Vec3 kD = (Vec3(1.0f, 1.0f, 1.0f) - kS) * (1.0f - metallic);

                float NdotL = std::max(dot(mapped, Ld), 0.0f);
                float denom = 4.0f * std::max(dot(mapped, V), 0.0f) * NdotL + 1e-4f;
                Vec3 specular = (NDF * G) * F / denom;
                Vec3 diffuse = kD * albedo / kPi;

                Vec4 lightClip = lightViewProj * Vec4(input.worldPos.x, input.worldPos.y, input.worldPos.z, 1.0f);
                Vec3 shadow = sampleShadow(shadowMap, lightClip, 0.002f);

                Vec3 radiance = dirLight.color * dirLight.intensity;
                Lo = Lo + (diffuse + specular) * radiance * NdotL * shadow.x;

                auto addPoint = [&](const PointLight& light) {
                    Vec3 L = light.position - input.worldPos;
                    float dist2 = std::max(dot(L, L), 0.01f);
                    L = normalize(L);
                    Vec3 Hp = normalize(V + L);
                    float NDFp = distributionGGX(mapped, Hp, roughness);
                    float Gp = geometrySmith(mapped, V, L, roughness);
                    Vec3 Fp = fresnelSchlick(std::max(dot(Hp, V), 0.0f), F0);
                    Vec3 kSp = Fp;
                    Vec3 kDp = (Vec3(1.0f, 1.0f, 1.0f) - kSp) * (1.0f - metallic);
                    float NdotLp = std::max(dot(mapped, L), 0.0f);
                    float denomP = 4.0f * std::max(dot(mapped, V), 0.0f) * NdotLp + 1e-4f;
                    Vec3 specP = (NDFp * Gp) * Fp / denomP;
                    Vec3 diffP = kDp * albedo / kPi;
                    Vec3 radianceP = light.color * (light.intensity / dist2);
                    Lo = Lo + (diffP + specP) * radianceP * NdotLp;
                };

                addPoint(pointA);
                addPoint(pointB);

                Vec3 ambient = albedo * 0.03f;
                Vec3 color = ambient + Lo;
                color = toneMapReinhard(color);
                return color;
            };

            rasterizeTriangle(v0, v1, v2, target, shade);
        }
    }

    writePPM("output.ppm", target);
    std::cout << "Rendered output.ppm\n";
    return 0;
}

