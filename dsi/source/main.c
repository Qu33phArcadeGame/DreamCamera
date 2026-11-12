// DreamCamera DSi — v4
// Top screen: plain camera viewfinder (no markings, just what the camera sees).
// Bottom screen: settings, progress, gallery.
// A / SHOOT: take a photo and dream it (tiny on-device dream, 50 frames max).
// Then A = Continue (+50), B = back to camera, Y = gallery.
// Camera: libnds DSi camera API (nds/camera.h) — cameraInit / cameraSelect /
// cameraStartTransfer, preview 256x192 RGB. Sources:
//   blocksds/libnds "DSi Camera" docs (deepwiki) — REG_CAM_CNT 0x4004202,
//   cameraStartTransfer(buffer, MCUREG_APT_SEQ_CMD_PREVIEW, ndma)
//   snap-camera.neocities.org/dev — ARM7 I2C sensor setup, ARM9 NDMA capture
// v3 FIX (camera still black on real DSi, Nov 12 — v2's diagnosis was
// wrong): v2 waited for cameraTransferActive() to go false before reading
// a frame. It never does. From libnds source (include/nds/arm9/camera.h):
// cameraTransferActive() just reads the TRANSFER_ENABLE bit of REG_CAM_CNT,
// and cameraStartTransfer SETS that bit — nothing in the transfer path
// ever clears it. Only an explicit cameraStopTransfer() clears it. So v2's
// cam_poll() returned on every call, fr stayed 0, the top screen kept
// showing the boot test pattern (the rainbow rings in Coby's photo), and A
// dreamed the stale dark boot frame (black). The sensor and NDMA were fine
// all along: the buffer held a real frame at boot (nz 507). The ONLY true
// completion signal is the NDMA busy bit (ndmaBusy), which the hardware
// clears when the frame's words are written. v3 waits on ndmaBusy alone
// and calls cameraStopTransfer() after each finished frame to clear the
// latch. Bottom screen still shows live camera numbers (nz/max/fr) —
// fr climbing is the proof the viewfinder is live.
// v4 (real neural dream, Nov 12): the v1-v3 dream was a fixed 3x3
// edge-enhance kernel, not a neural net. v4 ports the Dreamlings DS tiny
// CNN in (cnn.c, verbatim math): 24x24 grayscale in, Conv 3x3 1->8, 8->16,
// 16->16, all ReLU, 16.16 fixed point; dreaming is gradient ascent on the
// best channel of the last layer. Per dream frame the 256x192 photo is
// downscaled to a 24x24 gray tile, dreamed 1-3 steps, then the dreamed tile
// is upscaled and applied as luminance over the photo's own colors. The
// zoom still runs every frame, so the CNN sees a closer and closer crop as
// the dream goes on. HONEST LIMIT: the CNN only ever sees 24x24, so the
// dream is soft/chunky at full screen — that is the tiny net's real size,
// not a bug. One CNN step is ~0.2s on DSi, so a 50-frame dream is ~15-20s;
// use Frames 10/25 for a faster one. Steps = CNN steps per frame (1-3),
// Dream strength = step size per iteration (Gentle 4 / Medium 8 / Wild 16).
// HONEST LIMITS (see README.md): the DSi cannot run Inception/VGG
// (12-29MB weights, 16MB RAM, 133MHz). The on-device dream is the Dreamlings
// DS tiny CNN (24x24, 3 conv layers, gradient ascent), not Inception —
// frames save to SD at 256x192. The full neural dream is the web app.
// If the camera does not answer (DS mode, emulator, no DSi), the viewfinder
// shows a labelled test pattern instead of pretending — the dream still
// works on it.
#include <nds.h>
#include <nds/camera.h>
#include <nds/ndma.h>
#include <nds/arm9/cache.h>
#include <fat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <dirent.h>
#include "cnn.h"

#define W 256
#define H 192
#define NPIX (W * H)
#define MAX_RUN 50
#define TOTAL_MAX 500

// ---------- state ----------
typedef enum { ST_VIEW, ST_DREAM, ST_DONE, ST_GALLERY } State;
static State state = ST_VIEW;
static bool isDSi = false, camOk = false, sdOk = false;
static int camSel = 1;              // 1 = outer, 0 = inner
static int dreamStr = 1;            // 0 gentle, 1 medium, 2 wild
static float zoomF = 0.97f;
static int pathSel = 0;             // 0 straight, 1 square, 2 triangle
static int framesWanted = 50;
static int cnnSteps = 1;            // CNN dream steps per frame (1-3)
static unsigned char cnnTile[24 * 24];  // 24x24 gray tile the CNN dreams on
static int cnnCh = 0;               // current best channel of the CNN's last layer
static int frameCount = 0, pathFrame = 0, dreamNo = 0;
static int galCount = 0, galIdx = 0;
static char galNames[64][16];
static float *img = NULL, *tmp = NULL;   // RGB float buffers, 256x192x3
static u16 camBufAligned[NPIX] __attribute__((aligned(32)));
static u16 *camBuf = NULL;                // camera preview buffer (32B-aligned: NDMA target, cache-invalidate needs it)
static u16 *dispBuf = NULL;               // top-screen framebuffer copy
static bool stopDream = false;
static bool transferPending = false;      // a preview transfer is in flight
static int camNz = 0, camMax = 0, camFrames = 0;  // live camera diagnostics

static const char *PATH_NAMES[3] = { "Straight", "Square", "Triangle" };
static const char *STR_NAMES[3] = { "Gentle", "Medium", "Wild" };
static const int STR_STEP[3] = { 4, 8, 16 };  // CNN max pixel change per iteration

// ---------- tiny dream ----------
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void rgb15_to_float(const u16 *src, float *dst) {
    for (int i = 0; i < NPIX; i++) {
        u16 p = src[i];
        dst[i * 3 + 0] = ((p >> 10) & 31) * (255.0f / 31.0f);
        dst[i * 3 + 1] = ((p >> 5) & 31) * (255.0f / 31.0f);
        dst[i * 3 + 2] = (p & 31) * (255.0f / 31.0f);
    }
}
static void float_to_rgb15(const float *src, u16 *dst) {
    for (int i = 0; i < NPIX; i++) {
        int r = (int)clampf(src[i * 3 + 0], 0, 255) >> 3;
        int g = (int)clampf(src[i * 3 + 1], 0, 255) >> 3;
        int b = (int)clampf(src[i * 3 + 2], 0, 255) >> 3;
        dst[i] = (u16)(r | (g << 5) | (b << 10) | 0x8000);
    }
}
// ---------- tiny-CNN dream (v4) ----------
// The CNN only sees a 24x24 grayscale tile. Per dream frame: downscale the
// full-res photo to the tile, run gradient ascent on it, upscale the dreamed
// tile and apply it as luminance over the photo's own colors.
static void downscale_to_tile(void) {
    for (int ty = 0; ty < 24; ty++) {
        int y0 = ty * H / 24, y1 = (ty + 1) * H / 24;
        for (int tx = 0; tx < 24; tx++) {
            int x0 = tx * W / 24, x1 = (tx + 1) * W / 24;
            float sum = 0; int n = 0;
            for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
                const float *p = &img[(y * W + x) * 3];
                sum += p[0] + p[1] + p[2]; n++;
            }
            float g = n ? sum / (3.0f * n) : 0;
            cnnTile[ty * 24 + tx] = (unsigned char)clampf(g, 0, 255);
        }
    }
}
// Bilinear sample of the dreamed tile at full-res pixel (x, y).
static float tile_sample(int x, int y) {
    float fx = (float)x * 23.0f / (W - 1), fy = (float)y * 23.0f / (H - 1);
    int x0 = (int)fx, y0 = (int)fy;
    if (x0 > 22) x0 = 22;
    if (y0 > 22) y0 = 22;
    float dx = fx - x0, dy = fy - y0;
    float v00 = cnnTile[y0 * 24 + x0], v10 = cnnTile[y0 * 24 + x0 + 1];
    float v01 = cnnTile[(y0 + 1) * 24 + x0], v11 = cnnTile[(y0 + 1) * 24 + x0 + 1];
    float top = v00 + (v10 - v00) * dx, bot = v01 + (v11 - v01) * dx;
    return top + (bot - top) * dy;
}
// Apply the dreamed tile as luminance: keep each pixel's hue, impose the
// dreamed brightness. Near-black pixels get the dreamed gray directly so
// the scale factor can't blow up.
static void cnn_luminance_apply(void) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float *p = &img[(y * W + x) * 3];
            float dreamed = tile_sample(x, y);
            float lum = (p[0] + p[1] + p[2]) / 3.0f;
            if (lum < 4.0f) {
                p[0] = p[1] = p[2] = dreamed;
            } else {
                float s = dreamed / lum;
                if (s > 4.0f) s = 4.0f;
                p[0] = clampf(p[0] * s, 0, 255);
                p[1] = clampf(p[1] * s, 0, 255);
                p[2] = clampf(p[2] * s, 0, 255);
            }
        }
    }
}
// One frame's CNN dream: downscale, gradient-ascent steps on the best
// channel, re-pick the best channel, apply as luminance.
static void cnn_dream_frame(void) {
    downscale_to_tile();
    for (int i = 0; i < cnnSteps; i++)
        cnn_dream_step(cnnTile, cnnCh, STR_STEP[dreamStr]);
    cnnCh = cnn_best_channel();
    cnn_luminance_apply();
}
// Zoom one frame: crop around a centre that walks the chosen path, scale back.
static void zoom_frame(float *im, float zf, int n) {
    int seg = 10, sub = n % seg;
    float rx = W * 0.08f / 2, ry = H * 0.08f / 2, ox = 0, oy = 0;
    if (pathSel == 1) {
        int ph = (n % (4 * seg)) / seg; float f = (float)sub / seg;
        if (ph == 0) { ox = -rx + 2 * rx * f; oy = -ry; }
        else if (ph == 1) { ox = rx; oy = -ry + 2 * ry * f; }
        else if (ph == 2) { ox = rx - 2 * rx * f; oy = ry; }
        else { ox = -rx; oy = ry - 2 * ry * f; }
    } else if (pathSel == 2) {
        int ph = (n % (3 * seg)) / seg; float f = (float)sub / seg;
        float vx[3] = { 0, rx * 0.866f, -rx * 0.866f }, vy[3] = { -ry, ry / 2, ry / 2 };
        ox = vx[ph] + (vx[(ph + 1) % 3] - vx[ph]) * f;
        oy = vy[ph] + (vy[(ph + 1) % 3] - vy[ph]) * f;
    }
    int cw = (int)(W * zf), ch = (int)(H * zf);
    int x0 = (W - cw) / 2 + (int)ox, y0 = (H - ch) / 2 + (int)oy;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 + cw > W) x0 = W - cw;
    if (y0 + ch > H) y0 = H - ch;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int sx = x0 + x * cw / W, sy = y0 + y * ch / H;
        for (int c = 0; c < 3; c++) tmp[(y * W + x) * 3 + c] = im[(sy * W + sx) * 3 + c];
    }
    memcpy(im, tmp, NPIX * 3 * sizeof(float));
}

// ---------- test pattern (no camera) ----------
static void test_pattern(float *dst, int t) {
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        float fx = (float)x / W, fy = (float)y / H;
        int cx = x - 128, cy = y - 96, r2 = cx * cx + cy * cy;
        float wave = 127 + 127 * sinf((r2 / 900.0f) - t * 0.15f);
        dst[(y * W + x) * 3 + 0] = clampf(fx * 255, 0, 255);
        dst[(y * W + x) * 3 + 1] = clampf(fy * 255, 0, 255);
        dst[(y * W + x) * 3 + 2] = wave;
    }
}

// ---------- SD save / gallery ----------
static void ensure_dir(const char *path) { mkdir(path, 0777); }
static void save_frame(int n) {
    if (!sdOk) return;
    char dir[64], fn[80];
    snprintf(dir, sizeof(dir), "/dreamcamera/dc%04d", dreamNo);
    ensure_dir("/dreamcamera"); ensure_dir(dir);
    float_to_rgb15(img, dispBuf);
    snprintf(fn, sizeof(fn), "%s/f%03d.bin", dir, n);
    FILE *f = fopen(fn, "wb");
    if (f) { fwrite(dispBuf, 2, NPIX, f); fclose(f); }
    if (n == 0) {
        snprintf(fn, sizeof(fn), "%s/info.txt", dir);
        f = fopen(fn, "w");
        if (f) { fprintf(f, "DreamCamera DSi dream %04d\nframes=%d\nstrength=%s\nzoom=%.3f\npath=%s\nsteps=%d\n", dreamNo, frameCount, STR_NAMES[dreamStr], zoomF, PATH_NAMES[pathSel], cnnSteps); fclose(f); }
    }
}
static void scan_gallery(void) {
    galCount = 0; galIdx = 0;
    if (!sdOk) return;
    DIR *d = opendir("/dreamcamera");
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) && galCount < 64) {
        if (strncmp(e->d_name, "dc", 2) == 0) { strncpy(galNames[galCount], e->d_name, 15); galNames[galCount][15] = 0; galCount++; }
    }
    closedir(d);
}
static int gallery_frames(const char *name) {
    char dir[64]; snprintf(dir, sizeof(dir), "/dreamcamera/%s", name);
    DIR *d = opendir(dir); if (!d) return 0;
    int n = 0; struct dirent *e;
    while ((e = readdir(d))) if (strncmp(e->d_name, "f", 1) == 0 && strstr(e->d_name, ".bin")) n++;
    closedir(d); return n;
}
static bool gallery_load(const char *name, int n) {
    char fn[80]; snprintf(fn, sizeof(fn), "/dreamcamera/%s/f%03d.bin", name, n);
    FILE *f = fopen(fn, "rb");
    if (!f) return false;
    size_t r = fread(dispBuf, 2, NPIX, f); fclose(f);
    return r == NPIX;
}

// ---------- screens ----------
static PrintConsole conSub;
static u16 *bgMain = NULL;

static void show_top(void) {
    if (!bgMain) return;
    memcpy(bgMain, dispBuf, NPIX * 2);
}
static void draw_ui(void) {
    consoleSelect(&conSub); consoleClear();
    printf("\x1b[2J");
    printf("DreamCamera DSi v4\n");
    printf("------------------\n");
    if (state == ST_VIEW) {
        printf("%s\n", camOk ? (camSel ? "Camera: OUTER" : "Camera: INNER") : "NO CAMERA: TEST PATTERN");
        printf("SD: %s\n", sdOk ? "ok" : "no card");
        printf("DSi:%d nz:%d max:%04x fr:%d\n", isDSi, camNz, camMax, camFrames);
        printf("\n");
        printf("A / SHOOT: take photo\n");
        printf("X: switch camera\n");
        printf("Y: gallery (%d dreams)\n", galCount);
        printf("Up/Down: setting\n");
        printf("Left/Right: change\n");
        printf("\n");
        printf("> Dream %s\n", STR_NAMES[dreamStr]);
        printf("> Zoom %.3f\n", zoomF);
        printf("> Path %s\n", PATH_NAMES[pathSel]);
        printf("> Frames %d\n", framesWanted);
        printf("> Steps %d\n", cnnSteps);
    } else if (state == ST_DREAM) {
        printf("Dreaming... %d / %d\n", frameCount, framesWanted);
        printf("frame %d total\n", frameCount);
        printf("\nB: stop\n");
    } else if (state == ST_DONE) {
        printf("Done: %d frames.\n", frameCount);
        printf("Saved to SD:\n/dreamcamera/dc%04d/\n", dreamNo);
        printf("\nA: Continue +50\n");
        printf("B: back to camera\n");
        printf("Y: gallery\n");
    } else if (state == ST_GALLERY) {
        printf("Gallery %d / %d\n", galCount ? galIdx + 1 : 0, galCount);
        if (galCount) {
            printf("%s  (%d frames)\n", galNames[galIdx], gallery_frames(galNames[galIdx]));
            printf("\nA: play  L/R: browse\n");
        } else printf("No saved dreams yet.\n");
        printf("B: back\n");
    }
}

// ---------- camera ----------
static void camera_start(void) {
    if (!isDSi) return;
    cameraInit();
    cameraSelect(camSel ? CAMERA_OUTER : CAMERA_INNER);
    memset(camBuf, 0, NPIX * 2);
    cameraStartTransfer(camBuf, MCUREG_APT_SEQ_CMD_PREVIEW, 3);
    transferPending = true;
}
static void camera_stop(void) { if (isDSi) { cameraStopTransfer(); transferPending = false; } }
// Sample the preview buffer: invalidate the ARM9 data cache first, because
// NDMA writes bypass the cache and a stale cache reads black.
static void cam_sample_stats(void) {
    DC_InvalidateRange(camBuf, NPIX * 2);
    int nz = 0, mx = 0;
    for (int i = 0; i < NPIX; i += 97) {
        u16 p = camBuf[i];
        if (p) nz++;
        if (p > mx) mx = p;
    }
    camNz = nz; camMax = mx;
}
// Has the preview buffer got real pixels yet?
static bool camera_alive(void) {
    cam_sample_stats();
    return camNz > 4;
}
// Bounded wait for the in-flight NDMA frame to finish. ndmaBusy reads
// bit 31 of the channel control register, which the hardware clears when
// the frame's words are written — it is the only true completion signal.
static void ndma_wait_idle(void) {
    int w = 0;
    while (ndmaBusy(3) && w++ < 120) swiWaitForVBlank();
}
// Preview pump: the preview is one captured frame, not a stream. When the
// NDMA has finished writing the frame, stop the transfer engine (this
// clears the TRANSFER_ENABLE latch — cameraTransferActive() never clears
// on its own), read the finished frame (cache-invalidate first), show it,
// then start the next transfer. Call every frame.
static void cam_poll(void) {
    if (!isDSi || !camOk) return;
    if (ndmaBusy(3)) return;   // frame still streaming in
    if (transferPending) {
        cameraStopTransfer();
        cam_sample_stats();
        rgb15_to_float(camBuf, img);
        float_to_rgb15(img, dispBuf);
        show_top();
        camFrames++;
        transferPending = false;
    }
    cameraStartTransfer(camBuf, MCUREG_APT_SEQ_CMD_PREVIEW, 3);
    transferPending = true;
}

// ---------- dream run ----------
static void run_dream(void) {
    state = ST_DREAM; stopDream = false; draw_ui();
    int want = framesWanted;
    if (frameCount + want > TOTAL_MAX) want = TOTAL_MAX - frameCount;
    for (int f = 0; f < want && !stopDream; f++) {
        pathFrame++;
        zoom_frame(img, zoomF, pathFrame);
        cnn_dream_frame();
        frameCount++;
        float_to_rgb15(img, dispBuf); show_top();
        save_frame(frameCount - 1);
        if (f % 5 == 0) draw_ui();
        swiWaitForVBlank();
    }
    state = ST_DONE; draw_ui();
}
static void shoot(void) {
    if (camOk) {
        // Freeze the preview on a COMPLETE fresh frame: let the in-flight
        // NDMA frame finish first, then stop the transfer engine and read
        // it (cache-invalidate first). Stopping first can strand the NDMA
        // mid-frame and leave a torn/dark capture; reading mid-transfer
        // tears, and reading a stale frame is black.
        ndma_wait_idle();
        camera_stop();
        cam_sample_stats();
        rgb15_to_float(camBuf, img);
    }
    else test_pattern(img, 0);
    // Pick the CNN's best channel from this photo (like the DS game does
    // from the selected dreamling) so the first frame dreams on it.
    downscale_to_tile();
    cnn_forward(cnnTile);
    cnnCh = cnn_best_channel();
    dreamNo++;
    // count existing dreams so numbers don't clash; skip taken numbers
    if (sdOk) {
        scan_gallery();
        dreamNo = galCount + 1;
        char probe[64]; DIR *pd;
        do { snprintf(probe, sizeof(probe), "/dreamcamera/dc%04d", dreamNo); pd = opendir(probe); if (pd) { closedir(pd); dreamNo++; } } while (pd);
    }
    frameCount = 0; pathFrame = 0;
    float_to_rgb15(img, dispBuf); show_top();
    run_dream();
}

// ---------- main ----------
int main(void) {
    videoSetMode(MODE_5_2D);
    videoSetModeSub(MODE_0_2D);
    vramSetBankA(VRAM_A_MAIN_BG);
    vramSetBankC(VRAM_C_SUB_BG);
    int bg = bgInit(3, BgType_Bmp16, BgSize_B16_256x256, 0, 0);
    bgMain = (u16 *)bgGetGfxPtr(bg);
    consoleInit(&conSub, 0, BgType_Text4bpp, BgSize_T_256x256, 31, 0, false, true);
    consoleSelect(&conSub);

    img = (float *)malloc(NPIX * 3 * sizeof(float));
    tmp = (float *)malloc(NPIX * 3 * sizeof(float));
    camBuf = camBufAligned;
    dispBuf = (u16 *)malloc(NPIX * 2);
    memset(dispBuf, 0, NPIX * 2);

    isDSi = isDSiMode();
    sdOk = fatInitDefault();
    if (sdOk) { ensure_dir("/dreamcamera"); scan_gallery(); }

    if (isDSi) {
        camera_start();
        // give the sensor a moment, then wait for the first preview
        // transfer to finish (bounded, NDMA busy bit only) before
        // checking for real pixels. Then clear the transfer-enable latch
        // so cam_poll can cycle transfers from a clean state.
        for (int i = 0; i < 30; i++) swiWaitForVBlank();
        ndma_wait_idle();
        cameraStopTransfer();
        camOk = camera_alive();
        if (!camOk) camera_stop();
    }
    // No camera: show the labelled test pattern. Camera OK: leave the top
    // to cam_poll, which shows the first live frame on the first loop.
    if (!camOk) { test_pattern(img, 0); float_to_rgb15(img, dispBuf); show_top(); }
    draw_ui();

    int selRow = 0, t = 0, uiTick = 0;
    for (;;) {
        swiWaitForVBlank();
        scanKeys();
        u32 down = keysDown();
        touchPosition touch; touchRead(&touch);

        if (state == ST_VIEW) {
            if (camOk) { cam_poll(); }
            else { test_pattern(img, t++); float_to_rgb15(img, dispBuf); show_top(); }
            // refresh the live camera numbers a couple of times a second
            if ((++uiTick % 30) == 0) draw_ui();
            if (down & KEY_A) shoot();
            if (down & KEY_X) {
                camSel = !camSel;
                if (camOk) {
                    ndma_wait_idle();
                    camera_stop(); camera_start();
                    for (int i = 0; i < 30; i++) swiWaitForVBlank();
                    ndma_wait_idle();
                    cameraStopTransfer();
                    camOk = camera_alive();
                    if (!camOk) camera_stop();
                }
                draw_ui();
            }
            if (down & KEY_Y) { state = ST_GALLERY; scan_gallery(); draw_ui(); }
            if (down & KEY_UP) { selRow = (selRow + 4) % 5; draw_ui(); }
            if (down & KEY_DOWN) { selRow = (selRow + 1) % 5; draw_ui(); }
            if (down & KEY_LEFT || down & KEY_RIGHT) {
                int d = (down & KEY_RIGHT) ? 1 : -1;
                if (selRow == 0) { dreamStr = (dreamStr + d + 3) % 3; }
                else if (selRow == 1) { float z[3] = { 0.99f, 0.97f, 0.95f }; int cur = (zoomF > 0.98f) ? 0 : (zoomF > 0.96f) ? 1 : 2; cur = (cur + d + 3) % 3; zoomF = z[cur]; }
                else if (selRow == 2) { pathSel = (pathSel + d + 3) % 3; }
                else if (selRow == 3) { int f[3] = { 10, 25, 50 }; int cur = (framesWanted <= 10) ? 0 : (framesWanted <= 25) ? 1 : 2; cur = (cur + d + 3) % 3; framesWanted = f[cur]; }
                else { cnnSteps = ((cnnSteps - 1 + d + 3) % 3) + 1; }
                draw_ui();
            }
            // touch: bottom half = shoot, top strip = switch camera
            if (keysDown() & KEY_TOUCH) {
                if (touch.py > 120) shoot();
                else if (touch.py < 40) { camSel = !camSel; if (camOk) { ndma_wait_idle(); camera_stop(); camera_start(); } draw_ui(); }
            }
        } else if (state == ST_DREAM) {
            if (down & KEY_B) stopDream = true;
        } else if (state == ST_DONE) {
            if (down & KEY_A) { if (frameCount < TOTAL_MAX) run_dream(); }
            if (down & KEY_B) { state = ST_VIEW; draw_ui(); }
            if (down & KEY_Y) { state = ST_GALLERY; scan_gallery(); draw_ui(); }
        } else if (state == ST_GALLERY) {
            if (down & KEY_B) { state = ST_VIEW; draw_ui(); }
            if (down & KEY_LEFT && galCount) { galIdx = (galIdx + galCount - 1) % galCount; draw_ui(); }
            if (down & KEY_RIGHT && galCount) { galIdx = (galIdx + 1) % galCount; draw_ui(); }
            if ((down & KEY_A) && galCount) {
                int n = gallery_frames(galNames[galIdx]);
                for (int i = 0; i < n; i++) {
                    if (gallery_load(galNames[galIdx], i)) show_top();
                    for (int v = 0; v < 7; v++) swiWaitForVBlank();
                    scanKeys(); if (keysDown() & KEY_B) break;
                }
                draw_ui();
            }
        }
    }
    return 0;
}
