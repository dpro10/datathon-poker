/*
 * solver.c - card-abstracted CFR for heads-up short-deck 5-card PLO (the datathon rules).
 *
 *   ./solver features <flop|turn> <n> <out.f32>        8-bin final-strength histograms, for k-means
 *   ./solver pool <n> <out.bin> <flop_centroids.f32> <B1> <turn_centroids.f32> <B2> <B3>
 *   ./solver train <pool.bin> <B0> <B1> <B2> <B3> <seconds> <threads> <out_prefix> [resume_prefix]
 *
 * Hand evaluation uses the same rank tables as sdbot/evaluator.py (exported by export.py), so
 * every score is identical to the official omaha_score ordering.
 *
 * Bucket features (identical definitions are computed by the bot at runtime):
 *   preflop  cluster of the hand's equity profile (pre.bin)
 *   flop     histogram (8 bins) over random turn+river runouts of HS = the hand's percentile among
 *            random opponent hands on the final board; nearest centroid
 *   turn     the same over every possible river card
 *   river    HS itself, B3 equal-width bins
 *
 * Training: external-sampling MCCFR (Lanctot et al. 2009) over the exact betting tree; the
 * deal pool holds pre-computed buckets and the showdown winner for each deal. Threads share
 * the regret / strategy tables (lock-free, "hogwild"). Early iterations are discounted
 * linearly (as in Pluribus).
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BUILD "build/"

/* ------------------------------------------------------------------ tables -- */
static int16_t NF[45 * 165], FL[45 * 165];
static uint8_t PAIR_IDX[81], TRIP_IDX[729];
static uint16_t PRE[376992];
static int64_t BINOM[37][6];

static const int P5[10][2] = {{0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}};
static const int T5[10][3] = {{0,1,2},{0,1,3},{0,1,4},{0,2,3},{0,2,4},{0,3,4},{1,2,3},{1,2,4},{1,3,4},{2,3,4}};

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(1); }

static void load_tables(void) {
    FILE *f = fopen(BUILD "tables.bin", "rb");
    if (!f) die("tables.bin");
    if (fread(NF, 2, 45 * 165, f) != 45 * 165 || fread(FL, 2, 45 * 165, f) != 45 * 165 ||
        fread(PAIR_IDX, 1, 81, f) != 81 || fread(TRIP_IDX, 1, 729, f) != 729) die("tables.bin short");
    fclose(f);
    f = fopen(BUILD "pre.bin", "rb");
    if (!f) die("pre.bin");
    if (fread(PRE, 2, 376992, f) != 376992) die("pre.bin short");
    fclose(f);
    for (int n = 0; n <= 36; n++)
        for (int k = 0; k <= 5; k++) {
            int64_t c = 1;
            if (k > n) c = 0;
            else for (int i = 0; i < k; i++) c = c * (n - i) / (i + 1);
            BINOM[n][k] = c;
        }
}

static inline void board_table(const int *b, int16_t *tab) {
    int ti[10], ts[10];
    for (int j = 0; j < 10; j++) {
        int a = b[T5[j][0]], c = b[T5[j][1]], d = b[T5[j][2]];
        ti[j] = TRIP_IDX[(a >> 2) * 81 + (c >> 2) * 9 + (d >> 2)];
        int sa = a & 3, sc = c & 3, sd = d & 3;
        ts[j] = (sa == sc && sc == sd) ? sa : 4;
    }
    for (int p = 0; p < 45; p++) {
        int16_t best = 0;
        for (int j = 0; j < 10; j++) { int16_t v = NF[p * 165 + ti[j]]; if (v > best) best = v; }
        for (int s = 0; s < 5; s++) tab[s * 45 + p] = best;
    }
    for (int j = 0; j < 10; j++)
        if (ts[j] < 4) {
            int s = ts[j];
            for (int p = 0; p < 45; p++) { int16_t v = FL[p * 165 + ti[j]]; if (v > tab[s * 45 + p]) tab[s * 45 + p] = v; }
        }
}

static inline void hand_codes(const int *h, int *codes) {
    for (int j = 0; j < 10; j++) {
        int a = h[P5[j][0]], c = h[P5[j][1]];
        int st = ((a & 3) == (c & 3)) ? (a & 3) : 4;
        codes[j] = st * 45 + PAIR_IDX[(a >> 2) * 9 + (c >> 2)];
    }
}

static inline int score(const int16_t *tab, const int *codes) {
    int best = 0;
    for (int j = 0; j < 10; j++) { int v = tab[codes[j]]; if (v > best) best = v; }
    return best;
}

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

static int64_t colex(const int *h5) {
    int s[5];
    memcpy(s, h5, sizeof s);
    qsort(s, 5, sizeof(int), cmp_int);
    return BINOM[s[0]][1] + BINOM[s[1]][2] + BINOM[s[2]][3] + BINOM[s[3]][4] + BINOM[s[4]][5];
}

/* --------------------------------------------------------------------- rng -- */
typedef struct { uint64_t s[4]; } rng_t;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rnext(rng_t *r) {
    uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static void rseed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static inline int rint_(rng_t *r, int n) { return (int)((rnext(r) >> 11) % (uint64_t)n); }
static inline double runif(rng_t *r) { return (rnext(r) >> 11) * (1.0 / 9007199254740992.0); }

/* ---------------------------------------------------------------- features -- */
#define NHIST 8
#define M_FLOP 24
#define N_OPP 128
#define N_RIVER 256

static int deck_without(uint64_t used, int *out) {
    int n = 0;
    for (int c = 0; c < 36; c++) if (!((used >> c) & 1)) out[n++] = c;
    return n;
}

static inline uint64_t mask_of(const int *c, int n) { uint64_t m = 0; for (int i = 0; i < n; i++) m |= 1ULL << c[i]; return m; }

/* percentile of our hand among N random opponent hands on the full board b5 */
static double hs_random(const int *my_codes, const int *b5, uint64_t used, int N, rng_t *r) {
    int16_t tab[225];
    board_table(b5, tab);
    int me = score(tab, my_codes);
    int deck[36];
    int n = deck_without(used, deck);
    int acc2 = 0;
    for (int i = 0; i < N; i++) {
        int h[5];
        for (int k = 0; k < 5; k++) {
            int j = k + rint_(r, n - k);
            int t = deck[k]; deck[k] = deck[j]; deck[j] = t;
            h[k] = deck[k];
        }
        int codes[10];
        hand_codes(h, codes);
        int s = score(tab, codes);
        acc2 += (me > s) ? 2 : (me == s ? 1 : 0);
    }
    return acc2 / (2.0 * N);
}

/* street 1 (flop: 3 board cards) or 2 (turn: 4 board cards): normalized histogram of HS */
static void hist_feature(const int *hand, const int *board, int street, rng_t *r, float *out) {
    int codes[10];
    hand_codes(hand, codes);
    for (int i = 0; i < NHIST; i++) out[i] = 0;
    int nb = street == 1 ? 3 : 4;
    uint64_t used = mask_of(hand, 5) | mask_of(board, nb);
    int rem[36];
    int nrem = deck_without(used, rem);
    int b5[5];
    memcpy(b5, board, nb * sizeof(int));
    int cnt = 0;
    if (street == 1) {
        for (int m = 0; m < M_FLOP; m++) {
            int i = rint_(r, nrem), j = rint_(r, nrem - 1);
            if (j >= i) j++;
            b5[3] = rem[i]; b5[4] = rem[j];
            double hs = hs_random(codes, b5, used | (1ULL << rem[i]) | (1ULL << rem[j]), N_OPP, r);
            int bin = (int)(hs * NHIST); if (bin >= NHIST) bin = NHIST - 1;
            out[bin] += 1; cnt++;
        }
    } else {
        for (int i = 0; i < nrem; i++) {
            b5[4] = rem[i];
            double hs = hs_random(codes, b5, used | (1ULL << rem[i]), N_OPP, r);
            int bin = (int)(hs * NHIST); if (bin >= NHIST) bin = NHIST - 1;
            out[bin] += 1; cnt++;
        }
    }
    for (int i = 0; i < NHIST; i++) out[i] /= cnt;
}

static void draw_cards(rng_t *r, int n, int *out) {
    int deck[36];
    for (int i = 0; i < 36; i++) deck[i] = i;
    for (int k = 0; k < n; k++) {
        int j = k + rint_(r, 36 - k);
        int t = deck[k]; deck[k] = deck[j]; deck[j] = t;
        out[k] = deck[k];
    }
}

typedef struct { int street; long lo, hi; float *out; uint64_t seed; } feat_job;

static void *feat_worker(void *arg) {
    feat_job *J = arg;
    rng_t r;
    rseed(&r, J->seed);
    for (long i = J->lo; i < J->hi; i++) {
        int c[9];
        draw_cards(&r, 9, c);
        hist_feature(c, c + 5, J->street, &r, J->out + i * NHIST);
    }
    return NULL;
}

static int nthreads_default(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 4; }

static void cmd_features(int argc, char **argv) {
    if (argc < 5) die("features <flop|turn> <n> <out>");
    int street = strcmp(argv[2], "flop") == 0 ? 1 : 2;
    long n = atol(argv[3]);
    float *out = malloc(sizeof(float) * NHIST * n);
    int T = nthreads_default();
    pthread_t th[64];
    feat_job jobs[64];
    for (int t = 0; t < T; t++) {
        jobs[t] = (feat_job){street, n * t / T, n * (t + 1) / T, out, 1000 + 7919ULL * t + street};
        pthread_create(&th[t], NULL, feat_worker, &jobs[t]);
    }
    for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
    FILE *f = fopen(argv[4], "wb");
    fwrite(out, sizeof(float), NHIST * n, f);
    fclose(f);
    printf("wrote %ld %s features\n", n, argv[2]);
}

/* -------------------------------------------------------------------- pool -- */
typedef struct { uint8_t cards[15]; uint8_t b[2][4]; int8_t result; } deal_t;   /* 24 bytes */

static float *CF, *CT;
static int B1g, B2g, B3g;

static int nearest(const float *x, const float *C, int k) {
    int best = 0;
    float bd = 1e30f;
    for (int j = 0; j < k; j++) {
        float d = 0;
        for (int i = 0; i < NHIST; i++) { float t = x[i] - C[j * NHIST + i]; d += t * t; }
        if (d < bd) { bd = d; best = j; }
    }
    return best;
}

static void make_deal(rng_t *r, deal_t *D) {
    int c[15];
    draw_cards(r, 15, c);
    for (int i = 0; i < 15; i++) D->cards[i] = (uint8_t)c[i];
    const int *board = c + 10;
    for (int p = 0; p < 2; p++) {
        const int *h = c + 5 * p;
        float x[NHIST];
        D->b[p][0] = (uint8_t)PRE[colex(h)];
        hist_feature(h, board, 1, r, x);
        D->b[p][1] = (uint8_t)nearest(x, CF, B1g);
        hist_feature(h, board, 2, r, x);
        D->b[p][2] = (uint8_t)nearest(x, CT, B2g);
        int codes[10];
        hand_codes(h, codes);
        double hs = hs_random(codes, board, mask_of(h, 5) | mask_of(board, 5), N_RIVER, r);
        int b3 = (int)(hs * B3g); if (b3 >= B3g) b3 = B3g - 1;
        D->b[p][3] = (uint8_t)b3;
    }
    int16_t tab[225];
    board_table(board, tab);
    int k0[10], k1[10];
    hand_codes(c, k0);
    hand_codes(c + 5, k1);
    int s0 = score(tab, k0), s1 = score(tab, k1);
    D->result = s0 > s1 ? 1 : (s1 > s0 ? -1 : 0);
}

typedef struct { long lo, hi; deal_t *out; uint64_t seed; } pool_job;
static void *pool_worker(void *arg) {
    pool_job *J = arg;
    rng_t r;
    rseed(&r, J->seed);
    for (long i = J->lo; i < J->hi; i++) make_deal(&r, &J->out[i]);
    return NULL;
}

static float *load_f32(const char *path, long n) {
    float *x = malloc(sizeof(float) * n);
    FILE *f = fopen(path, "rb");
    if (!f || fread(x, sizeof(float), n, f) != (size_t)n) die(path);
    fclose(f);
    return x;
}

static void cmd_pool(int argc, char **argv) {
    if (argc < 9) die("pool <n> <out> <flop_centroids> <B1> <turn_centroids> <B2> <B3>");
    long n = atol(argv[2]);
    B1g = atoi(argv[5]); B2g = atoi(argv[7]); B3g = atoi(argv[8]);
    CF = load_f32(argv[4], (long)B1g * NHIST);
    CT = load_f32(argv[6], (long)B2g * NHIST);
    deal_t *D = malloc(sizeof(deal_t) * n);
    int T = nthreads_default();
    pthread_t th[64];
    pool_job jobs[64];
    for (int t = 0; t < T; t++) {
        jobs[t] = (pool_job){n * t / T, n * (t + 1) / T, D, 424242 + 104729ULL * t + (uint64_t)n};
        pthread_create(&th[t], NULL, pool_worker, &jobs[t]);
    }
    for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
    FILE *f = fopen(argv[3], "wb");
    fwrite(D, sizeof(deal_t), n, f);
    fclose(f);
    printf("wrote %ld deals (%zu bytes each)\n", n, sizeof(deal_t));
}

/* ------------------------------------------------------------------- train -- */
static int NN;                       /* tree nodes */
static int32_t *Tplayer, *Tstreet, *Tkind, *Twin, *Tchild, *Tcontrib;
static int64_t *Toff;                /* info-set offset of each decision node */
static int Bst[4];
static int64_t NI;                   /* number of info sets */
static float *REG;                   /* regrets [NI][4] */
static double *SUM;                  /* average-strategy sums [NI][4] */
static deal_t *POOL;
static long NPOOL;
static volatile int64_t ITER = 0;
static volatile int STOP = 0;
static double DISC = 1.0;            /* weight of new average-strategy contributions */

static void load_tree(void) {
    FILE *f = fopen(BUILD "tree.bin", "rb");
    if (!f) die("tree.bin");
    if (fread(&NN, 4, 1, f) != 1) die("tree header");
    Tplayer = malloc(4L * NN); Tstreet = malloc(4L * NN); Tkind = malloc(4L * NN); Twin = malloc(4L * NN);
    Tchild = malloc(16L * NN); Tcontrib = malloc(8L * NN);
    if (fread(Tplayer, 4, NN, f) != (size_t)NN || fread(Tstreet, 4, NN, f) != (size_t)NN ||
        fread(Tkind, 4, NN, f) != (size_t)NN || fread(Twin, 4, NN, f) != (size_t)NN ||
        fread(Tchild, 4, 4L * NN, f) != 4UL * NN || fread(Tcontrib, 4, 2L * NN, f) != 2UL * NN) die("tree short");
    fclose(f);
}

static inline void regret_match(const float *R, const int32_t *ch, float *sig) {
    float pos[4], tot = 0;
    int nl = 0;
    for (int a = 0; a < 4; a++) {
        if (ch[a] < 0) { pos[a] = 0; continue; }
        nl++;
        pos[a] = R[a] > 0 ? R[a] : 0;
        tot += pos[a];
    }
    for (int a = 0; a < 4; a++) sig[a] = ch[a] < 0 ? 0 : (tot > 0 ? pos[a] / tot : 1.0f / nl);
}

static double traverse(int n, int p, const deal_t *D, rng_t *r) {
    int kind = Tkind[n];
    if (kind == 1) return Twin[n] == p ? Tcontrib[2 * n + 1 - p] : -Tcontrib[2 * n + p];
    if (kind == 2) {
        int res = p == 0 ? D->result : -D->result;
        return (double)res * Tcontrib[2 * n + 1 - p];
    }
    int q = Tplayer[n], s = Tstreet[n];
    int64_t I = Toff[n] + D->b[q][s];
    const int32_t *ch = &Tchild[4 * n];
    float sig[4];
    float *R = &REG[4 * I];
    regret_match(R, ch, sig);
    if (q == p) {
        double v[4], u = 0;
        for (int a = 0; a < 4; a++) {
            if (ch[a] < 0) continue;
            v[a] = traverse(ch[a], p, D, r);
            u += sig[a] * v[a];
        }
        for (int a = 0; a < 4; a++) if (ch[a] >= 0) R[a] += (float)(v[a] - u);
        return u;
    }
    double *S = &SUM[4 * I];
    for (int a = 0; a < 4; a++) if (ch[a] >= 0) S[a] += DISC * sig[a];
    double x = runif(r), c = 0;
    int pick = -1;
    for (int a = 0; a < 4; a++) {
        if (ch[a] < 0) continue;
        pick = a;
        c += sig[a];
        if (x < c) break;
    }
    return traverse(ch[pick], p, D, r);
}

typedef struct { uint64_t seed; } train_job;
static void *train_worker(void *arg) {
    train_job *J = arg;
    rng_t r;
    rseed(&r, J->seed);
    while (!STOP) {
        for (int k = 0; k < 256; k++) {
            const deal_t *D = &POOL[(long)(rnext(&r) % (uint64_t)NPOOL)];
            traverse(0, 0, D, &r);
            traverse(0, 1, D, &r);
        }
        __sync_fetch_and_add(&ITER, 256);
    }
    return NULL;
}

static void save(const char *prefix) {
    char path[512];
    snprintf(path, sizeof path, "%s.strategy.f32", prefix);
    FILE *f = fopen(path, "wb");
    float *row = malloc(sizeof(float) * 4);
    for (int64_t i = 0; i < NI; i++) {
        double t = SUM[4 * i] + SUM[4 * i + 1] + SUM[4 * i + 2] + SUM[4 * i + 3];
        for (int a = 0; a < 4; a++) row[a] = t > 0 ? (float)(SUM[4 * i + a] / t) : -1.0f;  /* -1 = never reached */
        fwrite(row, sizeof(float), 4, f);
    }
    free(row);
    fclose(f);
    snprintf(path, sizeof path, "%s.regret.f32", prefix);
    f = fopen(path, "wb"); fwrite(REG, sizeof(float), 4 * NI, f); fclose(f);
    snprintf(path, sizeof path, "%s.sum.f64", prefix);
    f = fopen(path, "wb"); fwrite(SUM, sizeof(double), 4 * NI, f); fclose(f);
    snprintf(path, sizeof path, "%s.meta.txt", prefix);
    f = fopen(path, "w");
    fprintf(f, "iterations %lld\nbuckets %d %d %d %d\ninfosets %lld\n", (long long)ITER, Bst[0], Bst[1], Bst[2], Bst[3], (long long)NI);
    fclose(f);
}

static void cmd_train(int argc, char **argv) {
    if (argc < 10) die("train <pool> <B0> <B1> <B2> <B3> <seconds> <threads> <out_prefix> [resume_prefix]");
    load_tree();
    for (int s = 0; s < 4; s++) Bst[s] = atoi(argv[3 + s]);
    double seconds = atof(argv[7]);
    int T = atoi(argv[8]);
    const char *out = argv[9];
    Toff = malloc(8L * NN);
    NI = 0;
    for (int n = 0; n < NN; n++) {
        if (Tkind[n] == 0) { Toff[n] = NI; NI += Bst[Tstreet[n]]; } else Toff[n] = -1;
    }
    REG = calloc(4 * NI, sizeof(float));
    SUM = calloc(4 * NI, sizeof(double));
    if (argc > 10) {
        char path[512];
        snprintf(path, sizeof path, "%s.regret.f32", argv[10]);
        FILE *f = fopen(path, "rb");
        if (!f || fread(REG, sizeof(float), 4 * NI, f) != (size_t)(4 * NI)) die("resume regrets");
        fclose(f);
        snprintf(path, sizeof path, "%s.sum.f64", argv[10]);
        f = fopen(path, "rb");
        if (!f || fread(SUM, sizeof(double), 4 * NI, f) != (size_t)(4 * NI)) die("resume sums");
        fclose(f);
        printf("resumed from %s\n", argv[10]);
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) die("pool");
    fseek(f, 0, SEEK_END);
    NPOOL = ftell(f) / (long)sizeof(deal_t);
    fseek(f, 0, SEEK_SET);
    POOL = malloc(sizeof(deal_t) * NPOOL);
    if (fread(POOL, sizeof(deal_t), NPOOL, f) != (size_t)NPOOL) die("pool short");
    fclose(f);
    printf("tree %d nodes, %lld info sets, pool %ld deals, %d threads, %.0f s\n", NN, (long long)NI, NPOOL, T, seconds);
    fflush(stdout);
    pthread_t th[64];
    train_job jobs[64];
    for (int t = 0; t < T; t++) {
        jobs[t].seed = (uint64_t)time(NULL) * 1000003ULL + 7777ULL * t;
        pthread_create(&th[t], NULL, train_worker, &jobs[t]);
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double last_report = 0, last_save = 0;
    int64_t next_disc = 2000000;      /* linear discounting interval (iterations) */
    int disc_k = 1;
    while (1) {
        usleep(200000);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        if (ITER >= next_disc && disc_k <= 50 && argc <= 10) {
            /* linear CFR: scale old regrets and sums by k/(k+1) for the first intervals */
            double d = (double)disc_k / (disc_k + 1);
            for (int64_t i = 0; i < 4 * NI; i++) { REG[i] *= (float)d; SUM[i] *= d; }
            disc_k++;
            next_disc += 2000000;
        }
        if (el - last_report > 30) {
            printf("  %.0fs: %lld iterations (%.0f/s)\n", el, (long long)ITER, ITER / el);
            fflush(stdout);
            last_report = el;
        }
        if (el - last_save > 600) { save(out); last_save = el; }
        if (el >= seconds) break;
    }
    STOP = 1;
    for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
    save(out);
    printf("done: %lld iterations\n", (long long)ITER);
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: solver <features|pool|train> ...");
    load_tables();
    if (!strcmp(argv[1], "features")) cmd_features(argc, argv);
    else if (!strcmp(argv[1], "pool")) cmd_pool(argc, argv);
    else if (!strcmp(argv[1], "train")) cmd_train(argc, argv);
    else die("unknown command");
    return 0;
}
