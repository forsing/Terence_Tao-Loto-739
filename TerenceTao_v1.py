#!/usr/bin/env python3


"""
Prediktor inspirisan analizom strukture i korelacija viseg reda.

- Svaki CSV ima zaseban model i validaciju.
- Bez random postupaka.
- Bez nametnute normalne ili uniformne loto raspodele.
- Bez frekvencijskih tabela pojedinacnih brojeva.
- Bez filtera nizova, parnosti, grupa ili razmaka.
- Ne pretvara prosečne sortirane pozicije u kombinaciju.

Skor sluzi rangiranju kombinacija.
Pretraga pronalazi najbolju kombinaciju medju pregledanim kandidatima.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np


DEFAULTS = (
    "/data/loto7_4702_k82.csv",
    "/data/loto7_4702_k82_loto_2973.csv",
    "/data/loto7_4702_k82_loto_plus_1729.csv",
)

TRAIN = 192
VALIDATION = 48
TEST = 64
REFIT = 16

# Broj komponenti konteksta i jacina regularizacije.
CONFIGS = (
    (4, 0.05),
    (4, 0.5),
    (8, 0.5),
)

TRIPLES = np.array(
    list(itertools.combinations(range(7), 3))
)

QUADS = np.array(
    list(itertools.combinations(range(7), 4))
)

DOMAIN = np.arange(1, 40)[:, None]

# Prost broj 83 je veci od najveceg zbira dva loto broja.
LINEAR = np.exp(
    2j * np.pi
    * DOMAIN
    * np.arange(1, 9)[None, :]
    / 83
)

AB = np.array([
    (a, b)
    for a in (1, 2, 3)
    for b in (0, 1, 3)
])

QUADRATIC = np.exp(
    2j * np.pi
    * (
        DOMAIN ** 2 * AB[:, 0]
        + DOMAIN * AB[:, 1]
    )
    / 83
)

BISPECTRUM = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 5),
)


def read_csv(path):
    rows = []

    with open(
        path,
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for number, cells in enumerate(
            csv.reader(handle), 1
        ):
            if not cells or all(
                not c.strip() for c in cells
            ):
                continue

            try:
                row = tuple(
                    int(c.strip()) for c in cells
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {number}: "
                    "ocekujem sedam celih brojeva"
                ) from exc

            if (
                len(row) != 7
                or len(set(row)) != 7
                or min(row) < 1
                or max(row) > 39
            ):
                raise ValueError(
                    f"{path}, red {number}: "
                    f"neispravna kombinacija {row}"
                )

            rows.append(tuple(sorted(row)))

    if len(rows) < TRAIN + VALIDATION + TEST + 3:
        raise ValueError(
            f"{path}: nedovoljno redova"
        )

    return np.asarray(
        rows,
        dtype=np.int16,
    )


def features(candidates, higher=True):
    c = np.atleast_2d(candidates).astype(int)

    linear = LINEAR[c - 1].mean(axis=1)

    first = np.column_stack((
        linear.real,
        linear.imag,
    ))

    if not higher:
        return first

    quadratic = QUADRATIC[c - 1].mean(axis=1)

    bis = np.column_stack([
        linear[:, a - 1]
        * linear[:, b - 1]
        * np.conj(linear[:, a + b - 1])
        for a, b in BISPECTRUM
    ])

    q = c[:, QUADS]

    # Cetiri razlicita broja:
    # a + d = b + c.
    equal = (
        q[:, :, 0] + q[:, :, 3]
        == q[:, :, 1] + q[:, :, 2]
    )

    q_anchor = (
        equal
        * np.exp(
            2j * np.pi * q.sum(axis=2) / 83
        )
    ).mean(axis=1)

    q_span = q[:, :, 3] - q[:, :, 0]

    t = c[:, TRIPLES]

    # Tri razlicita broja:
    # a + c = 2b.
    progression = (
        t[:, :, 0] + t[:, :, 2]
        == 2 * t[:, :, 1]
    )

    t_anchor = (
        progression
        * np.exp(
            2j * np.pi * t[:, :, 1] / 83
        )
    ).mean(axis=1)

    relations = np.column_stack((
        equal.mean(axis=1),
        q_anchor.real,
        q_anchor.imag,
        (equal & (q_span <= 10)).mean(axis=1),
        (equal & (q_span <= 20)).mean(axis=1),
        progression.mean(axis=1),
        t_anchor.real,
        t_anchor.imag,
    ))

    return np.column_stack((
        first,
        quadratic.real,
        quadratic.imag,
        bis.real,
        bis.imag,
        relations,
    ))


def context(phi, t):
    if t < 3:
        raise ValueError(
            "Nedovoljno prethodnih izvlacenja"
        )

    return np.concatenate((
        phi[t - 1],
        phi[t - 2],
        phi[t - 1] - phi[t - 3],
    ))


def alternatives(rows, t):
    actual = tuple(map(int, rows[t]))
    candidates = {actual}

    # Zavrsene istorijske kombinacije kao trening kandidati.
    for lag in range(1, 33):
        if t >= lag:
            candidates.add(
                tuple(map(int, rows[t - lag]))
            )

    # Dodatni deterministicki kandidati.
    # Nisu predstavljeni kao stvarna izvlacenja.
    for position in range(7):
        shifts = (
            (1,) if position % 2 == 0 else (7,)
        )

        for shift in shifts:
            c = list(actual)

            c[position] = (
                (c[position] - 1 + shift) % 39 + 1
            )

            if len(set(c)) == 7:
                candidates.add(
                    tuple(sorted(c))
                )

    candidates = sorted(candidates)

    # Stvarni broj pogodaka prema poznatom trening ishodu.
    y = np.array([
        len(set(c) & set(actual))
        for c in candidates
    ], dtype=float)

    return candidates, y


class Model:
    def __init__(
        self,
        rows,
        phi,
        end,
        config,
        higher,
    ):
        self.higher = higher
        self.rank, regularization = config

        self.indices = list(range(
            max(8, end - TRAIN),
            end,
        ))

        contexts = []
        candidate_blocks = []
        targets = []
        weights = []

        for t in self.indices:
            candidates, y = alternatives(
                rows,
                t,
            )

            contexts.append(
                context(phi, t)
            )

            candidate_blocks.append(
                features(candidates, higher)
            )

            targets.append(y)

            weights.extend(
                [1 / len(candidates)]
                * len(candidates)
            )

        self.raw_contexts = np.asarray(contexts)

        self.c_center = np.median(
            self.raw_contexts,
            axis=0,
        )

        self.c_scale = np.maximum(
            np.quantile(
                self.raw_contexts,
                0.75,
                axis=0,
            )
            - np.quantile(
                self.raw_contexts,
                0.25,
                axis=0,
            ),
            0.05,
        )

        standardized = (
            self.raw_contexts - self.c_center
        ) / self.c_scale

        _, _, vt = np.linalg.svd(
            standardized,
            full_matrices=False,
        )

        self.components = (
            vt[:self.rank].T.copy()
        )

        # Fiksiranje znaka komponenti.
        for j in range(
            self.components.shape[1]
        ):
            i = int(np.argmax(
                np.abs(self.components[:, j])
            ))

            if self.components[i, j] < 0:
                self.components[:, j] *= -1

        projected = (
            standardized @ self.components
        )

        self.p_scale = np.maximum(
            np.sqrt(
                np.mean(projected ** 2, axis=0)
            ),
            0.2,
        )

        projected /= self.p_scale
        self.projected = projected

        raw = phi[self.indices]

        self.raw_center = np.median(
            raw,
            axis=0,
        )

        self.raw_scale = np.maximum(
            np.quantile(raw, 0.75, axis=0)
            - np.quantile(raw, 0.25, axis=0),
            0.05,
        )

        normalized = (
            raw - self.raw_center
        ) / self.raw_scale

        # Oslonci se biraju iz trening podataka.
        # Nisu obavezni izlazi niti ciljni raspored brojeva.
        chosen = [len(normalized) - 1]

        nearest = np.full(
            len(normalized),
            np.inf,
        )

        for _ in range(
            min(32, len(normalized)) - 1
        ):
            distance = np.mean(
                (
                    normalized
                    - normalized[chosen[-1]]
                ) ** 2,
                axis=1,
            )

            nearest = np.minimum(
                nearest,
                distance,
            )

            nearest[chosen] = -1
            index = int(np.argmax(nearest))

            if nearest[index] <= 1e-12:
                break

            chosen.append(index)

        self.anchors = normalized[chosen]

        self.anchor_norms = np.sum(
            self.anchors ** 2,
            axis=1,
        )

        distances = np.mean(
            (
                normalized[:, None, :]
                - self.anchors[None, :, :]
            ) ** 2,
            axis=2,
        )

        positive = distances[
            distances > 1e-12
        ]

        self.bandwidth = max(
            (
                float(np.median(positive))
                if len(positive)
                else 1.0
            ),
            0.05,
        )

        candidate_blocks = [
            self.kernel(block)
            for block in candidate_blocks
        ]

        candidate_values = np.concatenate(
            candidate_blocks
        )

        self.f_center = np.median(
            candidate_values,
            axis=0,
        )

        self.f_scale = np.maximum(
            np.quantile(
                candidate_values,
                0.75,
                axis=0,
            )
            - np.quantile(
                candidate_values,
                0.25,
                axis=0,
            ),
            0.05,
        )

        design = []

        for z, block in zip(
            projected,
            candidate_blocks,
        ):
            a = np.r_[1.0, z]

            b = np.column_stack((
                np.ones(len(block)),
                (
                    block - self.f_center
                ) / self.f_scale,
            ))

            design.append(
                (
                    a[None, :, None]
                    * b[:, None, :]
                ).reshape(len(block), -1)
            )

        design = np.concatenate(design)
        y = np.concatenate(targets)
        w = np.asarray(weights)

        gram = (
            design.T
            @ (w[:, None] * design)
        )

        penalty = regularization * w.sum()

        gram.flat[
            ::gram.shape[0] + 1
        ] += penalty

        # Globalni intercept se ne regularizuje.
        gram[0, 0] -= penalty

        right = design.T @ (w * y)

        solution = np.linalg.solve(
            gram,
            right,
        )

        self.coef = solution.reshape(
            len(projected[0]) + 1,
            len(self.f_center) + 1,
        )

        if not np.all(
            np.isfinite(self.coef)
        ):
            raise RuntimeError(
                "Ne-konacni koeficijenti modela"
            )

    def query(self, phi, t):
        standardized = (
            context(phi, t) - self.c_center
        ) / self.c_scale

        projected = (
            standardized @ self.components
        ) / self.p_scale

        coefficients = (
            np.r_[1.0, projected]
            @ self.coef
        )

        return coefficients, projected

    def kernel(self, raw):
        z = (
            raw - self.raw_center
        ) / self.raw_scale

        distance = (
            np.sum(z * z, axis=1)[:, None]
            + self.anchor_norms[None, :]
            - 2 * z @ self.anchors.T
        ) / z.shape[1]

        # Racionalno jezgro; nije Gaussian raspodela.
        return 1 / (
            1
            + np.maximum(distance, 0)
            / self.bandwidth
        ) ** 2

    def score(
        self,
        candidates,
        coefficients,
    ):
        encoded = self.kernel(
            features(
                candidates,
                self.higher,
            )
        )

        f = (
            encoded - self.f_center
        ) / self.f_scale

        return (
            coefficients[0]
            + f @ coefficients[1:]
        )

    def predict(self, rows, phi, t):
        coefficients, projected = self.query(
            phi,
            t,
        )

        order = np.argsort(
            np.sum(
                (
                    self.projected - projected
                ) ** 2,
                axis=1,
            ),
            kind="stable",
        )[:12]

        seeds = {
            tuple(map(int, row))
            for row in rows[max(0, t - 8):t]
        }

        seeds.update(
            tuple(map(
                int,
                rows[self.indices[i]],
            ))
            for i in order
        )

        states = sorted(seeds)

        values = self.score(
            states,
            coefficients,
        )

        order = sorted(
            range(len(states)),
            key=lambda i: (
                -values[i],
                states[i],
            ),
        )

        states = [
            states[i]
            for i in order[:4]
        ]

        best = states[0]
        best_score = float(
            values[order[0]]
        )

        # Jednaka pravila pretrage za sve oblike kombinacija.
        for _ in range(6):
            candidates = set(states)

            for c in states:
                for position in range(7):
                    rest = (
                        c[:position]
                        + c[position + 1:]
                    )

                    for number in range(1, 40):
                        if number not in rest:
                            candidates.add(
                                tuple(sorted(
                                    rest + (number,)
                                ))
                            )

            candidates = sorted(candidates)

            values = self.score(
                candidates,
                coefficients,
            )

            order = sorted(
                range(len(candidates)),
                key=lambda i: (
                    -values[i],
                    candidates[i],
                ),
            )

            next_states = [
                candidates[i]
                for i in order[:4]
            ]

            next_score = float(
                values[order[0]]
            )

            if (
                next_score
                > best_score + 1e-12
            ):
                best = next_states[0]
                best_score = next_score

            if next_states == states:
                break

            states = next_states

        assert len(best) == 7
        assert len(set(best)) == 7
        assert 1 <= best[0] < best[-1] <= 39

        return best


def hits(a, b):
    return len(set(a) & set(b))


def evaluate(
    rows,
    phi,
    start,
    end,
    config,
    higher,
):
    predictions = []
    model = None

    for t in range(start, end):
        if (
            model is None
            or (t - start) % REFIT == 0
        ):
            model = Model(
                rows,
                phi,
                t,
                config,
                higher,
            )

        predictions.append(
            model.predict(
                rows,
                phi,
                t,
            )
        )

    scores = np.array([
        hits(p, rows[t])
        for t, p in zip(
            range(start, end),
            predictions,
        )
    ])

    return scores, predictions


def select(
    rows,
    phi,
    start,
    end,
    higher,
):
    means = []

    label = (
        "Tao"
        if higher
        else "samo linearne Fourierove osobine"
    )

    for config in CONFIGS:
        scores, _ = evaluate(
            rows,
            phi,
            start,
            end,
            config,
            higher,
        )

        means.append(
            float(scores.mean())
        )

        print(
            f"Validacija {label}, {config}: "
            f"{scores.mean():.4f}",
            flush=True,
        )

    index = max(
        range(len(CONFIGS)),
        key=lambda i: (
            means[i],
            -i,
        ),
    )

    return CONFIGS[index]


def longest_run(c):
    current = longest = 1

    for a, b in zip(c, c[1:]):
        current = (
            current + 1
            if b == a + 1
            else 1
        )

        longest = max(
            longest,
            current,
        )

    return longest


def run(path):
    rows = read_csv(path)
    n = len(rows)

    phi = features(rows, True)
    linear_phi = features(rows, False)

    test_start = n - TEST
    valid_start = n - TEST - VALIDATION

    print(
        f"\n{path}\n"
        f"Redova: {n}; poslednja: "
        f"{tuple(map(int, rows[-1]))}",
        flush=True,
    )

    config = select(
        rows,
        phi,
        valid_start,
        test_start,
        True,
    )

    linear_config = select(
        rows,
        linear_phi,
        valid_start,
        test_start,
        False,
    )

    scores, predictions = evaluate(
        rows,
        phi,
        test_start,
        n,
        config,
        True,
    )

    linear_scores, _ = evaluate(
        rows,
        linear_phi,
        test_start,
        n,
        linear_config,
        False,
    )

    previous = np.array([
        hits(
            rows[t - 1],
            rows[t],
        )
        for t in range(test_start, n)
    ])

    print(
        f"Zamrznuti parametri: Tao={config}; "
        f"linearni={linear_config}"
    )

    print(
        f"Test {TEST}: "
        f"Tao={scores.mean():.4f}; "
        f"linearni={linear_scores.mean():.4f}"
    )

    print(
        "Kontrola: prethodno izvuceno kolo="
        f"{previous.mean():.4f}"
    )

    print(
        "Distribucija pogodaka 0..7:",
        np.bincount(
            scores,
            minlength=8,
        ).tolist(),
    )

    print(
        f"3+={int((scores >= 3).sum())}/{TEST}; "
        f"4+={int((scores >= 4).sum())}/{TEST}"
    )

    replays = sum(
        any(
            tuple(row) == prediction
            for row in rows[:t]
        )
        for t, prediction in zip(
            range(test_start, n),
            predictions,
        )
    )

    long_runs = sum(
        longest_run(p) >= 5
        for p in predictions
    )

    print(
        f"Razlicitih kombinacija="
        f"{len(set(predictions))}/{TEST}; "
        f"istorijskih ponavljanja="
        f"{replays}/{TEST}; "
        f"nizova 5+={long_runs}"
    )

    model = Model(
        rows,
        phi,
        n,
        config,
        True,
    )

    result = model.predict(
        rows,
        phi,
        n,
    )

    if result != model.predict(
        rows,
        phi,
        n,
    ):
        raise RuntimeError(
            "Predikcija nije ponovljiva"
        )

    print(
        f"NEXT red {n + 1}: "
        + ",".join(map(str, result))
    )

    print(
        "NEXT vec postoji u istoriji:",
        any(
            tuple(row) == result
            for row in rows
        ),
    )

    print(
        "NEXT najduzi niz:",
        longest_run(result),
    )

    print(
        "NEXT brojeva <=19:",
        sum(
            number <= 19
            for number in result
        ),
    )

    if scores.mean() <= linear_scores.mean():
        print(
            "Na ovom testu vise osobine nisu "
            "dale bolji prosek od linearne kontrole."
        )

    return result


def self_check():
    c = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [2, 6, 13, 19, 24, 31, 38],
    ])

    f = features(c)

    assert f.shape == (2, 52)
    assert np.all(np.isfinite(f))
    assert np.all(
        np.abs(f) <= 1 + 1e-12
    )

    assert longest_run(
        tuple(c[0])
    ) == 7

    assert np.array_equal(
        features(c, False),
        f[:, :16],
    )

    q = c[0, QUADS]

    direct = sum(
        int(a + d == b + cc)
        for a, b, cc, d in q
    ) / 35

    assert np.isclose(
        f[0, 44],
        direct,
    )

    rows = np.array([
        sorted({
            (3 * i + 5 * j) % 39 + 1
            for j in range(7)
        })
        for i in range(90)
    ], dtype=np.int16)

    changed = rows.copy()
    changed[75:] = rows[0]

    p = features(rows)
    p2 = features(changed)

    m = Model(
        rows,
        p,
        75,
        CONFIGS[0],
        True,
    )

    m2 = Model(
        changed,
        p2,
        75,
        CONFIGS[0],
        True,
    )

    assert np.array_equal(
        m.coef,
        m2.coef,
    )

    assert (
        m.predict(rows, p, 75)
        == m2.predict(changed, p2, 75)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        nargs="*",
        default=list(DEFAULTS),
    )

    args = parser.parse_args()

    self_check()

    print(
        "Tao V1: korelacije viseg reda; "
        "bez random generisanja i filtera oblika."
    )

    results = []

    for path in args.csv:
        results.append((
            path,
            run(path),
        ))

    for i, (left, a) in enumerate(results):
        for right, b in results[i + 1:]:
            if a == b:
                print(
                    f"ISTA NEXT: "
                    f"{Path(left).name} / "
                    f"{Path(right).name}: {a}"
                )


if __name__ == "__main__":
    main()



"""
Tao V1: korelacije viseg reda; bez random generisanja i filtera oblika.



/data/loto7_4702_k82.csv
Redova: 4702; poslednja: (8, 10, 13, 21, 22, 23, 32)
Validacija Tao, (4, 0.05): 1.2708
Validacija Tao, (4, 0.5): 1.2083
Validacija Tao, (8, 0.5): 1.0417
Validacija samo linearne Fourierove osobine, (4, 0.05): 1.2292
Validacija samo linearne Fourierove osobine, (4, 0.5): 1.1458
Validacija samo linearne Fourierove osobine, (8, 0.5): 1.2917
Zamrznuti parametri: Tao=(4, 0.05); linearni=(8, 0.5)
Test 64: Tao=1.3125; linearni=1.1719
Kontrola: prethodno izvuceno kolo=1.1719
Distribucija pogodaka 0..7: [16, 20, 21, 6, 1, 0, 0, 0]
3+=7/64; 4+=1/64
Razlicitih kombinacija=35/64; istorijskih ponavljanja=52/64; nizova 5+=1
NEXT red 4703: 3,x,12,y,18,z,33
NEXT vec postoji u istoriji: True
NEXT najduzi niz: 2
NEXT brojeva <=19: 5



/data/loto7_4702_k82_loto_2973.csv
Redova: 2973; poslednja: (1, 2, 8, 26, 29, 35, 39)
Validacija Tao, (4, 0.05): 1.2708
Validacija Tao, (4, 0.5): 1.2083
Validacija Tao, (8, 0.5): 1.3333
Validacija samo linearne Fourierove osobine, (4, 0.05): 1.1875
Validacija samo linearne Fourierove osobine, (4, 0.5): 1.2083
Validacija samo linearne Fourierove osobine, (8, 0.5): 1.3542
Zamrznuti parametri: Tao=(8, 0.5); linearni=(8, 0.5)
Test 64: Tao=1.3438; linearni=1.2500
Kontrola: prethodno izvuceno kolo=1.3281
Distribucija pogodaka 0..7: [10, 27, 22, 5, 0, 0, 0, 0]
3+=5/64; 4+=0/64
Razlicitih kombinacija=40/64; istorijskih ponavljanja=45/64; nizova 5+=0
NEXT red 2974: 18,x,21,y,29,z,38
NEXT vec postoji u istoriji: True
NEXT najduzi niz: 2
NEXT brojeva <=19: 1



/data/loto7_4702_k82_loto_plus_1729.csv
Redova: 1729; poslednja: (8, 10, 13, 21, 22, 23, 32)
Validacija Tao, (4, 0.05): 1.2708
Validacija Tao, (4, 0.5): 1.2917
Validacija Tao, (8, 0.5): 1.3333
Validacija samo linearne Fourierove osobine, (4, 0.05): 1.2500
Validacija samo linearne Fourierove osobine, (4, 0.5): 1.4375
Validacija samo linearne Fourierove osobine, (8, 0.5): 1.1042
Zamrznuti parametri: Tao=(8, 0.5); linearni=(4, 0.5)
Test 64: Tao=1.1406; linearni=1.3750
Kontrola: prethodno izvuceno kolo=1.2656
Distribucija pogodaka 0..7: [18, 26, 14, 5, 1, 0, 0, 0]
3+=6/64; 4+=1/64
Razlicitih kombinacija=52/64; istorijskih ponavljanja=37/64; nizova 5+=0
NEXT red 1730: 2,x,8,y,18,z,22
NEXT vec postoji u istoriji: True
NEXT najduzi niz: 2
NEXT brojeva <=19: 5
Na ovom testu vise osobine nisu dale bolji prosek od linearne kontrole.
"""



"""
matematicar Terence Tao (born 1975) 

Iz Taoovih radova analiza korelacija višeg reda — odnosa koji se ne vide kada brojeve ili parove posmatramo odvojeno. 
To je važan pravac u njegovoj knjizi Higher Order Fourier Analysis.

1. Opis cele kombinacije kroz zajedničke odnose brojeva. Na primer, koliko različitih četvorki zadovoljava odnos a + b = c + d, kako se preklapaju takvi odnosi i gde se nalaze unutar kombinacije. Time dobijam opis unutrašnje strukture, bez rangiranja po frekvenciji pojedinačnih brojeva.
2. Učenje vremenskih prelaza tih struktura. Ne bi bilo dovoljno prepoznati zanimljiv obrazac u jednom izvlačenju. Prediktor bi morao da nauči da li određeni opis prethodnih kombinacija pomaže ocenjivanju naredne.
3. Zajedničko ocenjivanje kandidata. Svaka predložena sedmorka dobija ocenu prema naučenim odnosima. Ne bih propisivao ravnomeran raspored, parnost, poželjan broj nizova ili obavezne aritmetičke progresije.

Druga korisna Taoova ideja jeste razdvajanje jednostavne strukture od ostatka koji s njom nije korelisan. U mojoj adaptaciji to bi značilo zadržavanje odnosa koji daju ponovljiv doprinos predikciji, umesto gomilanja matematičkih osobina. Tao taj princip razvija u radu The dichotomy between structure and randomness…. Ovde „randomness” opisuje svojstvo ostatka; ne zahteva random generisanje kombinacija.
Ovo vredi razmatrati kao zasebnu ideju: predikcija prelaza između višerednih struktura celih kombinacija. Predloženi loto postupak bio bi konstrukcija zasnovana na tim idejama, a ne direktna posledica Taoovih teorema.

Odnosi trojki i četvorki, kvadratne Fourierove osobine i učenje prelaza ka narednom izvlačenju. 
Konačna kombinacija biraće se deterministički, bez ručnih zabrana nizova, grupa ili razmaka.

Koristi više korelacije, determinističko učenje i ocenjivanje cele kombinacije. Nema random postupaka ni ručnih filtera oblika.

Rezultati poslednja 64 kola:

CSV	    Tao V1	Samo linearne osobine	Prethodno izvučeno kolo
4702	1,3125	1,1719	                1,1719
2973	1,3438	1,2500	                1,3281
1729	1,1406	1,3750	                1,2656

Ograničenje: model često ponavlja istorijske kombinacije — 52/64, 45/64 i 37/64. Sva tri sadašnja NEXT izlaza već postoje u istoriji. Na Plus fajlu više korelacije nisu poboljšale rezultat.
"""
