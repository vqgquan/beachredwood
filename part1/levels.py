"""Level layouts for the Part I gridworld.

Legend: S start, R rock, F fire, A apple, K key, C chest, M monster.

Level 0: apples only (Task 1, Q-learning).
Level 1: apples behind a fire corridor (Task 2, SARSA vs Q-learning).
Levels 2-3: multiple apples, a key and a chest (Task 3).
Levels 4-5: stochastic monsters (Task 4).
Level 6: intrinsic-reward exploration (Task 5).
"""
from typing import List

GRID_W, GRID_H = 12, 8


def pad_level(rows: List[str]) -> List[str]:
    return [row.ljust(GRID_W)[:GRID_W] for row in rows]


LEVELS = {
    0: pad_level(
        [
            "S           ",
            "            ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
        ]
    ),
    1: pad_level(
        [
            "            ",
            "            ",
            "     FFFFFF ",
            "S          A",
            "     FFFFFF ",
            "            ",
            "            ",
            "            ",
        ]
    ),
    2: pad_level(
        [
            "S  R   A   A",
            "   R   R    ",
            "   K   R    ",
            "   R   C   A",
            "       R    ",
            "   A       R",
            "            ",
            "            ",
        ]
    ),
    3: pad_level(
        [
            "S   A   R  A",
            "R   R   R   ",
            "K       C   ",
            "R   A   R   ",
            "    R   A   ",
            "A       R   ",
            "    F   R   ",
            "        A   ",
        ]
    ),
    4: pad_level(
        [
            "S     M   A ",
            "            ",
            "    R       ",
            "            ",
            "       R    ",
            "            ",
            "            ",
            "            ",
        ]
    ),
    5: pad_level(
        [
            "S  M      A ",
            "   R        ",
            "      M     ",
            "  A         ",
            "       R    ",
            "            ",
            "            ",
            "       A    ",
        ]
    ),
    6: pad_level(
        [
            "S           ",
            "  A     R   ",
            "      K     ",
            "    R   C A ",
            "            ",
            " A          ",
            "            ",
            "            ",
        ]
    ),
}


LEVEL_LABELS = {
    0: "Level 0 - apples only",
    1: "Level 1 - apples with hazards",
    2: "Level 2 - apples, key, chest",
    3: "Level 3 - mixed collectible layout",
    4: "Level 4 - stochastic monster",
    5: "Level 5 - two stochastic monsters",
    6: "Level 6 - intrinsic reward",
}
