from itertools import (
    chain,
    repeat,
)
from json import load
from operator import sub


def count_jumps(frame_indices, num_frames):
    jumps = list(
        chain.from_iterable(
            map(
                lambda x: repeat(*x),
                enumerate(
                    map(
                        sub,
                        chain(
                            frame_indices,
                            [frame_indices[-1] + 1]
                        ),
                        chain(
                            [0],
                            frame_indices,
                        )
                    )
                )
            )
        )
    )

    return list(
        chain(
            jumps,
            repeat(
                jumps[-1],
                num_frames - len(jumps)
            )
        )
    )


def get_jumps(filename, num_frames):
    with open(filename) as f:
        jumps = load(f)

    return list(map(lambda i: count_jumps(i, num_frames), jumps))


if __name__ == '__main__':
    print(list(map(len, get_jumps('jumps/2.json', 649))))
