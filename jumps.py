from itertools import (
    chain,
    repeat,
)
from operator import sub


def jumps(frame_indices, num_frames):
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
