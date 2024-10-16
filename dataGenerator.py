import torch


def input_data_generator(
    size=100,
    frac=0.5,
    mean=100,
    output_size=5,
    overlap=0,
):
    n = int(size * frac)
    data = torch.zeros((output_size, size))
    pattern_size = int((1 / (output_size - (output_size - 1) * overlap)) * size)
    for i in range(output_size):
        s = int(i * pattern_size - i * pattern_size * overlap)
        e = int(s + pattern_size)
        print(s, e)
        data[i][s:e] = (
            (output_size + torch.randn(pattern_size)) * mean * (1 + (i + 1) / 10)
        )
    return data
