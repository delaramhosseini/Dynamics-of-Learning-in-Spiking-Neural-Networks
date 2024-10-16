from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.spatial.distance import cosine


def format_parameters(inp_parameters, num_columns):
    """
    Formats a list of strings into a table-like string with specified number of columns.

    Args:
    parameters (list of str): List of parameter strings.
    num_columns (int): Number of columns in the table.

    Returns:
    str: Formatted table-like string.
    """
    parameters = inp_parameters + [
        "" for _ in range(num_columns - ((len(inp_parameters) % num_columns) or 4))
    ]
    max_len = max(len(param) for param in parameters) + 2  # +2 for padding
    rows = (len(parameters) + num_columns - 1) // num_columns  # ceiling division
    table_str = ""

    for r in range(rows):
        row_params = parameters[r::rows]  # Get every nth element starting from r
        row_str = " | ".join(param.ljust(max_len) for param in row_params)
        table_str += f" {row_str} \n"

    border_len = len(table_str.split("\n")[0]) - 1
    table_str = table_str

    return table_str


def Plot(
    net,
    title=None,
    parameters=[],
    ngs=[],
    sgs=[],
    scaling_factor=3,
    label_font_size=10,
    recorder_index=461,
    env_recorder_index=461,
    num_columns=4,
):
    n = len(ngs)
    fig, axd = plt.subplot_mosaic(
        (
            """
                AAAA
                AAAA
                BBBB
                BBBB
                CCCC
                CCCC
                EEGG
                FFHH
                DDDD
                IIII
                """
            if len(parameters)
            else """
                AAAA
                AAAA
                BBBB
                BBBB
                CCCC
                CCCC
                EEGG
                FFHH
                DDDD
                """
        ),
        layout="constrained",
        # "image" will contain a square image. We fine-tune the width so that
        # there is no excess horizontal or vertical margin around the image.
        figsize=(12 * scaling_factor, 6 * scaling_factor),
    )

    # Add parameters as text on the plot
    if len(parameters):
        axd["I"].axis("off")

        params_text = format_parameters(parameters, num_columns)
        axd["I"].text(
            0.5,
            0.5,
            params_text,
            fontsize=label_font_size * scaling_factor,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axd["I"].transAxes,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    sg = sgs[0]
    weight = sg[recorder_index, 0].variables["weights"]
    similarity = []
    weight_max = weight.max()
    for i in range(net.iteration):
        similarity.append(
            (weight_max * weight[i, :, 0].cpu() * (weight_max - weight[i, :, 0])).sum()
        )

    # similarity = []
    # for i in range(net.iteration):
    #     similarity.append(
    #         1
    #         - cosine(
    #             sg[recorder_index, 0].variables["weights"][i, :, 0].cpu(),
    #             sg[recorder_index, 0].variables["weights"][i, :, 1].cpu(),
    #         )
    #     )

    axd["A"].plot(similarity)
    axd["A"].set_ylabel("", fontsize=label_font_size * scaling_factor)
    axd["A"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["A"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["A"].set_xlim(0, net.iteration)
    axd["A"].set_title("Similarity", fontsize=(label_font_size + 1) * scaling_factor)
    axd["A"].set_xlabel(
        "time",
        fontsize=label_font_size * scaling_factor,
    )

    axd["B"].scatter(
        ngs[0][env_recorder_index, 0].variables["spikes"][:, 0].cpu(),
        ngs[0][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        c=(
            ngs[0][env_recorder_index, 0].variables["spikes"][:, 0].cpu()
            // ngs[0].network.inp_duration
        )
        % ngs[0].network.number_of_data,
        vmax=ngs[0].network.number_of_data + 1,
        label=f"{ngs[0].tag}",
    )
    axd["B"].set_ylabel("spikes", fontsize=label_font_size * scaling_factor)
    axd["B"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["B"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["B"].set_xlim(0, net.iteration)
    axd["B"].set_ylim(-1, ngs[0].size)

    axd["B"].set_title(
        "Input Neuron Group", fontsize=(label_font_size) * scaling_factor
    )
    axd["B"].set_xlabel(
        f"time ({ngs[0].tag})",
        fontsize=label_font_size * scaling_factor,
    )
    axd["B"].grid()

    axd["E"].plot(
        ngs[0][recorder_index, 0].variables["T"].cpu(),
    )
    axd["E"].set_ylabel("Activity", fontsize=label_font_size * scaling_factor)
    axd["E"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["E"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["E"].set_xlim(0, net.iteration)
    axd["E"].set_xlabel(
        f"time ({ngs[0].tag})",
        fontsize=label_font_size * scaling_factor,
    )

    axd["C"].scatter(
        ngs[1][env_recorder_index, 0].variables["spikes"][:, 0].cpu(),
        ngs[1][env_recorder_index, 0].variables["spikes"][:, 1].cpu(),
        c=(
            ngs[1][env_recorder_index, 0].variables["spikes"][:, 0].cpu()
            // ngs[1].network.inp_duration
        )
        % ngs[0].network.number_of_data,
        vmax=ngs[1].network.number_of_data + 1,
        label=f"{ngs[1].tag}",
    )
    axd["C"].set_ylabel("spikes", fontsize=label_font_size * scaling_factor)
    axd["C"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["C"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["C"].set_xlim(0, net.iteration)
    axd["C"].set_ylim(-1, ngs[1].size)
    axd["C"].set_title(
        "Output Neuron Group", fontsize=(label_font_size) * scaling_factor
    )
    axd["C"].set_xlabel(
        f"time ({ngs[1].tag})",
        fontsize=label_font_size * scaling_factor,
    )
    axd["C"].grid()
    axd["G"].plot(
        ngs[1][recorder_index, 0].variables["T"].cpu(),
    )
    axd["G"].set_ylabel("Activity", fontsize=label_font_size * scaling_factor)
    axd["G"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["G"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["G"].set_xlim(0, net.iteration)
    axd["G"].set_xlabel(
        f"time ({ngs[1].tag})",
        fontsize=label_font_size * scaling_factor,
    )
    axd["F"].plot(
        sg[recorder_index, 0].variables["weights"][:, :, 0].cpu(),
    )
    axd["F"].set_ylabel("Weight", fontsize=label_font_size * scaling_factor)
    axd["F"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["F"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["F"].set_xlim(0, net.iteration)
    axd["F"].set_xlabel(
        "time (output-0)",
        fontsize=label_font_size * scaling_factor,
    )
    axd["H"].plot(
        sg[recorder_index, 0].variables["weights"][:, :, 1].cpu(),
    )
    axd["H"].set_ylabel("Weight", fontsize=label_font_size * scaling_factor)
    axd["H"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["H"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["H"].set_xlim(0, net.iteration)
    axd["H"].set_xlabel(
        "time (output-1)",
        fontsize=label_font_size * scaling_factor,
    )
    fig.tight_layout()
    fig.show()

    for j in range(ngs[1][recorder_index, 0].variables["v"].shape[1]):
        axd["D"].plot(
            ngs[1][recorder_index, 0].variables["v"][:, j].cpu(), label=f"neuron-{j}"
        )
    axd["D"].set_ylabel("voltage", fontsize=label_font_size * scaling_factor)
    axd["D"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["D"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["D"].set_xlim(0, net.iteration)
    axd["D"].set_title(
        "Voltage of output layer", fontsize=(label_font_size + 1) * scaling_factor
    )
    axd["D"].set_xlabel(
        "time",
        fontsize=label_font_size * scaling_factor,
    )
    axd["D"].legend(
        loc="best",
        fontsize=label_font_size * scaling_factor / 2,
    )
