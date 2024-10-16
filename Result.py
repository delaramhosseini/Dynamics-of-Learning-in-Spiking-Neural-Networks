import pymonntorch as pmt


class Result(pmt.Behavior):
    def initialize(self, ng):
        ng.spike_counter = ng.vector(0)
        ng.network.winner = 0
        self.print_details = self.parameter("print_details", True)
        ng.result = [[-1] for _ in range(ng.network.number_of_data)]

    def forward(self, ng):
        ng.spike_counter += ng.spikes.byte()
        if ng.network.iteration % ng.network.inp_duration == 0:
            inx = (ng.network.current_inp_idx - 1) % ng.network.number_of_data
            result = ng.spike_counter == ng.spike_counter.max()
            self.print_details and print(
                f"for pattern: {inx}, result_neuron: {((result == True).nonzero(as_tuple=True))}"
            )
            ng.result[inx] = (result == True).nonzero(as_tuple=True)
            self.print_details and print(ng.spike_counter)
            ng.spike_counter = ng.vector(0)
