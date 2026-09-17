// Address decoder: demuxes a single write stream and a single read stream
// to one of NUM_CLUSTERS * NUM_FIFOS FIFOs based on the address. Purely
// combinational.
module router #(
    parameter NUM_CLUSTERS = 4,
    parameter NUM_FIFOS = 2,
    parameter DATA_WIDTH = 8
) (
    input w_valid,
    input [2:0] w_addr,  // [cluster_sel(2), fifo_sel(1)]
    input [DATA_WIDTH-1:0] w_data,
    input r_valid,
    input [2:0] r_addr,
    output [NUM_FIFOS-1:0] w_en[NUM_CLUSTERS],
    output [NUM_FIFOS-1:0] r_en[NUM_CLUSTERS],
    output [DATA_WIDTH-1:0] w_data_out
);

    wire [1:0] w_cluster_sel = w_addr[2:1];
    wire       w_fifo_sel    = w_addr[0];
    wire [1:0] r_cluster_sel = r_addr[2:1];
    wire       r_fifo_sel    = r_addr[0];

    genvar c, f;
    generate
        for (c = 0; c < NUM_CLUSTERS; c++) begin : dec_gen
            for (f = 0; f < NUM_FIFOS; f++) begin : dec_fifo_gen
                assign w_en[c][f] = w_valid && w_cluster_sel == c && w_fifo_sel == f;
                assign r_en[c][f] = r_valid && r_cluster_sel == c && r_fifo_sel == f;
            end
        end
    endgenerate

    assign w_data_out = w_data;
endmodule
