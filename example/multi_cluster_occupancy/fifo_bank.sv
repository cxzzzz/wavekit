// A cluster of independently-named FIFO instances. Manually unrolled
// (instead of `generate for`) so each instance keeps a plain `fifo_N` name
// that a query pattern can match directly.
module fifo_bank #(
    parameter DEPTH = 8,
    parameter DATA_WIDTH = 8
) (
    input clk,
    input rst_n,
    input [1:0] w_en,
    input [1:0] r_en,
    input [DATA_WIDTH-1:0] data_in0,
    input [DATA_WIDTH-1:0] data_in1,
    output [DATA_WIDTH-1:0] data_out0,
    output [DATA_WIDTH-1:0] data_out1,
    output [1:0] full,
    output [1:0] empty
);

    fifo #(
        .DEPTH(DEPTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) fifo_0 (
        .clk(clk),
        .rst_n(rst_n),
        .w_en(w_en[0]),
        .r_en(r_en[0]),
        .data_in(data_in0),
        .data_out(data_out0),
        .full(full[0]),
        .empty(empty[0])
    );

    fifo #(
        .DEPTH(DEPTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) fifo_1 (
        .clk(clk),
        .rst_n(rst_n),
        .w_en(w_en[1]),
        .r_en(r_en[1]),
        .data_in(data_in1),
        .data_out(data_out1),
        .full(full[1]),
        .empty(empty[1])
    );
endmodule
