module fifo #(
    parameter DEPTH = 8,
    DATA_WIDTH = 8,
    // BUGGY_DATA=0: data is stored exactly as written.
    // BUGGY_DATA=1: a combinational bit-flip corrupts bit 4 whenever the
    //               low 3 bits of the incoming data equal 3'b101. This is
    //               an isolated, data-dependent bug: most writes are
    //               unaffected, and each corruption is independent of the
    //               others (no cascading).
    BUGGY_DATA = 0
) (
    input clk,
    input rst_n,
    input w_en,
    input r_en,
    input [DATA_WIDTH-1:0] data_in,
    output reg [DATA_WIDTH-1:0] data_out,
    output full,
    output empty
);

    reg [$clog2(DEPTH)-1:0] w_ptr, r_ptr;
    reg [DATA_WIDTH-1:0] fifo[DEPTH];

    wire [DATA_WIDTH-1:0] data_to_store = (BUGGY_DATA && data_in[2:0] == 3'b101)
        ? (data_in ^ 8'h10)
        : data_in;

    // Set Default values on reset.
    always @(posedge clk) begin
        if (!rst_n) begin
            w_ptr <= 0;
            r_ptr <= 0;
            data_out <= 0;
        end
    end

    // To write data to FIFO
    always @(posedge clk) begin
        if (w_en & !full) begin
            fifo[w_ptr] <= data_to_store;
            w_ptr <= w_ptr + 1;
        end
    end

    // To read data from FIFO
    always @(posedge clk) begin
        if (r_en & !empty) begin
            data_out <= fifo[r_ptr];
            r_ptr <= r_ptr + 1;
        end
    end

    assign full  = ((w_ptr + 1'b1) == r_ptr);
    assign empty = (w_ptr == r_ptr);
endmodule
