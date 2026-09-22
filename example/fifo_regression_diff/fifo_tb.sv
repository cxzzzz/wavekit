module fifo_tb #(
    parameter BUGGY_DATA = 0
);
    parameter DATA_WIDTH = 8;

    reg clk, rst_n;
    reg w_en, r_en;
    reg  [DATA_WIDTH-1:0] data_in;
    wire [DATA_WIDTH-1:0] data_out;
    wire full, empty;

    fifo #(.BUGGY_DATA(BUGGY_DATA)) s_fifo (
        clk,
        rst_n,
        w_en,
        r_en,
        data_in,
        data_out,
        full,
        empty
    );

    always #5 clk = ~clk;

    integer seed = 32'hC0FFEE;

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        w_en = 1'b0;
        data_in = 0;

        repeat (10) @(posedge clk);
        rst_n = 1'b1;

        for (int i = 0; i < 300; i++) begin
            @(posedge clk);
            #1;
            w_en = ($random(seed) % 10) < 9;
            if (w_en & !full) begin
                data_in = i[DATA_WIDTH-1:0];
            end
        end
    end

    initial begin
        clk   = 1'b0;
        rst_n = 1'b0;
        r_en  = 1'b0;

        repeat (10) @(posedge clk);
        rst_n = 1'b1;

        for (int i = 0; i < 300; i++) begin
            @(posedge clk);
            #1;
            r_en = ($random(seed) % 10) < 8;
        end

        #20;
        $finish;
    end

    initial begin
        $dumpfile("fifo_tb.vcd");
        $dumpvars;
    end
endmodule
