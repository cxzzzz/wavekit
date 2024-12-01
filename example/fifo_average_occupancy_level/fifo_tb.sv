module fifo_tb;
    parameter DATA_WIDTH = 8;

    reg clk, rst_n;
    reg w_en, r_en;
    reg  [DATA_WIDTH-1:0] data_in;
    wire [DATA_WIDTH-1:0] data_out;
    wire full, empty;

    fifo s_fifo (
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

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        w_en = 1'b0;
        data_in = 0;

        repeat (10) @(posedge clk);
        rst_n = 1'b1;

        repeat (2) begin
            for (int i = 0; i < 30; i++) begin
                @(posedge clk);
                #1;
                data_in = $urandom;
                w_en = (i % 2 == 0) ? 1'b1 : 1'b0;
                if (w_en & !full) begin
                    $display("[%0t] wdata = %0d", $time, data_in);
                end
            end
            #50;
        end
    end

    initial begin
        clk   = 1'b0;
        rst_n = 1'b0;
        r_en  = 1'b0;

        repeat (20) @(posedge clk);
        rst_n = 1'b1;

        repeat (2) begin
            for (int i = 0; i < 30; i++) begin
                @(posedge clk);
                #1;
                r_en = (i % 3 == 0) ? 1'b1 : 1'b0;
                if (r_en & !empty) begin
                    $display("[%0t] rdata = %0d", $time, data_out);
                end
            end
            #50;
        end

        $finish;
    end

    initial begin
        $dumpfile("fifo_tb.vcd");
        $dumpvars;
    end
endmodule
