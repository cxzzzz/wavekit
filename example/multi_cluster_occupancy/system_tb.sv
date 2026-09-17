module system_tb;
    parameter DATA_WIDTH = 8;
    parameter NUM_CLUSTERS = 4;
    parameter NUM_FIFOS = 2;

    reg clk, rst_n;
    reg w_valid, r_valid;
    reg [2:0] w_addr, r_addr;
    reg [DATA_WIDTH-1:0] w_data;

    wire [NUM_FIFOS-1:0] w_en[NUM_CLUSTERS];
    wire [NUM_FIFOS-1:0] r_en[NUM_CLUSTERS];
    wire [DATA_WIDTH-1:0] w_data_routed;

    router #(
        .NUM_CLUSTERS(NUM_CLUSTERS),
        .NUM_FIFOS(NUM_FIFOS),
        .DATA_WIDTH(DATA_WIDTH)
    ) router_inst (
        .w_valid(w_valid),
        .w_addr(w_addr),
        .w_data(w_data),
        .r_valid(r_valid),
        .r_addr(r_addr),
        .w_en(w_en),
        .r_en(r_en),
        .w_data_out(w_data_routed)
    );

    genvar c;
    generate
        for (c = 0; c < NUM_CLUSTERS; c++) begin : cluster_gen
            fifo_bank #(.DATA_WIDTH(DATA_WIDTH)) cluster_inst (
                clk, rst_n, w_en[c], r_en[c],
                w_data_routed, w_data_routed,
                /* data_out0 */, /* data_out1 */,
                /* full */, /* empty */
            );
        end
    endgenerate

    always #5 clk = ~clk;

    // Skewed write-address distribution: cluster 0 gets hit twice as often
    // as the others, so the per-cluster/per-fifo occupancy report has a
    // real "busiest" answer instead of four statistically-identical clusters.
    function [2:0] skewed_addr();
        int roll;
        roll = $urandom % 10;
        if (roll < 4) skewed_addr = {2'd0, $urandom % 2 == 0};
        else if (roll < 6) skewed_addr = {2'd1, $urandom % 2 == 0};
        else if (roll < 8) skewed_addr = {2'd2, $urandom % 2 == 0};
        else skewed_addr = {2'd3, $urandom % 2 == 0};
    endfunction

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        w_valid = 1'b0;
        w_addr = 0;
        w_data = 0;

        repeat (10) @(posedge clk);
        rst_n = 1'b1;

        for (int i = 0; i < 400; i++) begin
            @(posedge clk);
            #1;
            w_data = $urandom;
            w_addr = skewed_addr();
            w_valid = ($urandom % 100) < 60;
        end

        #50;
    end

    initial begin
        clk   = 1'b0;
        rst_n = 1'b0;
        r_valid = 1'b0;
        r_addr = 0;

        repeat (20) @(posedge clk);
        rst_n = 1'b1;

        for (int i = 0; i < 400; i++) begin
            @(posedge clk);
            #1;
            r_addr = $urandom % 8;
            r_valid = ($urandom % 100) < 40;
        end

        #50;
        $finish;
    end

    initial begin
        $dumpfile("system_tb.vcd");
        $dumpvars;
    end
endmodule
